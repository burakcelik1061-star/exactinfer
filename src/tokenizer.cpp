#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "gguf.h"
#include "unicode_tables.h"

namespace el {

namespace {

// ---------------------------------------------------------------- UTF-8

void put_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += char(cp);
    } else if (cp < 0x800) {
        s += char(0xC0 | (cp >> 6));
        s += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += char(0xE0 | (cp >> 12));
        s += char(0x80 | ((cp >> 6) & 0x3F));
        s += char(0x80 | (cp & 0x3F));
    } else {
        s += char(0xF0 | (cp >> 18));
        s += char(0x80 | ((cp >> 12) & 0x3F));
        s += char(0x80 | ((cp >> 6) & 0x3F));
        s += char(0x80 | (cp & 0x3F));
    }
}

// Geçersiz UTF-8 reddedilir (aşırı uzun kodlama, vekil çiftler, 0x10FFFF üstü dahil).
struct Cp { uint32_t cp; uint32_t off; uint32_t len; };

std::vector<Cp> decode_utf8(std::string_view s) {
    std::vector<Cp> out;
    out.reserve(s.size());
    const auto* p = reinterpret_cast<const uint8_t*>(s.data());
    size_t n = s.size(), i = 0;
    while (i < n) {
        uint8_t c = p[i];
        uint32_t cp, len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else throw std::invalid_argument("gecersiz UTF-8");
        if (i + len > n) throw std::invalid_argument("gecersiz UTF-8 (kesik)");
        for (uint32_t k = 1; k < len; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) throw std::invalid_argument("gecersiz UTF-8");
            cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        static const uint32_t kMin[5] = {0, 0, 0x80, 0x800, 0x10000};
        if (cp < kMin[len] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            throw std::invalid_argument("gecersiz UTF-8 kod noktasi");
        out.push_back({cp, uint32_t(i), len});
        i += len;
    }
    return out;
}

// ---------------------------------------------------------------- karakter sınıfları

template <size_t N>
bool in_ranges(const CpRange (&t)[N], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < t[mid].lo) hi = mid;
        else if (cp > t[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

bool is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
    return in_ranges(kLetter, cp);
}
bool is_number(uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(kNumber, cp);
}
// Unicode White_Space
bool is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}
bool is_nl(uint32_t cp) { return cp == '\r' || cp == '\n'; }

uint32_t lower_ascii(uint32_t cp) { return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp; }

// qwen2 ön-tokenleştirici düzenli ifadesinin elle uygulanışı:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ |
//   \s+(?!\S) | \s+
// Seçenekler sırayla denenir; ilk eşleşen kazanır (düzenli ifade motorunun sol-öncelikli davranışı).
// Dönen değer: cps[i]'den başlayan eşleşmenin uzunluğu (kod noktası sayısı, >= 1).
size_t match_at(const std::vector<Cp>& cps, size_t i) {
    const size_t n = cps.size();
    auto cp = [&](size_t k) -> uint32_t { return k < n ? cps[k].cp : UINT32_MAX; };
    auto letter = [&](size_t k) { return k < n && is_letter(cps[k].cp); };
    auto number = [&](size_t k) { return k < n && is_number(cps[k].cp); };
    auto space = [&](size_t k) { return k < n && is_space(cps[k].cp); };
    auto other = [&](size_t k) { return k < n && !is_space(cps[k].cp) && !is_letter(cps[k].cp) && !is_number(cps[k].cp); };

    const uint32_t c = cp(i);

    // 1) kısaltmalar
    if (c == '\'' && i + 1 < n) {
        uint32_t a = lower_ascii(cp(i + 1));
        uint32_t b = lower_ascii(cp(i + 2));
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
    // 2) [^\r\n\p{L}\p{N}]?\p{L}+
    {
        size_t k = i;
        if (!is_nl(c) && !letter(i) && !number(i) && letter(i + 1)) k = i + 1;
        if (letter(k)) {
            while (letter(k)) ++k;
            return k - i;
        }
    }
    // 3) \p{N}
    if (number(i)) return 1;
    // 4) ' '?[^\s\p{L}\p{N}]+[\r\n]*
    {
        size_t k = (c == ' ' && other(i + 1)) ? i + 1 : i;
        if (other(k)) {
            while (other(k)) ++k;
            while (k < n && is_nl(cps[k].cp)) ++k;
            return k - i;
        }
    }
    // Buradan sonrası boşluk koşusuyla başlar
    size_t end = i;
    while (space(end)) ++end;
    if (end == i) return 1;  // ulaşılamaz (her kod noktası bir sınıfa düşer), güvenlik için
    // 5) \s*[\r\n]+ : koşudaki son satır sonuna kadar
    for (size_t k = end; k > i; --k)
        if (is_nl(cps[k - 1].cp)) return k - i;
    // 6) \s+(?!\S) : metin sonu değilse son boşluğu bir sonraki parçaya bırak
    if (end == n) return end - i;
    if (end - i >= 2) return end - i - 1;
    // 7) \s+
    return end - i;
}

// GPT-2 bytes_to_unicode
void build_byte_map(std::string (&byte_char)[256], std::unordered_map<uint32_t, uint8_t>& char_byte) {
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        uint32_t cp = printable ? uint32_t(b) : uint32_t(256 + extra++);
        byte_char[b].clear();
        put_utf8(byte_char[b], cp);
        char_byte[cp] = uint8_t(b);
    }
}

}  // namespace

// ---------------------------------------------------------------- kurulum

Tokenizer::Tokenizer(const Gguf& g) {
    const std::string& model = g.get_str("tokenizer.ggml.model");
    if (model != "gpt2") throw std::runtime_error("desteklenmeyen tokenizer modeli: " + model);
    const Value* pre = g.find("tokenizer.ggml.pre");
    if (pre && pre->s != "qwen2") throw std::runtime_error("desteklenmeyen on-tokenlestirici: " + pre->s);

    const Value* tokens = g.find("tokenizer.ggml.tokens");
    const Value* types = g.find("tokenizer.ggml.token_type");
    const Value* merges = g.find("tokenizer.ggml.merges");
    if (!tokens || tokens->as.empty()) throw std::runtime_error("tokenizer.ggml.tokens yok");
    if (!merges || merges->as.empty()) throw std::runtime_error("tokenizer.ggml.merges yok");

    vocab_ = tokens->as;
    type_.assign(vocab_.size(), 1);
    if (types) {
        if (types->ai.size() != vocab_.size()) throw std::runtime_error("token_type boyutu uyusmuyor");
        for (size_t k = 0; k < vocab_.size(); ++k) type_[k] = int32_t(types->ai[k]);
    }
    lookup_.reserve(vocab_.size());
    for (size_t k = 0; k < vocab_.size(); ++k) lookup_.emplace(vocab_[k], int32_t(k));

    merge_rank_.reserve(merges->as.size());
    for (size_t k = 0; k < merges->as.size(); ++k) {
        const std::string& m = merges->as[k];
        if (m.find(' ', 1) == std::string::npos) throw std::runtime_error("bozuk merge satiri");
        merge_rank_.emplace(m, int32_t(k));
    }

    for (size_t k = 0; k < vocab_.size(); ++k)
        if ((type_[k] == 3 || type_[k] == 4) && !vocab_[k].empty()) specials_.push_back(int32_t(k));
    std::stable_sort(specials_.begin(), specials_.end(),
                     [&](int32_t a, int32_t b) { return vocab_[a].size() > vocab_[b].size(); });
    for (int32_t id : specials_) special_first_[uint8_t(vocab_[id][0])] = true;

    eos_ = int32_t(g.get_int_or("tokenizer.ggml.eos_token_id", -1));
    build_byte_map(byte_char_, char_byte_);
}

bool Tokenizer::is_special(int32_t id) const {
    return id >= 0 && size_t(id) < type_.size() && (type_[id] == 3 || type_[id] == 4);
}

// ---------------------------------------------------------------- kodlama

std::vector<std::string> Tokenizer::pretokenize(std::string_view utf8) const {
    std::vector<Cp> cps = decode_utf8(utf8);
    std::vector<std::string> out;
    for (size_t i = 0; i < cps.size();) {
        size_t len = match_at(cps, i);
        uint32_t a = cps[i].off, b = cps[i + len - 1].off + cps[i + len - 1].len;
        out.emplace_back(utf8.substr(a, b - a));
        i += len;
    }
    return out;
}

void Tokenizer::bpe(std::string_view piece, std::vector<int32_t>& out) const {
    std::vector<std::string> sym;
    sym.reserve(piece.size());
    for (unsigned char b : piece) sym.push_back(byte_char_[b]);

    std::string key;
    while (sym.size() > 1) {
        int32_t best = INT32_MAX;
        size_t at = 0;
        for (size_t k = 0; k + 1 < sym.size(); ++k) {
            key.assign(sym[k]).append(1, ' ').append(sym[k + 1]);
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best) { best = it->second; at = k; }
        }
        if (best == INT32_MAX) break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + std::ptrdiff_t(at) + 1);
    }
    for (const std::string& s : sym) {
        auto it = lookup_.find(s);
        if (it != lookup_.end()) { out.push_back(it->second); continue; }
        // Sözlükte yoksa bayt bayt (bayt düzeyinde sözlükte her bayt vardır)
        for (uint8_t b : token_bytes_raw(s)) {
            auto jt = lookup_.find(byte_char_[b]);
            if (jt == lookup_.end()) throw std::runtime_error("bayt tokeni sozlukte yok");
            out.push_back(jt->second);
        }
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special) const {
    std::vector<int32_t> out;
    size_t start = 0;  // henüz işlenmemiş düz metin parçasının başı
    auto flush = [&](size_t end) {
        if (end <= start) return;
        std::string_view part = text.substr(start, end - start);
        for (const std::string& w : pretokenize(part)) bpe(w, out);
    };
    size_t i = 0;
    while (i < text.size()) {
        int32_t hit = -1;
        if (parse_special && special_first_[uint8_t(text[i])]) {
            for (int32_t id : specials_) {
                const std::string& s = vocab_[id];
                if (text.compare(i, s.size(), s) == 0) { hit = id; break; }  // uzundan kısaya: en uzun eşleşme
            }
        }
        if (hit < 0) { ++i; continue; }
        flush(i);
        out.push_back(hit);
        i += vocab_[hit].size();
        start = i;
    }
    flush(text.size());
    return out;
}

// ---------------------------------------------------------------- çözme

std::string Tokenizer::token_bytes_raw(const std::string& mapped) const {
    std::string out;
    std::vector<Cp> cps = decode_utf8(mapped);
    for (const Cp& c : cps) {
        auto it = char_byte_.find(c.cp);
        if (it == char_byte_.end()) {  // eşlemede olmayan karakter: olduğu gibi
            put_utf8(out, c.cp);
        } else {
            out += char(it->second);
        }
    }
    return out;
}

std::string Tokenizer::token_bytes(int32_t id) const {
    if (id < 0 || size_t(id) >= vocab_.size()) throw std::out_of_range("token id sinir disi");
    if (is_special(id)) return vocab_[id];
    return token_bytes_raw(vocab_[id]);
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) out += token_bytes(id);
    return out;
}

}  // namespace el
