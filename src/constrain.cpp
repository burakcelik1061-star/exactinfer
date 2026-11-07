#include "constrain.h"

#include <cerrno>
#include <cstdlib>
#include <stdexcept>

#include "json.h"
#include "unicode_tables.h"

namespace el {

namespace {

constexpr uint8_t kMaxWs = 4;  // ardışık boşluk sınırı

// Görünmez / yön değiştiren karakterler — bekçinin listesiyle aynı (Cc, Cf, satır/paragraf ayırıcı).
bool hidden(uint32_t c) {
    return c < 0x20 || (c >= 0x7F && c <= 0x9F) || c == 0xAD || c == 0x61C || c == 0x180E ||
           (c >= 0x200B && c <= 0x200F) || (c >= 0x2028 && c <= 0x202E) || (c >= 0x2060 && c <= 0x2064) ||
           (c >= 0x2066 && c <= 0x206F) || c == 0xFEFF || (c >= 0xFFF9 && c <= 0xFFFB) ||
           (c >= 0xE0000 && c <= 0xE007F);
}

// Unicode White_Space
bool white(uint32_t c) {
    return c == 0x20 || (c >= 0x09 && c <= 0x0D) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// [lo, hi] aralığı tablodaki bir aralıkla kesişiyor mu (tablo sıralı, çakışmasız)
template <size_t N>
bool overlaps(const CpRange (&t)[N], uint32_t lo, uint32_t hi) {
    size_t a = 0, b = N;
    while (a < b) {  // hi >= lo olan ilk aralık
        const size_t m = (a + b) / 2;
        if (t[m].hi < lo) a = m + 1; else b = m;
    }
    return a < N && t[a].lo <= hi;
}

bool alnum(uint32_t cp) { return overlaps(kLetter, cp, cp) || overlaps(kNumber, cp, cp); }

// Geçerli UTF-8 → kod noktaları
std::vector<uint32_t> code_points(std::string_view s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const auto b = uint8_t(s[i]);
        const int n = b < 0x80 ? 0 : b < 0xE0 ? 1 : b < 0xF0 ? 2 : 3;
        uint32_t cp = n == 0 ? b : b & (0x3F >> n);
        for (int k = 1; k <= n; ++k) cp = cp << 6 | (uint8_t(s[i + size_t(k)]) & 0x3F);
        out.push_back(cp);
        i += size_t(n) + 1;
    }
    return out;
}

// Windows aygıt adı mı (bekçi validate::is_device_name ile aynı): ilk noktaya kadarki kök, sondaki boşluklar
// atılıp küçük harfe çevrilerek con/prn/aux/nul/conin$/conout$/clock$, comN/lptN (N rakam ya da ¹²³).
bool device_name(std::string_view part) {
    std::vector<uint32_t> s = code_points(part.substr(0, part.find('.')));
    while (!s.empty() && white(s.back())) s.pop_back();
    for (uint32_t& c : s)
        if (c >= 'A' && c <= 'Z') c += 32;
    auto is = [&](const char* w) {
        size_t k = 0;
        for (; w[k]; ++k)
            if (k >= s.size() || s[k] != uint32_t(uint8_t(w[k]))) return false;
        return k == s.size();
    };
    if (is("con") || is("prn") || is("aux") || is("nul") || is("conin$") || is("conout$") || is("clock$")) return true;
    if (s.size() != 4 || !((s[0] == 'c' && s[1] == 'o' && s[2] == 'm') || (s[0] == 'l' && s[1] == 'p' && s[2] == 't')))
        return false;
    return (s[3] >= '0' && s[3] <= '9') || s[3] == 0xB9 || s[3] == 0xB2 || s[3] == 0xB3;
}

bool host_char(uint32_t cp) { return alnum(cp) || cp == '.' || cp == '-' || cp == '[' || cp == ']' || cp == ':'; }

bool is_ws(uint8_t c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
bool is_digit(uint8_t c) { return c >= '0' && c <= '9'; }

void put_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += char(cp);
    else if (cp < 0x800) { out += char(0xC0 | cp >> 6); out += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        out += char(0xE0 | cp >> 12); out += char(0x80 | (cp >> 6 & 0x3F)); out += char(0x80 | (cp & 0x3F));
    } else {
        out += char(0xF0 | cp >> 18); out += char(0x80 | (cp >> 12 & 0x3F));
        out += char(0x80 | (cp >> 6 & 0x3F)); out += char(0x80 | (cp & 0x3F));
    }
}

bool starts_with(const std::string& s, std::string_view p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }

// Şemadaki adlar ve enum değerleri kaçışsız yazılabilmeli (eşleştirici metinde kaçışa yalnız değerlerde izin verir).
void check_plain(const std::string& s, const char* what) {
    if (s.empty()) throw std::runtime_error(std::string("şema: boş ") + what);
    for (unsigned char c : s)
        if (c < 0x20 || c == '"' || c == '\\') throw std::runtime_error(std::string("şema: ") + what + " kaçış gerektiriyor: " + s);
}

int64_t to_i64(const Json& j, const char* what) {
    if (!j.is(Json::Type::Number) || j.str.find_first_of(".eE") != std::string::npos)
        throw std::runtime_error(std::string("şema: ") + what + " tamsayı olmalı");
    errno = 0;
    const long long v = std::strtoll(j.str.c_str(), nullptr, 10);
    if (errno == ERANGE) throw std::runtime_error(std::string("şema: ") + what + " aralık dışı");
    return int64_t(v);
}

}  // namespace

ToolSchema ToolSchema::from_json(std::string_view text) { return from_value(parse_json(text)); }

ToolSchema ToolSchema::from_value(const Json& root) {
    if (!root.is(Json::Type::Array)) throw std::runtime_error("şema: araç listesi (dizi) bekleniyordu");
    ToolSchema out;
    for (const Json& t : root.arr) {
        const Json* f = t.get("function");
        const Json* name = f ? f->get("name") : nullptr;
        if (!name || !name->is(Json::Type::String)) throw std::runtime_error("şema: function.name eksik");
        ToolSpec ts;
        ts.name = name->str;
        check_plain(ts.name, "araç adı");
        for (const ToolSpec& o : out.tools)
            if (o.name == ts.name) throw std::runtime_error("şema: araç iki kez: " + ts.name);
        const Json* params = f->get("parameters");
        const Json* props = params ? params->get("properties") : nullptr;
        if (props && !props->is(Json::Type::Object)) throw std::runtime_error("şema: properties nesne olmalı");
        if (props) {
            if (props->obj.size() > 64) throw std::runtime_error("şema: en çok 64 parametre");
            for (const auto& [pn, pj] : props->obj) {
                ParamSpec p;
                p.name = pn;
                check_plain(p.name, "parametre adı");
                const Json* ty = pj.get("type");
                const std::string type = ty && ty->is(Json::Type::String) ? ty->str : "";
                if (type == "string") p.type = ParamSpec::Type::String;
                else if (type == "integer") p.type = ParamSpec::Type::Integer;
                else if (type == "number") p.type = ParamSpec::Type::Number;
                else if (type == "boolean") p.type = ParamSpec::Type::Boolean;
                else throw std::runtime_error("şema: " + ts.name + "." + pn + ": desteklenmeyen tür '" + type + "'");
                if (const Json* e = pj.get("enum")) {
                    if (p.type != ParamSpec::Type::String || !e->is(Json::Type::Array) || e->arr.empty())
                        throw std::runtime_error("şema: " + pn + ": enum yalnız metin dizisi");
                    for (const Json& v : e->arr) {
                        if (!v.is(Json::Type::String)) throw std::runtime_error("şema: enum değeri metin olmalı");
                        check_plain(v.str, "enum değeri");
                        p.values.push_back(v.str);
                    }
                    p.kind = StrKind::Enum;
                } else if (const Json* fm = pj.get("format"); fm && fm->is(Json::Type::String)) {
                    if (fm->str == "multiline") p.kind = StrKind::Multiline;
                    else if (fm->str == "name") p.kind = StrKind::Name;
                    else if (fm->str == "path") p.kind = StrKind::Path;
                    else if (fm->str == "uri") p.kind = StrKind::Uri;
                }
                if (const Json* m = pj.get("maxLength")) {
                    const int64_t v = to_i64(*m, "maxLength");
                    if (v < 1 || v > 100000) throw std::runtime_error("şema: maxLength 1..100000");
                    p.max_len = uint32_t(v);
                }
                if (p.kind == StrKind::Path && p.max_len < 4) throw std::runtime_error("şema: yol için maxLength en az 4");
                if (p.kind == StrKind::Uri && p.max_len < 9) throw std::runtime_error("şema: adres için maxLength en az 9");
                const Json* mn = pj.get("minimum");
                const Json* mx = pj.get("maximum");
                if (p.type == ParamSpec::Type::Integer) {
                    if (mn) p.imin = to_i64(*mn, "minimum");
                    if (mx) p.imax = to_i64(*mx, "maximum");
                    if (p.imin > p.imax) throw std::runtime_error("şema: minimum > maximum");
                } else if (p.type == ParamSpec::Type::Number) {
                    if (mn) p.nmin = mn->num;
                    if (mx) p.nmax = mx->num;
                    if (!(p.nmin <= p.nmax)) throw std::runtime_error("şema: minimum > maximum");
                }
                ts.params.push_back(std::move(p));
            }
        }
        if (const Json* req = params ? params->get("required") : nullptr) {
            if (!req->is(Json::Type::Array)) throw std::runtime_error("şema: required dizi olmalı");
            for (const Json& r : req->arr) {
                size_t k = 0;
                while (k < ts.params.size() && ts.params[k].name != r.str) ++k;
                if (!r.is(Json::Type::String) || k == ts.params.size())
                    throw std::runtime_error("şema: required'da tanımsız parametre: " + r.str);
                ts.params[k].required = true;
                ts.required_mask |= uint64_t(1) << k;
            }
        }
        out.tools.push_back(std::move(ts));
    }
    if (out.tools.empty()) throw std::runtime_error("şema: araç yok");
    return out;
}

// ---------------------------------------------------------------- eşleştirici

uint64_t CallMatcher::unseen() const {
    const size_t n = sc_->tools[size_t(tool_)].params.size();
    const uint64_t all = n == 64 ? ~uint64_t(0) : (uint64_t(1) << n) - 1;
    return all & ~seen_;
}

bool CallMatcher::feed(std::string_view bytes) {
    for (char c : bytes)
        if (!byte(uint8_t(c))) return false;
    return true;
}

bool CallMatcher::byte(uint8_t c) {
    switch (st_) {
        case S::Str: return str_byte(c);
        case S::Int: return int_byte(c);
        case S::Num: return num_byte(c);
        case S::Lit:
            if (c != uint8_t(lit_[lit_pos_])) return false;
            if (!lit_[++lit_pos_]) { st_ = lit_next_; nws_ = 0; }
            return true;
        default: break;
    }
    if (is_ws(c)) return ++nws_ <= kMaxWs;
    nws_ = 0;
    auto start_str = [&](Role r) {
        role_ = r;
        buf_.clear();
        ncp_ = last_cp_ = 0;
        comp_ = 0;
        esc_ = false;
        u8n_ = u8want_ = 0;
        st_ = S::Str;
        return true;
    };
    auto lit = [&](const char* l, S next) {
        lit_ = l;
        lit_pos_ = 1;
        lit_next_ = next;
        st_ = S::Lit;
        return true;
    };
    switch (st_) {
        case S::TopOpen: return c == '{' && (st_ = S::NameKey, true);
        case S::NameKey: return c == '"' && lit("\"name\"", S::Colon1);
        case S::Colon1: return c == ':' && (st_ = S::NameVal, true);
        case S::NameVal: return c == '"' && start_str(Role::ToolName);
        case S::AfterName: return c == ',' && (st_ = S::ArgsKey, true);
        case S::ArgsKey: return c == '"' && lit("\"arguments\"", S::Colon2);
        case S::Colon2: return c == ':' && (st_ = S::ArgsOpen, true);
        case S::ArgsOpen: return c == '{' && (st_ = S::ArgFirst, true);
        case S::ArgFirst:
            if (c == '}') return sc_->tools[size_t(tool_)].required_mask == 0 && (st_ = S::TopClose, true);
            return c == '"' && unseen() != 0 && start_str(Role::Key);
        case S::ArgColon: return c == ':' && (st_ = S::ArgValue, true);
        case S::ArgValue: return begin_value(c);
        case S::AfterValue: {
            if (c == ',') return unseen() != 0 && (st_ = S::ArgNext, true);
            const uint64_t req = sc_->tools[size_t(tool_)].required_mask;
            return c == '}' && (seen_ & req) == req && (st_ = S::TopClose, true);
        }
        case S::ArgNext: return c == '"' && start_str(Role::Key);
        case S::TopClose: return c == '}' && (st_ = S::Done, true);
        default: return false;  // Done: yalnız boşluk
    }
}

bool CallMatcher::begin_value(uint8_t c) {
    const ParamSpec& p = param();
    switch (p.type) {
        case ParamSpec::Type::String:
            if (c != '"') return false;
            role_ = Role::Value;
            buf_.clear();
            ncp_ = last_cp_ = 0;
            comp_ = 0;
            esc_ = false;
            u8n_ = u8want_ = 0;
            st_ = S::Str;
            return true;
        case ParamSpec::Type::Integer:
            st_ = S::Int;
            buf_.clear();
            return int_byte(c);
        case ParamSpec::Type::Number:
            st_ = S::Num;
            buf_.clear();
            numst_ = 0;
            return num_byte(c);
        case ParamSpec::Type::Boolean:
            if (c != 't' && c != 'f') return false;
            lit_ = c == 't' ? "true" : "false";
            lit_pos_ = 1;
            lit_next_ = S::AfterValue;
            st_ = S::Lit;
            return true;
    }
    return false;
}

// ---- metin

bool CallMatcher::prefix_ok() const {
    std::string cur = buf_;
    cur.append(reinterpret_cast<const char*>(u8_), u8n_);
    auto any = [&](auto&& names) {
        for (const std::string& n : names)
            if (starts_with(n, cur)) return true;
        return false;
    };
    if (role_ == Role::ToolName) {
        for (const ToolSpec& t : sc_->tools)
            if (starts_with(t.name, cur)) return true;
        return false;
    }
    if (role_ == Role::Key) {
        const auto& ps = sc_->tools[size_t(tool_)].params;
        const uint64_t u = unseen();
        for (size_t k = 0; k < ps.size(); ++k)
            if ((u >> k & 1) && starts_with(ps[k].name, cur)) return true;
        return false;
    }
    return any(param().values);
}

bool CallMatcher::value_cp_ok(uint32_t cp, bool escaped) const {
    const ParamSpec& p = param();
    const uint32_t pos = ncp_, last = last_cp_;
    if (pos >= p.max_len) return false;
    if (hidden(cp) && !(p.kind == StrKind::Multiline && escaped && (cp == '\n' || cp == '\t'))) return false;
    if (cp >= 0xD800 && cp <= 0xDFFF) return false;
    switch (p.kind) {
        case StrKind::Text:
        case StrKind::Multiline:
        case StrKind::Enum: return true;  // Enum: önek denetimi ayrıca
        case StrKind::Name:
            if (cp == ' ') return pos > 0 && pos + 1 < p.max_len;  // ad boşlukla başlamaz/bitmez
            return alnum(cp) || cp == '_' || cp == '-' || cp == '.';
        case StrKind::Path: {
            if (pos == 0) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
            if (pos == 1) return cp == ':';
            if (pos == 2) return cp == '\\' || cp == '/';
            if (cp == ':' || cp == '*' || cp == '?' || cp == '"' || cp == '<' || cp == '>' || cp == '|') return false;
            if (white(cp) && cp != ' ') return false;  // bekçi uçtaki Unicode boşlukları kırpar: hiç olmasın
            const std::string_view comp = std::string_view(buf_).substr(comp_);
            if (cp == '\\' || cp == '/') return last != '.' && last != ' ' && !device_name(comp);  // bileşen kapanışı
            if (cp == '.' && comp.find('.') == std::string_view::npos && device_name(comp))
                return false;  // kök "con" olarak donardı: bu bileşen artık hiç geçerli olamaz
            if (pos + 1 >= p.max_len) {  // son karakter: bileşen burada biter
                if (cp == '.' || cp == ' ') return false;
                std::string c(comp);
                put_utf8(c, cp);
                return !device_name(c);
            }
            return true;
        }
        case StrKind::Uri: {
            if (pos < 8) return cp == uint32_t("https://"[pos]);
            if (white(cp)) return false;
            if (buf_.find_first_of("/?#", 8) != std::string::npos) return true;  // yol/sorgu kısmı
            if (cp == '/' || cp == '?' || cp == '#') return pos > 8;              // ana makine boş olamaz
            return host_char(cp) && !(pos == 8 && cp == ':');                     // kullanıcı:parola@ yok
        }
    }
    return false;
}

bool CallMatcher::add_cp(uint32_t cp, bool escaped) {
    if (role_ == Role::Value && !value_cp_ok(cp, escaped)) return false;
    if (role_ != Role::Value && (escaped || hidden(cp))) return false;
    put_utf8(buf_, cp);
    if (role_ == Role::Value && param().kind == StrKind::Path && (cp == '\\' || cp == '/')) comp_ = buf_.size();
    ++ncp_;
    last_cp_ = cp;
    const bool pref = role_ != Role::Value || param().kind == StrKind::Enum;
    return !pref || prefix_ok();
}

bool CallMatcher::partial_ok() const {
    // Gelen baytlardan olası kod noktası aralığı
    const uint8_t total = uint8_t(u8want_ + 1);
    const uint32_t lead_bits = total == 2 ? 0x1F : total == 3 ? 0x0F : 0x07;
    uint32_t v = u8_[0] & lead_bits;
    for (uint8_t k = 1; k < u8n_; ++k) v = v << 6 | (u8_[k] & 0x3F);
    const int rem = 6 * (total - u8n_);
    uint32_t lo = v << rem, hi = lo | ((uint32_t(1) << rem) - 1);
    const uint32_t minv = total == 2 ? 0x80 : total == 3 ? 0x800 : 0x10000;
    if (lo < minv) lo = minv;
    if (hi > 0x10FFFF) hi = 0x10FFFF;
    if (lo > hi) return false;
    if (role_ != Role::Value || param().kind == StrKind::Enum) return prefix_ok();
    const ParamSpec& p = param();
    if (ncp_ >= p.max_len || (p.kind == StrKind::Path && ncp_ < 3) || (p.kind == StrKind::Uri && ncp_ < 8))
        return false;  // bu konumlarda yalnız ASCII
    if (p.kind == StrKind::Name || (p.kind == StrKind::Uri && buf_.find_first_of("/?#", 8) == std::string::npos))
        return overlaps(kLetter, lo, hi) || overlaps(kNumber, lo, hi);  // ad / ana makine: ASCII dışı yalnız harf-rakam
    for (uint32_t cp = lo; cp <= hi; ++cp) {  // gizli/boşluk aralıkları küçük: ilk izinli karakter hemen bulunur
        if (cp == 0xD800) { cp = 0xDFFF; continue; }
        if (value_cp_ok(cp, false)) return true;
    }
    return false;
}

bool CallMatcher::str_byte(uint8_t c) {
    if (u8want_) {  // UTF-8 devam baytı
        uint8_t lo = 0x80, hi = 0xBF;
        if (u8n_ == 1) {
            if (u8_[0] == 0xE0) lo = 0xA0;
            if (u8_[0] == 0xED) hi = 0x9F;
            if (u8_[0] == 0xF0) lo = 0x90;
            if (u8_[0] == 0xF4) hi = 0x8F;
        }
        if (c < lo || c > hi) return false;
        u8_[u8n_++] = c;
        if (u8n_ < u8want_ + 1) return partial_ok();
        const uint8_t total = u8n_;
        uint32_t cp = u8_[0] & (total == 2 ? 0x1F : total == 3 ? 0x0F : 0x07);
        for (uint8_t k = 1; k < total; ++k) cp = cp << 6 | (u8_[k] & 0x3F);
        u8n_ = u8want_ = 0;
        return add_cp(cp, false);
    }
    if (esc_) {
        esc_ = false;
        uint32_t cp;
        switch (c) {
            case '"': cp = '"'; break;
            case '\\': cp = '\\'; break;
            case '/': cp = '/'; break;
            case 'n': cp = '\n'; break;
            case 't': cp = '\t'; break;
            default: return false;  // \u, \b, \f, \r yok
        }
        return add_cp(cp, true);
    }
    if (c == '"') return str_close();
    if (c == '\\') {
        if (role_ != Role::Value || param().kind == StrKind::Enum) return false;
        for (uint32_t cp : {uint32_t('"'), uint32_t('\\'), uint32_t('/'), uint32_t('\n'), uint32_t('\t')})
            if (value_cp_ok(cp, true)) { esc_ = true; return true; }
        return false;
    }
    if (c < 0x20) return false;
    if (c < 0x80) return add_cp(c, false);
    if (c >= 0xC2 && c <= 0xDF) u8want_ = 1;
    else if (c >= 0xE0 && c <= 0xEF) u8want_ = 2;
    else if (c >= 0xF0 && c <= 0xF4) u8want_ = 3;
    else return false;
    u8_[0] = c;
    u8n_ = 1;
    if (!partial_ok()) { u8want_ = 0; return false; }
    return true;
}

bool CallMatcher::str_close() {
    switch (role_) {
        case Role::ToolName:
            for (size_t k = 0; k < sc_->tools.size(); ++k)
                if (sc_->tools[k].name == buf_) { tool_ = int(k); seen_ = 0; st_ = S::AfterName; return true; }
            return false;
        case Role::Key: {
            const auto& ps = sc_->tools[size_t(tool_)].params;
            const uint64_t u = unseen();
            for (size_t k = 0; k < ps.size(); ++k)
                if ((u >> k & 1) && ps[k].name == buf_) {
                    param_ = int(k);
                    seen_ |= uint64_t(1) << k;
                    st_ = S::ArgColon;
                    return true;
                }
            return false;
        }
        case Role::Value: {
            const ParamSpec& p = param();
            bool ok = true;
            switch (p.kind) {
                case StrKind::Enum: {
                    ok = false;
                    for (const std::string& v : p.values) ok |= v == buf_;
                    break;
                }
                case StrKind::Name: ok = ncp_ > 0 && last_cp_ != ' '; break;
                case StrKind::Path:
                    ok = ncp_ >= 3 && last_cp_ != '.' && last_cp_ != ' ' && !device_name(std::string_view(buf_).substr(comp_));
                    break;
                case StrKind::Uri: ok = ncp_ > 8; break;
                default: break;
            }
            if (ok) st_ = S::AfterValue;
            return ok;
        }
    }
    return false;
}

// ---- tamsayı: "-"? rakamlar; her önek aralıkta bir değere tamamlanabilmeli

namespace {

// Büyüklük (mutlak değer) aralığı: neg ise değer = −m, değilse değer = m
bool mag_range(const ParamSpec& p, bool neg, uint64_t& a, uint64_t& b) {
    if (!neg) {
        if (p.imax < 0) return false;
        a = p.imin > 0 ? uint64_t(p.imin) : 0;
        b = uint64_t(p.imax);
    } else {
        if (p.imin >= 0) return false;  // "-0" yazılmaz: negatif büyüklük en az 1
        a = p.imax >= 0 ? 1 : uint64_t(-(p.imax + 1)) + 1;
        b = uint64_t(-(p.imin + 1)) + 1;
    }
    return a <= b;
}

bool int_feasible(const ParamSpec& p, const std::string& s) {
    const bool neg = !s.empty() && s[0] == '-';
    const std::string d = neg ? s.substr(1) : s;
    uint64_t a, b;
    if (!mag_range(p, neg, a, b)) return false;
    if (d.empty()) return true;
    uint64_t P = 0;
    for (char c : d) P = P * 10 + uint64_t(c - '0');  // en çok 19 hane: taşmaz
    uint64_t pw = 1;
    for (size_t k = 0; d.size() + k <= 19; ++k) {
        if (d == "0" && k > 0) break;
        if (P > UINT64_MAX / pw) break;
        const uint64_t lo = P * pw, hi = lo > UINT64_MAX - (pw - 1) ? UINT64_MAX : lo + (pw - 1);
        if (hi >= a && lo <= b) return true;
        if (pw > UINT64_MAX / 10) break;
        pw *= 10;
    }
    return false;
}

bool int_final(const ParamSpec& p, const std::string& s) {
    const bool neg = !s.empty() && s[0] == '-';
    const std::string d = neg ? s.substr(1) : s;
    if (d.empty()) return false;
    uint64_t a, b, P = 0;
    if (!mag_range(p, neg, a, b)) return false;
    for (char c : d) P = P * 10 + uint64_t(c - '0');
    return P >= a && P <= b;
}

}  // namespace

bool CallMatcher::int_byte(uint8_t c) {
    const ParamSpec& p = param();
    const bool neg = !buf_.empty() && buf_[0] == '-';
    const size_t nd = buf_.size() - (neg ? 1 : 0);
    if (c == '-' && buf_.empty()) {
        buf_ = "-";
        return int_feasible(p, buf_);
    }
    if (is_digit(c)) {
        if (nd == 1 && buf_.back() == '0') return false;  // baştaki sıfır
        if (neg && nd == 0 && c == '0') return false;     // "-0"
        if (nd >= 19) return false;
        buf_ += char(c);
        return int_feasible(p, buf_);
    }
    if (is_ws(c) || c == ',' || c == '}') {
        if (!int_final(p, buf_)) return false;
        st_ = S::AfterValue;
        nws_ = 0;
        return byte(c);
    }
    return false;
}

// ---- sayı (üssüz): -? (0 | [1-9][0-9]*) (. [0-9]+)?

namespace {

bool num_feasible(const ParamSpec& p, const std::string& s, bool final) {
    const bool neg = !s.empty() && s[0] == '-';
    const std::string body = neg ? s.substr(1) : s;
    const size_t dot = body.find('.');
    const std::string ip = body.substr(0, dot), fp = dot == std::string::npos ? "" : body.substr(dot + 1);
    // izinli büyüklük aralığı [a, b]
    double a, b;
    if (!neg) { if (p.nmax < 0) return false; a = p.nmin > 0 ? p.nmin : 0; b = p.nmax; }
    else { if (p.nmin > 0) return false; a = p.nmax < 0 ? -p.nmax : 0; b = -p.nmin; }
    if (ip.empty()) return true;
    const double I = std::strtod(ip.c_str(), nullptr);
    if (final) {
        const double v = std::strtod(body.c_str(), nullptr);
        return v >= a && v <= b;
    }
    if (dot != std::string::npos) {
        const double lo = std::strtod(body.c_str(), nullptr);
        double step = 1;
        for (size_t k = 0; k < fp.size(); ++k) step /= 10;
        return lo + step >= a && lo <= b;
    }
    double pw = 1;
    for (size_t k = 0; ip.size() + k <= 15; ++k) {  // daha çok tam hane ya da kesir
        if (ip == "0" && k > 0) break;
        const double lo = I * pw, hi = (I + 1) * pw;
        if (hi >= a && lo <= b) return true;
        pw *= 10;
    }
    return false;
}

}  // namespace

bool CallMatcher::num_byte(uint8_t c) {
    const ParamSpec& p = param();
    // DFA: 0 başlangıç, 1 '-' sonrası, 2 tam kısım "0", 3 tam kısım, 4 '.' sonrası, 5 kesir
    if (is_ws(c) || c == ',' || c == '}') {
        if (!(numst_ == 2 || numst_ == 3 || numst_ == 5) || !num_feasible(p, buf_, true)) return false;
        st_ = S::AfterValue;
        nws_ = 0;
        return byte(c);
    }
    if (buf_.size() >= 24) return false;
    uint8_t next;
    if (c == '-' && numst_ == 0) next = 1;
    else if (is_digit(c)) {
        if (numst_ == 0 || numst_ == 1) next = c == '0' ? 2 : 3;
        else if (numst_ == 3) next = 3;
        else if (numst_ == 4 || numst_ == 5) next = 5;
        else return false;  // "0" ardından rakam
    } else if (c == '.' && (numst_ == 2 || numst_ == 3)) next = 4;
    else return false;
    buf_ += char(c);
    numst_ = next;
    if (numst_ == 3 && buf_.size() - (buf_[0] == '-') > 15) return false;
    return num_feasible(p, buf_, false);
}

}  // namespace el
