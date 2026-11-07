// el — komut satırı.
//
//   el info <model.gguf>
//   el tokenize <model.gguf> [--no-special] [--lines]   stdin metin → JSON id dizisi
//        --lines: her satır onaltılık (hex) kodlanmış bir metin; her biri için bir JSON satırı
//   el generate <model.gguf> [-n 32] [--top 5] [--ctx N] [--tools araclar.json]   stdin istem → JSON (açgözlü)
//        --tools: kısıtlı çözümleme; <tool_call> ile </tool_call> arası yalnızca şemaya uyan JSON olabilir
//   el serve <model.gguf> [--ctx 8192]   kalıcı sunucu: stdin/stdout JSON satırları (asistan bağlantısı; cmd_serve)
//   el constrain-test <araclar.json> [--fuzz N] [--seed S]
//        stdin'deki her onaltılık satırı eşleştiriciye verir → complete | prefix | invalid <bayt konumu>
//        --fuzz: N rastgele yürüyüşle "kabul edilen her önek tamamlanabilir" kuralını sınar
//   el bench <model.gguf> [--pp 512] [--tg 128] [--reps 3]  istem işleme / üretim hızı (token/sn)
//   el dequant <model.gguf> <tensör> <ilk satır> <satır sayısı>  satırları fp32 (ham ikili) olarak stdout'a yazar
//   Ortak: -f <dosya> (girdiyi stdin yerine UTF-8 dosyadan oku), --backend auto|cpu|cuda, --threads T
#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>
#include <random>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "constrain.h"
#include "gguf.h"
#include "json.h"
#include "model.h"
#include "ops.h"
#include "tokenizer.h"

using namespace el;

namespace {

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "kullanim:\n"
                 "  el info <model.gguf>\n"
                 "  el tokenize <model.gguf> [--no-special] [--lines] [-f dosya]\n"
                 "  el generate <model.gguf> [-n 32] [--top 5] [--ctx N] [--tools araclar.json] [-f dosya]\n"
                 "  el serve <model.gguf> [--ctx 8192]\n"
                 "  el constrain-test <araclar.json> [--fuzz N] [--seed S]\n"
                 "  el bench <model.gguf> [--pp 512] [--tg 128] [--reps 3]\n"
                 "  el dequant <model.gguf> <tensor> <ilk satir> <satir sayisi>\n"
                 "  ortak: --backend auto|cpu|cuda  --threads T\n");
    std::exit(2);
}

struct Opts {
    bool special = true, lines = false;
    int n_predict = 32, top = 5, ctx = 0;
    int pp = 512, tg = 128, reps = 3, fuzz = 0;
    uint32_t seed = 1;
    std::string backend = "auto", prompt_file, tools_file;
};

std::string read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("dosya acilamadi: " + path);
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    if (s.size() >= 3 && s.compare(0, 3, "\xEF\xBB\xBF") == 0) s.erase(0, 3);  // UTF-8 BOM
    if (s.size() >= 2 && (s.compare(0, 2, "\xFF\xFE") == 0 || s.compare(0, 2, "\xFE\xFF") == 0))
        throw std::runtime_error("dosya UTF-16 (PowerShell 5.1 '>' boyle yazar); UTF-8 olarak kaydedin: " + path);
    return s;
}

std::string read_input(const Opts& o) {
    if (!o.prompt_file.empty()) return read_file(o.prompt_file);
    _setmode(_fileno(stdin), _O_BINARY);
    return std::string(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
}

std::string hex(const std::string& s) {
    static const char* d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) { out += d[c >> 4]; out += d[c & 15]; }
    return out;
}

std::string unhex(const std::string& s) {
    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    if (s.size() % 2) throw std::invalid_argument("hex uzunlugu tek");
    std::string out;
    for (size_t i = 0; i < s.size(); i += 2) {
        int a = val(s[i]), b = val(s[i + 1]);
        if (a < 0 || b < 0) throw std::invalid_argument("gecersiz hex");
        out += char(a * 16 + b);
    }
    return out;
}

std::string ids_json(const std::vector<int32_t>& ids) {
    std::string s = "[";
    for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
    return s + "]";
}

int parse_int(const char* s, int lo, int hi, const char* what) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end || v < lo || v > hi) {
        std::fprintf(stderr, "%s gecersiz: %s (%d..%d)\n", what, s, lo, hi);
        std::exit(2);
    }
    return int(v);
}

std::unique_ptr<Backend> make_backend(const Gguf& g, const std::string& which, int ctx) {
    if (which == "cpu") return make_cpu_backend(g, ctx);
    if (which == "cuda") return make_cuda_backend(g, ctx);
    std::string why;
    if (cuda_available(&why)) return make_cuda_backend(g, ctx);
    std::fprintf(stderr, "bilgi: CUDA yok (%s), CPU kullaniliyor\n", why.c_str());
    return make_cpu_backend(g, ctx);
}

int cmd_info(const Gguf& g) {
    std::printf("GGUF v%u, %zu metadata, %zu tensor\n", g.version(), g.meta().size(), g.tensors().size());
    std::vector<std::string> keys;
    for (auto& [k, v] : g.meta()) keys.push_back(k);
    std::sort(keys.begin(), keys.end());
    for (auto& k : keys) {
        const Value& v = g.meta().at(k);
        std::printf("  %-48s ", k.c_str());
        switch (v.type) {
            case ValueType::STRING: std::printf("\"%.60s\"\n", v.s.c_str()); break;
            case ValueType::F32: case ValueType::F64: std::printf("%g\n", v.f); break;
            case ValueType::ARRAY:
                std::printf("[dizi, %zu oge]\n", v.as.size() + v.ai.size() + v.af.size());
                break;
            default: std::printf("%lld\n", static_cast<long long>(v.i));
        }
    }
    uint64_t total = 0, bytes = 0;
    for (auto& t : g.tensors()) { total += t.numel; bytes += t.nbytes; }
    for (auto& t : g.tensors()) {
        if (t.name.rfind("blk.", 0) == 0 && t.name.rfind("blk.0.", 0) != 0) continue;
        std::string dims;
        for (auto n : t.ne) dims += (dims.empty() ? "" : " x ") + std::to_string(n);
        std::printf("  %-32s %-5s [%s]\n", t.name.c_str(), type_name(t.type), dims.c_str());
    }
    std::printf("toplam: %.2fM parametre, %.1f MB (blk.0 disindaki bloklar gizli)\n", total / 1e6, bytes / 1048576.0);
    return 0;
}

int cmd_tokenize(const Gguf& g, const Opts& o) {
    Tokenizer tok(g);
    std::string in = read_input(o);
    if (!o.lines) {
        std::printf("%s\n", ids_json(tok.encode(in, o.special)).c_str());
        return 0;
    }
    size_t start = 0;
    while (start < in.size()) {
        size_t end = in.find('\n', start);
        if (end == std::string::npos) end = in.size();
        std::string line = in.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        start = end + 1;
        std::printf("%s\n", ids_json(tok.encode(unhex(line), o.special)).c_str());
    }
    return 0;
}

bool is_eog(const Tokenizer& tok, int32_t id) {
    if (id == tok.eos()) return true;
    if (!tok.is_special(id)) return false;
    const std::string s = tok.token_bytes(id);
    return s == "<|endoftext|>" || s == "<|im_end|>";
}

int32_t special_id(const Tokenizer& tok, const char* text) {
    const std::vector<int32_t> ids = tok.encode(text, true);
    if (ids.size() != 1 || !tok.is_special(ids[0]))
        throw std::runtime_error(std::string("modelde ozel token yok: ") + text);
    return ids[0];
}

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

// Kısıtlı çözümleme durumu (generate ve serve ortak). <tool_call> üretilince eşleştirici devreye girer; çağrı içinde
// yalnız şemaya uyan (ve geçerli bir çağrıya tamamlanabilen) tokenler, çağrı bitince yalnız </tool_call> seçilebilir.
// Serbest metinde </tool_call> ve etiketlerin düz metinle yazımı yasak (kısıt atlatılamasın).
struct Constraint {
    const Tokenizer& tok;
    const ToolSchema& schema;
    const int32_t open, close;
    std::optional<CallMatcher> call, cand;
    std::string text_tail;  // serbest metnin son baytları
    int n_constrained = 0, n_forced = 0, n_calls = 0;
    bool cur_altered = false;            // açık çağrıda kısıt modelin seçimini değiştirdi mi
    std::vector<int32_t> altered_calls;  // bekçiye / asistana bildirilir: bu çağrılar onaya yükseltilir

    Constraint(const Tokenizer& t, const ToolSchema& s)
        : tok(t), schema(s), open(special_id(t, "<tool_call>")), close(special_id(t, "</tool_call>")) {}

    bool allowed(int32_t id) {
        if (call) {
            if (id == close) return call->complete();
            if (tok.is_special(id)) return false;
            const std::string b = tok.token_bytes(id);
            if (b.empty()) return false;
            cand = *call;
            return cand->feed(b);
        }
        if (id == close) return false;
        if (tok.is_special(id)) return true;
        const std::string t = text_tail + tok.token_bytes(id);
        return t.find("<tool_call>") == std::string::npos && t.find("</tool_call>") == std::string::npos;
    }

    // order'ın ilk `sorted` ögesi logit sırasında; gerekirse sıralama genişletilir (64 → 1024 → tümü).
    // İzinli ilk adayın sırasını döndürür ve durumu ilerletir.
    template <class Cmp>
    int pick(std::vector<int32_t>& order, int sorted, Cmp by_logit) {
        const int V = int(order.size());
        int j = 0;
        while (!allowed(order[size_t(j)])) {
            if (++j == sorted) {
                if (sorted == V) throw std::runtime_error("kisitli cozumleme: izinli token kalmadi");
                const int more = std::min(sorted < 64 ? 64 : sorted < 1024 ? 1024 : V, V);
                std::partial_sort(order.begin() + sorted, order.begin() + more, order.end(), by_logit);
                sorted = more;
            }
        }
        commit(order[size_t(j)], j > 0);
        return j;
    }

    void commit(int32_t next, bool forced) {
        n_forced += forced;
        if (call) {
            ++n_constrained;
            cur_altered |= forced;
            if (next == close) {
                call.reset();
                if (cur_altered) altered_calls.push_back(n_calls - 1);
            } else {
                call = std::move(cand);
            }
        } else if (next == open) {
            call.emplace(schema);
            cur_altered = false;
            text_tail.clear();
            ++n_calls;
        } else if (!tok.is_special(next)) {
            text_tail += tok.token_bytes(next);
            if (text_tail.size() > 16) text_tail.erase(0, text_tail.size() - 16);
        }
    }

    // Üretim bitti: açık kalan çağrı da (kısıt değiştirdiyse) bildirilir.
    std::string summary() {
        if (call && cur_altered) altered_calls.push_back(n_calls - 1);
        char buf[160];
        std::snprintf(buf, sizeof buf, ",\"calls\":%d,\"constrained\":%d,\"forced\":%d,\"open_call\":%s", n_calls,
                      n_constrained, n_forced, call ? "true" : "false");
        return buf + (",\"altered_calls\":" + ids_json(altered_calls));
    }
};

int cmd_generate(const Gguf& g, const Opts& o) {
    Tokenizer tok(g);
    const std::string prompt = read_input(o);
    const std::vector<int32_t> ids = tok.encode(prompt, true);
    if (ids.empty()) throw std::runtime_error("istem bos");
    const int ctx = o.ctx ? o.ctx : int(ids.size()) + o.n_predict;
    if (int(ids.size()) >= ctx) throw std::runtime_error("istem baglam penceresinden uzun");
    auto t_load = clk::now();
    std::unique_ptr<Backend> m = make_backend(g, o.backend, ctx);
    const double ms_load = ms_since(t_load);
    const int V = m->cfg().n_vocab;
    std::vector<float> logits(static_cast<size_t>(V));

    auto t0 = clk::now();
    m->eval(ids.data(), int(ids.size()), 0, logits.data());
    const double ms_p = ms_since(t0);

    std::optional<ToolSchema> schema;
    std::optional<Constraint> con;
    if (!o.tools_file.empty()) {
        schema = ToolSchema::from_json(read_file(o.tools_file));
        con.emplace(tok, *schema);
    }

    auto t1 = clk::now();
    std::string out = "{\"prompt_ids\":" + ids_json(ids) + ",\"tokens\":[";
    int pos = int(ids.size()), produced = 0;
    std::vector<int32_t> order(static_cast<size_t>(V));
    for (int step = 0; step < o.n_predict; ++step) {
        // log-softmax (double)
        double mx = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        for (float l : logits) sum += std::exp(double(l) - mx);
        const double lse = mx + std::log(sum);

        std::iota(order.begin(), order.end(), 0);
        const int k = std::max(1, std::min(o.top, V));
        auto by_logit = [&](int32_t a, int32_t b) {
            return logits[size_t(a)] > logits[size_t(b)] || (logits[size_t(a)] == logits[size_t(b)] && a < b);
        };
        std::partial_sort(order.begin(), order.begin() + k, order.end(), by_logit);
        const int j = con ? con->pick(order, k, by_logit) : 0;
        const int32_t next = order[size_t(j)];
        if (is_eog(tok, next)) break;

        char buf[64];
        std::snprintf(buf, sizeof buf, "%.6f", double(logits[size_t(next)]) - lse);
        out += std::string(produced ? "," : "") + "{\"id\":" + std::to_string(next) + ",\"hex\":\"" +
               hex(tok.token_bytes(next)) + "\",\"logprob\":" + buf + (j > 0 ? ",\"forced\":true" : "") + ",\"top\":[";
        for (int t = 0; t < k; ++t) {
            const int32_t id = order[size_t(t)];
            std::snprintf(buf, sizeof buf, "%.6f", double(logits[size_t(id)]) - lse);
            out += std::string(t ? "," : "") + "{\"id\":" + std::to_string(id) + ",\"hex\":\"" +
                   hex(tok.token_bytes(id)) + "\",\"logprob\":" + buf + "}";
        }
        out += "]}";
        ++produced;
        if (step + 1 == o.n_predict || pos >= ctx) break;
        m->eval(&next, 1, pos++, logits.data());
    }
    const double ms_g = ms_since(t1);
    char tail[200];
    std::snprintf(tail, sizeof tail,
                  "],\"backend\":\"%s\",\"ms_load\":%.1f,\"ms_prompt\":%.1f,\"ms_gen\":%.1f,\"threads\":%d",
                  m->name(), ms_load, ms_p, ms_g, threads());
    out += tail;
    if (con) out += con->summary();
    std::printf("%s}\n", out.c_str());
    return 0;
}

std::string json_escape(std::string_view s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
        else out += char(c);
    }
    return out + "\"";
}

// Örnekleme (temperature > 0): top_k → top_p (T=1 olasılıklarıyla) → sıcaklık, llama.cpp/Ollama zinciri sırasında.
// Yalnız serbest metinde kullanılır; araç çağrısının içi her zaman açgözlüdür (parametreler kararlı, `altered`
// "kısıt modelin en olası tokenini reddetti" anlamında kalır). Açgözlü seçim tekrar döngüsüne girebiliyor
// (ölçüldü: "fıkra anlat" 1024 token aynı cümle); Qwen3 açgözlü seçimi önermiyor.
struct Sampler {
    float temp = 0.0f, top_p = 1.0f;
    int top_k = 0;  // 0: sınırsız
    std::mt19937 rng;

    // order: logit sırasında ilk `sorted` öge hazır (en az top_k). allow: izinli aday mı (kısıt). İzinli aday
    // yoksa -1 döner (çağıran açgözlü kısıtlı seçime düşer).
    template <class Allow>
    int32_t pick(const std::vector<float>& logits, std::vector<int32_t>& order, int sorted, Allow allow) {
        const int k = std::min(top_k > 0 ? top_k : sorted, sorted);
        std::vector<int32_t> cand;
        cand.reserve(size_t(k));
        for (int i = 0; i < k; ++i)
            if (allow(order[size_t(i)])) cand.push_back(order[size_t(i)]);
        if (cand.empty()) return -1;
        const float lmax = logits[size_t(cand[0])];
        std::vector<double> p(cand.size());
        double sum = 0;
        for (size_t i = 0; i < cand.size(); ++i) sum += p[i] = std::exp(double(logits[size_t(cand[i])] - lmax));
        size_t keep = 0;
        for (double cum = 0; keep < cand.size() && (keep == 0 || cum < top_p * sum); ++keep) cum += p[keep];
        for (size_t i = 0; i < keep; ++i) p[i] = std::exp(double(logits[size_t(cand[i])] - lmax) / temp);
        std::discrete_distribution<size_t> d(p.begin(), p.begin() + std::ptrdiff_t(keep));
        return cand[d(rng)];
    }
};

// Tekrar cezası (DRY, "don't repeat yourself"; llama.cpp'deki örnekleyicinin karşılığı): yanıtın son L tokeni daha
// önce aynı sırayla geçmişse, o geçişten sonra gelen tokenin logit'i mult * base^(L - allowed) kadar düşer (L >= allowed);
// eşleşme ayırıcı içeren tokenleri (\n : " *) aşmaz. Yalnız bu yanıtın serbest metnine uygulanır: istemden (araç
// sonucu, kullanıcının sözleri) aynen alıntı cezalanmaz, araç çağrısının içi zaten açgözlü ve cezasız. Gerekçe: 0.2
// sıcaklıkta "fıkra anlat" üç denemenin birinde 1024 token aynı kalıba girdi.
struct Dry {
    float mult = 0.0f, base = 1.75f;
    int allowed = 2;
    static constexpr int kMaxMatch = 32;  // ceza bu uzunlukta zaten logit ölçeğinin çok üstünde
    std::vector<int32_t> x;               // yanıtın serbest metin tokenleri
    std::vector<char> brk;                // x[i] ayırıcı mı
    int hits = 0;                         // ceza uygulanan adım sayısı (yanıtta bildirilir)

    void push(const Tokenizer& tok, int32_t id) {
        const std::string b = tok.token_bytes(id);
        x.push_back(id);
        brk.push_back(b.find_first_of("\n:\"*") != std::string::npos);
    }

    void apply(std::vector<float>& logits) {
        const int n = int(x.size());
        if (mult <= 0 || n <= allowed || brk[size_t(n - 1)]) return;
        std::unordered_map<int32_t, int> longest;  // aday token → en uzun eşleşme
        for (int j = n - 1; j >= 1; --j) {        // x[j], eski geçişin ardından gelen token; sonek x[..j-1] ile x[..n-1]
            int L = 0;
            while (L < kMaxMatch && j - 1 - L >= 0 && !brk[size_t(n - 1 - L)] && x[size_t(j - 1 - L)] == x[size_t(n - 1 - L)])
                ++L;
            if (L >= allowed) {
                int& b = longest[x[size_t(j)]];
                b = std::max(b, L);
            }
        }
        for (const auto& [id, L] : longest) logits[size_t(id)] -= mult * std::pow(base, float(L - allowed));
        hits += !longest.empty();
    }
};

// Kalıcı sunucu: model bir kez yüklenir; stdin'den JSON satırı istek, stdout'a JSON satırı yanıt (asistan böyle
// bağlanır, her turda yükleme beklemez). Varsayılan açgözlü seçim; "temperature" verilirse serbest metinde örnekleme.
//   {"id"?, "prompt": "<sohbet şablonu uygulanmış metin>", "n": 256, "tools": [araç şeması]?,
//    "temperature"?: 0..2, "top_k"?: 1..1000 (varsayılan 20), "top_p"?: (0,1] (varsayılan 0.8), "seed"?: tamsayı,
//    "dry_multiplier"?: 0..5 (varsayılan 0 = kapalı), "dry_base"?: 1..4 (1.75), "dry_allowed_length"?: 1..20 (2)}
//   → {"ok":true, "hex": üretilen baytlar, "tokens", "stop": "eog"|"length"|"ctx", "prompt_tokens", "reused",
//      "ms_prompt", "ms_gen" (+ tools varsa: calls, constrained, forced, open_call, altered_calls; DRY açıksa dry_hits)}
//   Hatalı istek: {"ok":false,"error":...}; süreç sürer. {"op":"quit"} ya da stdin'in kapanması çıkış.
// Önek önbelleği: KV'de duran token dizisiyle yeni istemin ortak öneki yeniden hesaplanmaz (sistem istemi ve geçmiş).
int cmd_serve(const Gguf& g, const Opts& o) {
    Tokenizer tok(g);
    const int ctx = o.ctx ? o.ctx : 8192;
    auto t_load = clk::now();
    std::unique_ptr<Backend> m = make_backend(g, o.backend, ctx);
    const int V = m->cfg().n_vocab;
    std::vector<float> logits(static_cast<size_t>(V));
    std::vector<int32_t> cached, order(static_cast<size_t>(V));  // cached: KV'de konum konum duran tokenler
    std::mt19937 seeds(std::random_device{}());                  // "seed" verilmeyen istekler için
    std::printf("{\"ready\":true,\"backend\":\"%s\",\"n_ctx\":%d,\"ms_load\":%.1f}\n", m->name(), ctx, ms_since(t_load));
    std::fflush(stdout);
    _setmode(_fileno(stdin), _O_BINARY);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::string resp, id;
        try {
            if (line.size() > (size_t(16) << 20)) throw std::runtime_error("istek cok uzun (16 MB)");
            const Json req = parse_json(line);
            if (!req.is(Json::Type::Object)) throw std::runtime_error("istek bir JSON nesnesi olmali");
            if (const Json* i = req.get("id")) {
                if (i->is(Json::Type::String)) id = json_escape(i->str);
                else if (i->is(Json::Type::Number)) id = i->str;
            }
            if (const Json* op = req.get("op"); op && op->is(Json::Type::String) && op->str == "quit") break;
            const Json* pr = req.get("prompt");
            if (!pr || !pr->is(Json::Type::String)) throw std::runtime_error("'prompt' (metin) eksik");
            int n = 256;
            if (const Json* nj = req.get("n")) {
                if (!nj->is(Json::Type::Number) || nj->num != std::floor(nj->num) || nj->num < 1 || nj->num > 32768)
                    throw std::runtime_error("'n' 1..32768 tamsayi olmali");
                n = int(nj->num);
            }
            auto num = [&](const char* key, double lo, double hi, double def, bool integer) {
                const Json* j = req.get(key);
                if (!j) return def;
                if (!j->is(Json::Type::Number) || j->num < lo || j->num > hi || (integer && j->num != std::floor(j->num)))
                    throw std::runtime_error(std::string("'") + key + "' gecersiz");
                return j->num;
            };
            Sampler smp;
            smp.temp = float(num("temperature", 0, 2, 0, false));
            smp.top_k = int(num("top_k", 1, 1000, 20, true));
            smp.top_p = float(num("top_p", 1e-6, 1, 0.8, false));
            smp.rng.seed(req.get("seed") ? uint32_t(num("seed", 0, 4294967295.0, 0, true)) : seeds());
            const bool sample = smp.temp > 0;
            Dry dry;
            dry.mult = float(num("dry_multiplier", 0, 5, 0, false));
            dry.base = float(num("dry_base", 1, 4, 1.75, false));
            dry.allowed = int(num("dry_allowed_length", 1, 20, 2, true));
            std::optional<ToolSchema> schema;
            std::optional<Constraint> con;
            if (const Json* t = req.get("tools"); t && !t->is(Json::Type::Null)) {
                schema = ToolSchema::from_value(*t);
                con.emplace(tok, *schema);
            }
            const std::vector<int32_t> ids = tok.encode(pr->str, true);
            if (ids.empty()) throw std::runtime_error("istem bos");
            if (int(ids.size()) >= ctx) throw std::runtime_error("istem baglam penceresinden uzun");

            size_t common = 0;
            while (common < cached.size() && common < ids.size() && cached[common] == ids[common]) ++common;
            const size_t start = std::min(common, ids.size() - 1);  // son istem tokeninin logit'i her zaman gerekli
            auto t0 = clk::now();
            cached.resize(start);  // eval yarıda kalırsa yalnız [0, start) güvenilir (ortak önek)
            m->eval(ids.data() + start, int(ids.size() - start), int(start), logits.data());
            cached = ids;
            const double ms_p = ms_since(t0);

            auto t1 = clk::now();
            std::string text;
            const char* stop = "length";
            int pos = int(ids.size()), produced = 0;
            for (int step = 0; step < n; ++step) {
                int32_t next = -1;
                const bool free_text = !(con && con->call);
                if (free_text) dry.apply(logits);
                if (con || sample) {
                    std::iota(order.begin(), order.end(), 0);
                    auto by_logit = [&](int32_t a, int32_t b) {
                        return logits[size_t(a)] > logits[size_t(b)] || (logits[size_t(a)] == logits[size_t(b)] && a < b);
                    };
                    const int sorted = std::min(std::max(8, smp.top_k > 0 ? smp.top_k : 64), V);
                    std::partial_sort(order.begin(), order.begin() + sorted, order.end(), by_logit);
                    if (sample && !(con && con->call)) {  // serbest metin: örnekle (kısıt varsa izinlilerden)
                        next = smp.pick(logits, order, sorted, [&](int32_t t) { return !con || con->allowed(t); });
                        if (next >= 0 && con) {
                            const bool forced = !con->allowed(order[0]);  // serbest metinde yan etkisiz
                            con->commit(next, forced);
                        }
                    }
                    if (next < 0) next = con ? order[size_t(con->pick(order, sorted, by_logit))] : order[0];
                } else {
                    next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
                }
                if (is_eog(tok, next)) { stop = "eog"; break; }
                if (free_text && dry.mult > 0 && !tok.is_special(next)) dry.push(tok, next);
                text += tok.token_bytes(next);
                ++produced;
                if (step + 1 == n) break;
                if (pos >= ctx) { stop = "ctx"; break; }
                m->eval(&next, 1, pos, logits.data());  // yalnız `pos` konumunu yazar; [0, pos) etkilenmez
                cached.push_back(next);
                ++pos;
            }
            char buf[256];
            std::snprintf(buf, sizeof buf,
                          ",\"tokens\":%d,\"stop\":\"%s\",\"prompt_tokens\":%zu,\"reused\":%zu,\"ms_prompt\":%.1f,\"ms_gen\":%.1f",
                          produced, stop, ids.size(), start, ms_p, ms_since(t1));
            resp = "{\"ok\":true,\"hex\":\"" + hex(text) + "\"" + buf + (con ? con->summary() : "") +
                   (dry.mult > 0 ? ",\"dry_hits\":" + std::to_string(dry.hits) : "") + "}";
        } catch (const std::exception& e) {
            resp = std::string("{\"ok\":false,\"error\":") + json_escape(e.what()) + "}";
        }
        if (!id.empty()) resp.insert(resp.size() - 1, ",\"id\":" + id);
        std::printf("%s\n", resp.c_str());
        std::fflush(stdout);
    }
    return 0;
}

// Eşleştirici sınaması: model olmadan, yalnız şema ve bayt dizileriyle.
int cmd_constrain_test(const std::string& tools_path, const Opts& o) {
    const ToolSchema sc = ToolSchema::from_json(read_file(tools_path));
    if (o.fuzz) {
        // Rastgele yürüyüş: her adımda 256 baytın hangilerinin kabul edildiği bulunur. Kabul edilen bir önekten
        // sonra hiçbir bayt kabul edilmiyor ve çağrı bitmemişse çıkmaz var demektir (değişmez kural bozuk).
        std::mt19937 rng(o.seed);
        long long steps = 0, done = 0;
        for (int w = 0; w < o.fuzz; ++w) {
            CallMatcher m(sc);
            std::string text;
            const unsigned bias = rng() % 4;  // 0: düzgün rastgele, 1-3: kapanışa doğru artan eğilim
            for (int step = 0; step < 20000 && !m.complete(); ++step, ++steps) {
                std::vector<uint8_t> ok;
                for (int b = 0; b < 256; ++b) {
                    CallMatcher t = m;
                    if (t.feed(std::string(1, char(b)))) ok.push_back(uint8_t(b));
                }
                if (ok.empty()) {
                    std::printf("{\"dead_end\":\"%s\"}\n", hex(text).c_str());
                    return 1;
                }
                uint8_t c = ok[rng() % ok.size()];
                for (uint8_t q : {uint8_t('"'), uint8_t('}'), uint8_t(',')})
                    if (bias && rng() % 8 < bias && std::find(ok.begin(), ok.end(), q) != ok.end()) c = q;
                m.feed(std::string(1, char(c)));
                text += char(c);
            }
            if (!m.complete()) {
                std::printf("{\"unfinished\":\"%s\"}\n", hex(text).c_str());
                return 1;
            }
            ++done;
            std::printf("%s\n", hex(text).c_str());
        }
        std::fprintf(stderr, "yuruyus %d, tamamlanan %lld, bayt %lld\n", o.fuzz, done, steps);
        return 0;
    }
    const std::string in = read_input(o);
    size_t start = 0;
    while (start < in.size()) {
        size_t end = in.find('\n', start);
        if (end == std::string::npos) end = in.size();
        std::string line = in.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        start = end + 1;
        const std::string bytes = unhex(line);
        CallMatcher m(sc);
        size_t i = 0;
        while (i < bytes.size() && m.feed(bytes.substr(i, 1))) ++i;
        if (i < bytes.size()) std::printf("invalid %zu\n", i);
        else std::printf("%s\n", m.complete() ? "complete" : "prefix");
    }
    return 0;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int cmd_bench(const Gguf& g, const Opts& o) {
    const int ctx = std::max(o.pp, o.tg) + 1;
    auto t_load = clk::now();
    std::unique_ptr<Backend> m = make_backend(g, o.backend, ctx);
    const double ms_load = ms_since(t_load);
    const int V = m->cfg().n_vocab;
    std::vector<float> logits(static_cast<size_t>(V));
    std::vector<int32_t> ids(size_t(std::max(o.pp, o.tg)));
    uint32_t s = 12345;  // sabit tohumlu sözde rastgele tokenler (llama-bench gibi)
    for (int32_t& t : ids) { s = s * 1664525u + 1013904223u; t = int32_t((s >> 8) % uint32_t(std::min(V, 150000))); }

    m->eval(ids.data(), std::min(8, int(ids.size())), 0, logits.data());  // ısınma
    std::vector<double> pp_ms, tg_ms;
    for (int r = 0; r < o.reps; ++r) {
        auto t0 = clk::now();
        m->eval(ids.data(), o.pp, 0, logits.data());
        pp_ms.push_back(ms_since(t0));
        auto t1 = clk::now();
        for (int i = 0; i < o.tg; ++i) m->eval(&ids[size_t(i)], 1, i, logits.data());
        tg_ms.push_back(ms_since(t1));
    }
    const double pp = o.pp * 1000.0 / median(pp_ms), tg = o.tg * 1000.0 / median(tg_ms);
    std::fprintf(stderr, "%s: yukleme %.0f ms | pp%d %.1f token/sn | tg%d %.1f token/sn (%.2f ms/token)\n",
                 m->name(), ms_load, o.pp, pp, o.tg, tg, median(tg_ms) / o.tg);
    std::printf("{\"backend\":\"%s\",\"ms_load\":%.1f,\"pp\":%d,\"pp_tps\":%.2f,\"tg\":%d,\"tg_tps\":%.2f}\n",
                m->name(), ms_load, o.pp, pp, o.tg, tg);
    return 0;
}

int cmd_dequant(const Gguf& g, const std::string& name, const char* first, const char* count) {
    const Tensor& t = g.require_tensor(name);
    const int rows = int(std::min<uint64_t>(t.rows(), uint64_t(INT32_MAX)));
    const int r0 = parse_int(first, 0, rows - 1, "ilk satir");
    const int n = parse_int(count, 1, rows - r0, "satir sayisi");
    std::vector<float> buf(size_t(t.ne[0]));
    _setmode(_fileno(stdout), _O_BINARY);
    for (int r = r0; r < r0 + n; ++r) {
        get_row(t, uint64_t(r), buf.data());
        std::fwrite(buf.data(), sizeof(float), buf.size(), stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage();
    const std::string cmd = argv[1];
    const std::string path = argv[2];
    if (cmd == "dequant") {
        if (argc != 6) usage();
        try {
            Gguf g(path);
            return cmd_dequant(g, argv[3], argv[4], argv[5]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "hata: %s\n", e.what());
            return 1;
        }
    }
    Opts o;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--no-special") o.special = false;
        else if (a == "--lines") o.lines = true;
        else if (a == "-f") o.prompt_file = next();
        else if (a == "-n") o.n_predict = parse_int(next(), 1, 100000, "-n");
        else if (a == "--top") o.top = parse_int(next(), 1, 100, "--top");
        else if (a == "--ctx") o.ctx = parse_int(next(), 1, 1 << 20, "--ctx");
        else if (a == "--pp") o.pp = parse_int(next(), 1, 32768, "--pp");
        else if (a == "--tg") o.tg = parse_int(next(), 1, 32768, "--tg");
        else if (a == "--reps") o.reps = parse_int(next(), 1, 100, "--reps");
        else if (a == "--tools") o.tools_file = next();
        else if (a == "--fuzz") o.fuzz = parse_int(next(), 1, 10000000, "--fuzz");
        else if (a == "--seed") o.seed = uint32_t(parse_int(next(), 0, INT_MAX, "--seed"));
        else if (a == "--threads") set_threads(parse_int(next(), 1, 256, "--threads"));
        else if (a == "--backend") {
            o.backend = next();
            if (o.backend != "auto" && o.backend != "cpu" && o.backend != "cuda") usage();
        } else usage();
    }
    try {
        if (cmd == "constrain-test") return cmd_constrain_test(path, o);
        Gguf g(path);
        if (cmd == "info") return cmd_info(g);
        if (cmd == "tokenize") return cmd_tokenize(g, o);
        if (cmd == "generate") return cmd_generate(g, o);
        if (cmd == "serve") return cmd_serve(g, o);
        if (cmd == "bench") return cmd_bench(g, o);
        usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hata: %s\n", e.what());
        return 1;
    }
}
