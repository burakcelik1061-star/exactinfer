#include "json.h"

#include <cstdlib>
#include <stdexcept>

namespace el {

const Json* Json::get(std::string_view key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& [k, v] : obj)
        if (k == key) return &v;
    return nullptr;
}

namespace {

constexpr int kMaxDepth = 64;

struct Parser {
    std::string_view s;
    size_t i = 0;

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error("JSON: " + std::string(what) + " (bayt " + std::to_string(i) + ")");
    }
    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    bool eat(char c) {
        if (i < s.size() && s[i] == c) { ++i; return true; }
        return false;
    }
    void lit(const char* w) {
        for (const char* p = w; *p; ++p)
            if (!eat(*p)) fail("geçersiz sabit");
    }

    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) { out += char(0xC0 | cp >> 6); out += char(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            out += char(0xE0 | cp >> 12); out += char(0x80 | (cp >> 6 & 0x3F)); out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | cp >> 18); out += char(0x80 | (cp >> 12 & 0x3F));
            out += char(0x80 | (cp >> 6 & 0x3F)); out += char(0x80 | (cp & 0x3F));
        }
    }

    uint32_t hex4() {
        if (i + 4 > s.size()) fail("kısa \\u kaçışı");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else fail("geçersiz \\u kaçışı");
        }
        return v;
    }

    // Ham UTF-8 baytını doğrulayarak kopyalar (aşırı uzun, vekil, > U+10FFFF reddedilir).
    void utf8_char(std::string& out) {
        const auto b0 = uint8_t(s[i]);
        int n;
        uint8_t lo = 0x80, hi = 0xBF;
        if (b0 >= 0xC2 && b0 <= 0xDF) n = 1;
        else if (b0 >= 0xE0 && b0 <= 0xEF) { n = 2; if (b0 == 0xE0) lo = 0xA0; if (b0 == 0xED) hi = 0x9F; }
        else if (b0 >= 0xF0 && b0 <= 0xF4) { n = 3; if (b0 == 0xF0) lo = 0x90; if (b0 == 0xF4) hi = 0x8F; }
        else fail("geçersiz UTF-8");
        out += char(b0);
        ++i;
        for (int k = 0; k < n; ++k, ++i) {
            if (i >= s.size()) fail("yarım UTF-8");
            const auto b = uint8_t(s[i]);
            if (b < (k == 0 ? lo : 0x80) || b > (k == 0 ? hi : 0xBF)) fail("geçersiz UTF-8");
            out += char(b);
        }
    }

    std::string string() {
        if (!eat('"')) fail("metin bekleniyordu");
        std::string out;
        for (;;) {
            if (i >= s.size()) fail("kapanmamış metin");
            const auto c = uint8_t(s[i]);
            if (c == '"') { ++i; return out; }
            if (c < 0x20) fail("metinde kontrol karakteri");
            if (c == '\\') {
                ++i;
                if (i >= s.size()) fail("yarım kaçış");
                const char e = s[i++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        uint32_t cp = hex4();
                        if (cp >= 0xDC00 && cp <= 0xDFFF) fail("eşleşmemiş vekil");
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (!eat('\\') || !eat('u')) fail("eşleşmemiş vekil");
                            const uint32_t lo = hex4();
                            if (lo < 0xDC00 || lo > 0xDFFF) fail("eşleşmemiş vekil");
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        put_utf8(out, cp);
                        break;
                    }
                    default: fail("geçersiz kaçış");
                }
            } else if (c < 0x80) {
                out += char(c);
                ++i;
            } else {
                utf8_char(out);
            }
        }
    }

    Json number() {
        const size_t st = i;
        eat('-');
        if (eat('0')) {
        } else if (i < s.size() && s[i] >= '1' && s[i] <= '9') {
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        } else fail("geçersiz sayı");
        if (eat('.')) {
            if (!(i < s.size() && s[i] >= '0' && s[i] <= '9')) fail("geçersiz sayı");
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (!eat('+')) eat('-');
            if (!(i < s.size() && s[i] >= '0' && s[i] <= '9')) fail("geçersiz sayı");
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        }
        Json j;
        j.type = Json::Type::Number;
        j.str = std::string(s.substr(st, i - st));
        j.num = std::strtod(j.str.c_str(), nullptr);
        return j;
    }

    Json value(int depth) {
        if (depth > kMaxDepth) fail("çok derin iç içelik");
        ws();
        if (i >= s.size()) fail("beklenmeyen son");
        Json j;
        switch (s[i]) {
            case '{': {
                ++i;
                j.type = Json::Type::Object;
                ws();
                if (eat('}')) return j;
                for (;;) {
                    ws();
                    std::string k = string();
                    if (j.get(k)) fail("yinelenen anahtar");
                    ws();
                    if (!eat(':')) fail("':' bekleniyordu");
                    j.obj.emplace_back(std::move(k), value(depth + 1));
                    ws();
                    if (eat('}')) return j;
                    if (!eat(',')) fail("',' ya da '}' bekleniyordu");
                }
            }
            case '[': {
                ++i;
                j.type = Json::Type::Array;
                ws();
                if (eat(']')) return j;
                for (;;) {
                    j.arr.push_back(value(depth + 1));
                    ws();
                    if (eat(']')) return j;
                    if (!eat(',')) fail("',' ya da ']' bekleniyordu");
                }
            }
            case '"':
                j.type = Json::Type::String;
                j.str = string();
                return j;
            case 't': lit("true"); j.type = Json::Type::Bool; j.b = true; return j;
            case 'f': lit("false"); j.type = Json::Type::Bool; return j;
            case 'n': lit("null"); return j;
            default: return number();
        }
    }
};

}  // namespace

Json parse_json(std::string_view text) {
    Parser p{text};
    if (text.size() >= 3 && text.substr(0, 3) == "\xEF\xBB\xBF") p.i = 3;  // UTF-8 BOM
    Json j = p.value(0);
    p.ws();
    if (p.i != text.size()) p.fail("değerden sonra fazladan veri");
    return j;
}

}  // namespace el
