// Kısıtlı çözümleme: <tool_call> içindeki JSON'u araç şemasına bağlayan bayt düzeyinde eşleştirici.
//
// Şema, bekçinin `tools` çıktısıdır (OpenAI/Qwen3 "function" listesi). Eşleştirici yalnızca şu biçimi kabul eder:
//   {"name": <araç adı>, "arguments": {<parametreler>}}
// - araç adı ve enum değerleri listeden; parametre adları aracın şemasından, her biri en çok bir kez, zorunlular
//   kapanıştan önce; fazladan alan yok;
// - türler birebir: metin / tamsayı (aralık) / sayı (aralık) / true-false;
// - metinde görünmez ya da yön değiştiren karakter, \u kaçışı ve ham kontrol karakteri yok; `format` notuna göre
//   ek kural: name (harf, rakam, boşluk, _ - .), path (sürücü harfiyle başlayan mutlak yol), uri (https://),
//   multiline (\n ve \t kaçışlarına izin); maxLength kod noktası olarak. Kurallar bekçinin doğrulayıcılarıyla
//   aynıdır (yolda aygıt adı yok, adreste kullanıcı bilgisi yok); yalnız izinli kök klasör denetimi bekçide kalır.
// Değişmez kural: kabul edilen her önek, geçerli bir çağrıya tamamlanabilir (çıkmaz yok). Bu yüzden yarım UTF-8
// karakterleri, tamsayı önekleri ve sınırlar "tamamlanabilir mi?" diye denetlenir.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace el {

struct Json;

enum class StrKind : uint8_t { Text, Multiline, Name, Path, Uri, Enum };

struct ParamSpec {
    enum class Type : uint8_t { String, Integer, Number, Boolean };
    std::string name;
    Type type = Type::String;
    StrKind kind = StrKind::Text;
    uint32_t max_len = 2000;
    std::vector<std::string> values;  // enum
    int64_t imin = INT64_MIN, imax = INT64_MAX;
    double nmin = -1e300, nmax = 1e300;
    bool required = false;
};

struct ToolSpec {
    std::string name;
    std::vector<ParamSpec> params;
    uint64_t required_mask = 0;
};

struct ToolSchema {
    std::vector<ToolSpec> tools;
    static ToolSchema from_json(std::string_view text);  // hata: std::runtime_error
    static ToolSchema from_value(const Json& root);
};

class CallMatcher {
public:
    explicit CallMatcher(const ToolSchema& s) : sc_(&s) {}
    bool feed(std::string_view bytes);  // false: geçersiz (durum artık kullanılmamalı; denemeden önce kopyalayın)
    bool complete() const { return st_ == S::Done; }
    int tool() const { return tool_; }

private:
    enum class S : uint8_t {
        TopOpen, NameKey, Colon1, NameVal, AfterName, ArgsKey, Colon2, ArgsOpen, ArgFirst,
        ArgColon, ArgValue, Str, Int, Num, Lit, AfterValue, ArgNext, TopClose, Done
    };
    enum class Role : uint8_t { ToolName, Key, Value };

    bool byte(uint8_t c);
    bool str_byte(uint8_t c);
    bool str_close();
    bool add_cp(uint32_t cp, bool escaped);
    bool value_cp_ok(uint32_t cp, bool escaped) const;  // değer metninde bu konumda `cp` izinli mi
    bool partial_ok() const;  // yarım UTF-8 karakteri izinli bir karaktere tamamlanabilir mi
    bool prefix_ok() const;   // ToolName / Key / Enum: buf_ (+ yarım baytlar) bir adayın öneki mi
    bool int_byte(uint8_t c);
    bool num_byte(uint8_t c);
    bool begin_value(uint8_t c);
    const ParamSpec& param() const { return sc_->tools[size_t(tool_)].params[size_t(param_)]; }
    uint64_t unseen() const;

    const ToolSchema* sc_;
    S st_ = S::TopOpen;
    uint8_t nws_ = 0;  // ardışık boşluk (sonsuz boşluk döngüsüne karşı sınır)
    const char* lit_ = nullptr;
    uint8_t lit_pos_ = 0;
    S lit_next_ = S::Done;
    int tool_ = -1, param_ = -1;
    uint64_t seen_ = 0;
    Role role_ = Role::Value;
    std::string buf_;      // metnin çözülmüş baytları / sayının yazımı
    uint32_t ncp_ = 0;     // metindeki kod noktası sayısı
    uint32_t last_cp_ = 0;
    size_t comp_ = 0;      // yol: geçerli bileşenin buf_ içindeki başlangıcı
    bool esc_ = false;
    uint8_t u8_[4] = {};   // yarım UTF-8 karakteri
    uint8_t u8n_ = 0, u8want_ = 0;
    uint8_t numst_ = 0;    // sayı DFA durumu
};

}  // namespace el
