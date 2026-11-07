// Küçük, katı JSON okuyucu (araç şeması dosyası için). RFC 8259: yinelenen anahtar, sondaki virgül, yorum,
// geçersiz UTF-8, eşleşmemiş vekil (surrogate) ve 64 düzeyden derin iç içelik hatadır. Sayıların ham yazımı da
// saklanır (büyük tamsayı sınırları double'a yuvarlanmasın).
#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace el {

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;  // String: değer; Number: ham yazım
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json* get(std::string_view key) const;  // nesnede anahtar; yoksa/nesne değilse nullptr
    bool is(Type t) const { return type == t; }
};

Json parse_json(std::string_view text);  // hata: std::runtime_error (konumlu ileti)

}  // namespace el
