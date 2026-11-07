// GGUF model dosyası okuyucu.
//
// Dosya GÜVENİLMEZ girdi sayılır: her uzunluk, sayı ve konum dosya sınırlarına göre denetlenir; taşma, sınır dışı
// okuma ya da bilinmeyen tür bulunan dosya reddedilir (GgufError). Veri bölümü kopyalanmaz, bellek eşlemesiyle okunur.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace el {

struct GgufError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ggml tensör türleri (yalnız boyutlarını bildiklerimiz; hesap desteği ops.cpp'de ayrıca denetlenir)
enum class GgmlType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, BF16 = 30,
};
const char* type_name(GgmlType t);

struct Tensor {
    std::string name;
    GgmlType type = GgmlType::F32;
    std::vector<uint64_t> ne;   // boyutlar; ne[0] bellekte ardışık olan (satır uzunluğu)
    uint64_t numel = 0;
    uint64_t nbytes = 0;
    const uint8_t* data = nullptr;

    uint64_t rows() const { return numel / ne[0]; }
};

enum class ValueType : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, BOOL = 7, STRING = 8, ARRAY = 9,
    U64 = 10, I64 = 11, F64 = 12,
};

struct Value {
    ValueType type = ValueType::U8;
    ValueType elem = ValueType::U8;  // dizi ise öğe türü
    int64_t i = 0;                   // tamsayı ve bool
    double f = 0.0;                  // kayan nokta
    std::string s;
    std::vector<int64_t> ai;         // tamsayı dizisi
    std::vector<double> af;          // kayan nokta dizisi
    std::vector<std::string> as;     // metin dizisi
};

class MappedFile;

class Gguf {
public:
    explicit Gguf(const std::string& path);
    ~Gguf();
    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;

    uint32_t version() const { return version_; }
    const std::unordered_map<std::string, Value>& meta() const { return meta_; }
    const std::vector<Tensor>& tensors() const { return tensors_; }

    const Value* find(const std::string& key) const;
    const Tensor* tensor(const std::string& name) const;       // yoksa nullptr
    const Tensor& require_tensor(const std::string& name) const;

    // Zorunlu anahtarlar (yoksa ya da tür uymuyorsa GgufError)
    int64_t get_int(const std::string& key) const;
    double get_float(const std::string& key) const;
    const std::string& get_str(const std::string& key) const;
    int64_t get_int_or(const std::string& key, int64_t def) const;
    double get_float_or(const std::string& key, double def) const;

private:
    std::unique_ptr<MappedFile> file_;
    uint32_t version_ = 0;
    std::unordered_map<std::string, Value> meta_;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string, size_t> index_;
};

}  // namespace el
