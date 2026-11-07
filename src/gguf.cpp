#include "gguf.h"

#include <cstring>
#include <limits>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace el {

// ---------------------------------------------------------------- bellek eşlemeli dosya (salt okunur)

class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
        int wlen = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1, nullptr, 0);
        if (wlen <= 0) throw GgufError("dosya yolu UTF-8 degil");
        std::wstring wpath(size_t(wlen), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

        file_ = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw GgufError("dosya acilamadi: " + path);
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(file_, &sz) || sz.QuadPart <= 0) { close(); throw GgufError("dosya bos ya da okunamadi"); }
        size_ = uint64_t(sz.QuadPart);
        map_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map_) { close(); throw GgufError("dosya eslenemedi"); }
        data_ = static_cast<const uint8_t*>(MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0));
        if (!data_) { close(); throw GgufError("dosya gorunumu acilamadi"); }
    }
    ~MappedFile() { close(); }

    const uint8_t* data() const { return data_; }
    uint64_t size() const { return size_; }

private:
    void close() {
        if (data_) UnmapViewOfFile(data_);
        if (map_) CloseHandle(map_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        data_ = nullptr; map_ = nullptr; file_ = INVALID_HANDLE_VALUE;
    }
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE map_ = nullptr;
    const uint8_t* data_ = nullptr;
    uint64_t size_ = 0;
};

// ---------------------------------------------------------------- tür bilgisi

namespace {

struct TypeInfo { uint32_t block; uint32_t bytes; const char* name; };

bool type_info(uint32_t t, TypeInfo& out) {
    switch (GgmlType(t)) {
        case GgmlType::F32:  out = {1, 4, "F32"}; return true;
        case GgmlType::F16:  out = {1, 2, "F16"}; return true;
        case GgmlType::BF16: out = {1, 2, "BF16"}; return true;
        case GgmlType::Q4_0: out = {32, 18, "Q4_0"}; return true;
        case GgmlType::Q4_1: out = {32, 20, "Q4_1"}; return true;
        case GgmlType::Q5_0: out = {32, 22, "Q5_0"}; return true;
        case GgmlType::Q5_1: out = {32, 24, "Q5_1"}; return true;
        case GgmlType::Q8_0: out = {32, 34, "Q8_0"}; return true;
        case GgmlType::Q2_K: out = {256, 84, "Q2_K"}; return true;
        case GgmlType::Q3_K: out = {256, 110, "Q3_K"}; return true;
        case GgmlType::Q4_K: out = {256, 144, "Q4_K"}; return true;
        case GgmlType::Q5_K: out = {256, 176, "Q5_K"}; return true;
        case GgmlType::Q6_K: out = {256, 210, "Q6_K"}; return true;
    }
    return false;
}

// Sınır denetimli sıralı okuyucu
class Reader {
public:
    Reader(const uint8_t* p, uint64_t n) : p_(p), n_(n) {}

    uint64_t pos() const { return pos_; }

    void need(uint64_t k) const {
        if (k > n_ - pos_) throw GgufError("dosya beklenenden kisa (bozuk ya da kesik)");
    }
    template <class T> T pod() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t len = pod<uint64_t>();
        if (len > (1ull << 24)) throw GgufError("metin uzunlugu makul degil");
        need(len);
        std::string s(reinterpret_cast<const char*>(p_ + pos_), size_t(len));
        pos_ += len;
        return s;
    }

private:
    const uint8_t* p_;
    uint64_t n_;
    uint64_t pos_ = 0;
};

bool is_int(ValueType t) {
    switch (t) {
        case ValueType::U8: case ValueType::I8: case ValueType::U16: case ValueType::I16:
        case ValueType::U32: case ValueType::I32: case ValueType::U64: case ValueType::I64:
        case ValueType::BOOL:
            return true;
        default:
            return false;
    }
}

int64_t read_int(Reader& r, ValueType t) {
    switch (t) {
        case ValueType::U8: return r.pod<uint8_t>();
        case ValueType::I8: return r.pod<int8_t>();
        case ValueType::U16: return r.pod<uint16_t>();
        case ValueType::I16: return r.pod<int16_t>();
        case ValueType::U32: return r.pod<uint32_t>();
        case ValueType::I32: return r.pod<int32_t>();
        case ValueType::I64: return r.pod<int64_t>();
        case ValueType::U64: {
            uint64_t v = r.pod<uint64_t>();
            if (v > uint64_t(std::numeric_limits<int64_t>::max())) throw GgufError("u64 deger cok buyuk");
            return int64_t(v);
        }
        case ValueType::BOOL: {
            uint8_t b = r.pod<uint8_t>();
            if (b > 1) throw GgufError("gecersiz bool");
            return b;
        }
        default: throw GgufError("tamsayi turu beklenirdi");
    }
}

uint64_t elem_size(ValueType t) {
    switch (t) {
        case ValueType::U8: case ValueType::I8: case ValueType::BOOL: return 1;
        case ValueType::U16: case ValueType::I16: return 2;
        case ValueType::U32: case ValueType::I32: case ValueType::F32: return 4;
        case ValueType::U64: case ValueType::I64: case ValueType::F64: return 8;
        case ValueType::STRING: return 8;  // en az uzunluk alanı
        default: return 0;
    }
}

Value read_value(Reader& r, ValueType t) {
    Value v;
    v.type = t;
    if (is_int(t)) { v.i = read_int(r, t); v.f = double(v.i); return v; }
    switch (t) {
        case ValueType::F32: v.f = r.pod<float>(); return v;
        case ValueType::F64: v.f = r.pod<double>(); return v;
        case ValueType::STRING: v.s = r.str(); return v;
        case ValueType::ARRAY: {
            v.elem = ValueType(r.pod<uint32_t>());
            uint64_t n = r.pod<uint64_t>();
            uint64_t es = elem_size(v.elem);
            if (es == 0) throw GgufError("desteklenmeyen dizi oge turu (ic ice dizi?)");
            if (n > (1ull << 26)) throw GgufError("dizi cok uzun");
            r.need(n * es);  // en az bu kadar bayt olmalı; dev sayılarla bellek ayırmayı önler
            if (v.elem == ValueType::STRING) {
                v.as.reserve(size_t(n));
                for (uint64_t k = 0; k < n; ++k) v.as.push_back(r.str());
            } else if (v.elem == ValueType::F32 || v.elem == ValueType::F64) {
                v.af.reserve(size_t(n));
                for (uint64_t k = 0; k < n; ++k)
                    v.af.push_back(v.elem == ValueType::F32 ? double(r.pod<float>()) : r.pod<double>());
            } else {
                v.ai.reserve(size_t(n));
                for (uint64_t k = 0; k < n; ++k) v.ai.push_back(read_int(r, v.elem));
            }
            return v;
        }
        default: throw GgufError("bilinmeyen metadata turu " + std::to_string(uint32_t(t)));
    }
}

bool mul_ok(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

}  // namespace

const char* type_name(GgmlType t) {
    TypeInfo ti;
    return type_info(uint32_t(t), ti) ? ti.name : "?";
}

// ---------------------------------------------------------------- ayrıştırma

Gguf::Gguf(const std::string& path) : file_(std::make_unique<MappedFile>(path)) {
    Reader r(file_->data(), file_->size());
    if (r.pod<uint32_t>() != 0x46554747u) throw GgufError("GGUF dosyasi degil (sihirli sayi yanlis)");
    version_ = r.pod<uint32_t>();
    if (version_ != 2 && version_ != 3) throw GgufError("desteklenmeyen GGUF surumu " + std::to_string(version_));
    uint64_t n_tensors = r.pod<uint64_t>();
    uint64_t n_kv = r.pod<uint64_t>();
    if (n_tensors > 100000 || n_kv > 100000) throw GgufError("tensor/metadata sayisi makul degil");

    for (uint64_t k = 0; k < n_kv; ++k) {
        std::string key = r.str();
        ValueType t = ValueType(r.pod<uint32_t>());
        Value v = read_value(r, t);
        if (!meta_.emplace(std::move(key), std::move(v)).second) throw GgufError("yinelenen metadata anahtari");
    }

    struct Raw { uint64_t offset; };
    std::vector<Raw> raw;
    tensors_.reserve(size_t(n_tensors));
    for (uint64_t k = 0; k < n_tensors; ++k) {
        Tensor t;
        t.name = r.str();
        uint32_t nd = r.pod<uint32_t>();
        if (nd == 0 || nd > 4) throw GgufError("tensor boyut sayisi gecersiz: " + t.name);
        t.numel = 1;
        for (uint32_t d = 0; d < nd; ++d) {
            uint64_t n = r.pod<uint64_t>();
            if (n == 0 || !mul_ok(t.numel, n, t.numel)) throw GgufError("tensor boyutu gecersiz: " + t.name);
            t.ne.push_back(n);
        }
        uint32_t ty = r.pod<uint32_t>();
        TypeInfo ti;
        if (!type_info(ty, ti)) throw GgufError("bilinmeyen tensor turu " + std::to_string(ty) + ": " + t.name);
        t.type = GgmlType(ty);
        if (t.ne[0] % ti.block != 0) throw GgufError("satir uzunlugu blok boyutuna bolunmuyor: " + t.name);
        if (!mul_ok(t.numel / ti.block, ti.bytes, t.nbytes)) throw GgufError("tensor cok buyuk: " + t.name);
        raw.push_back({r.pod<uint64_t>()});
        if (!index_.emplace(t.name, tensors_.size()).second) throw GgufError("yinelenen tensor adi: " + t.name);
        tensors_.push_back(std::move(t));
    }

    uint64_t align = uint64_t(get_int_or("general.alignment", 32));
    if (align == 0 || (align & (align - 1)) != 0 || align > 4096) throw GgufError("gecersiz hizalama");
    uint64_t data_start = (r.pos() + align - 1) / align * align;
    if (data_start > file_->size()) throw GgufError("veri bolumu dosya disinda");
    uint64_t data_size = file_->size() - data_start;

    for (size_t k = 0; k < tensors_.size(); ++k) {
        Tensor& t = tensors_[k];
        uint64_t off = raw[k].offset;
        if (off % align != 0) throw GgufError("tensor hizali degil: " + t.name);
        if (off > data_size || t.nbytes > data_size - off) throw GgufError("tensor dosya disina tasiyor: " + t.name);
        t.data = file_->data() + data_start + off;
    }
}

Gguf::~Gguf() = default;

const Value* Gguf::find(const std::string& key) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? nullptr : &it->second;
}

const Tensor* Gguf::tensor(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const Tensor& Gguf::require_tensor(const std::string& name) const {
    const Tensor* t = tensor(name);
    if (!t) throw GgufError("tensor yok: " + name);
    return *t;
}

int64_t Gguf::get_int(const std::string& key) const {
    const Value* v = find(key);
    if (!v) throw GgufError("metadata yok: " + key);
    if (!is_int(v->type)) throw GgufError("metadata tamsayi degil: " + key);
    return v->i;
}

double Gguf::get_float(const std::string& key) const {
    const Value* v = find(key);
    if (!v) throw GgufError("metadata yok: " + key);
    if (v->type != ValueType::F32 && v->type != ValueType::F64 && !is_int(v->type))
        throw GgufError("metadata sayi degil: " + key);
    return v->f;
}

const std::string& Gguf::get_str(const std::string& key) const {
    const Value* v = find(key);
    if (!v) throw GgufError("metadata yok: " + key);
    if (v->type != ValueType::STRING) throw GgufError("metadata metin degil: " + key);
    return v->s;
}

int64_t Gguf::get_int_or(const std::string& key, int64_t def) const {
    return find(key) ? get_int(key) : def;
}

double Gguf::get_float_or(const std::string& key, double def) const {
    return find(key) ? get_float(key) : def;
}

}  // namespace el
