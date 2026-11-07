// Bayt düzeyinde BPE tokenleştirici (GPT-2 / Qwen2 ailesi), GGUF metadata'sından kurulur.
//
// Akış: özel tokenler (<|im_start|> vb.) ayrılır → kalan metin qwen2 düzenli ifadesine göre ön-tokenlere bölünür →
// her ön-token baytları GPT-2 bayt→unicode eşlemesiyle karakterlere çevrilir → birleştirme sırasına (merges) göre BPE.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace el {

class Gguf;

class Tokenizer {
public:
    explicit Tokenizer(const Gguf& g);

    // parse_special: metindeki özel token yazımlarını tek token olarak tanı (Ollama raw istemleri böyle çalışır)
    std::vector<int32_t> encode(std::string_view utf8, bool parse_special = true) const;
    std::string token_bytes(int32_t id) const;  // tokenin ham baytları (özel tokenler kendi yazımıyla)
    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t vocab_size() const { return int32_t(vocab_.size()); }
    int32_t eos() const { return eos_; }
    bool is_special(int32_t id) const;

    // test için: ön-tokenleştirici çıktısı
    std::vector<std::string> pretokenize(std::string_view utf8) const;

private:
    void bpe(std::string_view piece, std::vector<int32_t>& out) const;
    std::string token_bytes_raw(const std::string& mapped) const;  // eşlenmiş yazım → ham baytlar

    std::vector<std::string> vocab_;  // id → token yazımı (bayt→unicode eşlenmiş)
    std::vector<int32_t> type_;
    std::unordered_map<std::string, int32_t> lookup_;
    std::unordered_map<std::string, int32_t> merge_rank_;  // "sol sağ" → sıra
    std::vector<int32_t> specials_;                          // uzundan kısaya
    bool special_first_[256] = {};                           // bir özel token bu baytla başlayabilir mi
    std::string byte_char_[256];                             // bayt → eşlenmiş karakterin UTF-8 yazımı
    std::unordered_map<uint32_t, uint8_t> char_byte_;        // eşlenmiş kod noktası → bayt
    int32_t eos_ = -1;
};

}  // namespace el
