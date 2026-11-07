// Qwen3 yoğun (dense) dönüştürücü: yapılandırma, ağırlık tablosu ve hesap arka uçları (CPU / CUDA).
//
// Her katman:  x += Wo · attn(RoPE(qnorm(Wq·h)), RoPE(knorm(Wk·h)), Wv·h),   h = rmsnorm(x)
//              x += Wdown · (silu(Wgate·h) ⊙ Wup·h),                          h = rmsnorm(x)
// Dikkat: gruplu sorgu (GQA), nedensel maske, RoPE NEOX düzeni (yarımlar döner).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf.h"

namespace el {

struct Config {
    int n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_vocab = 0;
    int rope_dim = 0;
    double rope_base = 10000.0;
    float eps = 1e-6f;
    int n_ctx_train = 0;
};

struct LayerTensors {
    const Tensor *attn_norm, *ffn_norm, *q_norm, *k_norm;
    const Tensor *wq, *wk, *wv, *wo, *gate, *up, *down;
};

struct ModelTensors {
    const Tensor* tok_embd;
    const Tensor* output;  // bağlı gömmede tok_embd ile aynı
    const Tensor* output_norm;
    std::vector<LayerTensors> layers;
};

// Metadata'yı okur ve doğrular; tensör boyutlarını denetleyip tabloyu döndürür (c.n_vocab'ı doldurur).
Config read_config(const Gguf& g);
ModelTensors read_tensors(const Gguf& g, Config& c);
std::vector<double> rope_inv_freq(const Config& c);

class Backend {
public:
    virtual ~Backend() = default;
    virtual const Config& cfg() const = 0;
    virtual int n_ctx() const = 0;
    virtual const char* name() const = 0;
    // tokens[0..n) sırasıyla pos0.. konumlarında işlenir (KV önbelleğine yazılır). logits boş değilse SON tokenin
    // n_vocab logit'i yazılır. Konumlar 0'dan ardışık verilmelidir.
    virtual void eval(const int32_t* tokens, int n, int pos0, float* logits) = 0;
};

std::unique_ptr<Backend> make_cpu_backend(const Gguf& g, int n_ctx);

// CUDA derlenmemişse ya da uygun GPU yoksa false (neden `why`ye yazılır)
bool cuda_available(std::string* why);
std::unique_ptr<Backend> make_cuda_backend(const Gguf& g, int n_ctx);

}  // namespace el
