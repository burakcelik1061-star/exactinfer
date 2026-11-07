#include "model.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "ops.h"

namespace el {

namespace {

void expect_shape(const Tensor& t, std::initializer_list<uint64_t> ne) {
    std::vector<uint64_t> want(ne);
    if (t.ne != want) {
        std::string got, exp;
        for (auto v : t.ne) got += std::to_string(v) + " ";
        for (auto v : want) exp += std::to_string(v) + " ";
        throw std::runtime_error("beklenmeyen tensor boyutu " + t.name + ": [" + got + "] yerine [" + exp + "]");
    }
}

void rope(float* v, int pos, int dim, const std::vector<double>& inv_freq) {
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const double th = pos * inv_freq[i];
        const float cs = float(std::cos(th)), sn = float(std::sin(th));
        const float a = v[i], b = v[i + half];
        v[i] = a * cs - b * sn;
        v[i + half] = a * sn + b * cs;
    }
}

}  // namespace

// ---------------------------------------------------------------- yapılandırma ve ağırlık tablosu

Config read_config(const Gguf& g) {
    Config c;
    const std::string arch = g.get_str("general.architecture");
    if (arch != "qwen3") throw std::runtime_error("desteklenmeyen mimari: " + arch + " (su an yalnizca qwen3)");
    const std::string p = arch + ".";

    c.n_layer = int(g.get_int(p + "block_count"));
    c.n_embd = int(g.get_int(p + "embedding_length"));
    c.n_ff = int(g.get_int(p + "feed_forward_length"));
    c.n_head = int(g.get_int(p + "attention.head_count"));
    c.n_head_kv = int(g.get_int_or(p + "attention.head_count_kv", c.n_head));
    c.head_dim = int(g.get_int_or(p + "attention.key_length", c.n_head ? c.n_embd / c.n_head : 0));
    if (g.get_int_or(p + "attention.value_length", c.head_dim) != c.head_dim)
        throw std::runtime_error("key_length != value_length desteklenmiyor");
    c.rope_dim = int(g.get_int_or(p + "rope.dimension_count", c.head_dim));
    c.rope_base = g.get_float_or(p + "rope.freq_base", 10000.0);
    c.eps = float(g.get_float_or(p + "attention.layer_norm_rms_epsilon", 1e-6));
    c.n_ctx_train = int(g.get_int_or(p + "context_length", 0));

    if (c.n_layer <= 0 || c.n_layer > 512 || c.n_embd <= 0 || c.n_head <= 0 || c.n_head_kv <= 0 ||
        c.n_head % c.n_head_kv != 0 || c.head_dim <= 0 || c.head_dim % 2 != 0 || c.rope_dim > c.head_dim ||
        c.rope_dim % 2 != 0 || c.n_ff <= 0)
        throw std::runtime_error("gecersiz model yapilandirmasi");
    if (g.find(p + "rope.scaling.type") && g.get_str(p + "rope.scaling.type") != "none")
        throw std::runtime_error("RoPE olcekleme henuz desteklenmiyor");
    return c;
}

ModelTensors read_tensors(const Gguf& g, Config& c) {
    const uint64_t E = uint64_t(c.n_embd), F = uint64_t(c.n_ff), D = uint64_t(c.head_dim);
    const uint64_t QD = D * uint64_t(c.n_head), KD = D * uint64_t(c.n_head_kv);
    ModelTensors m;
    m.tok_embd = &g.require_tensor("token_embd.weight");
    if (m.tok_embd->ne.size() != 2 || m.tok_embd->ne[0] != E) throw std::runtime_error("token_embd boyutu gecersiz");
    c.n_vocab = int(m.tok_embd->ne[1]);
    m.output = g.tensor("output.weight");
    if (!m.output) m.output = m.tok_embd;  // bağlı (tied) gömme
    expect_shape(*m.output, {E, uint64_t(c.n_vocab)});
    m.output_norm = &g.require_tensor("output_norm.weight");
    expect_shape(*m.output_norm, {E});

    for (int l = 0; l < c.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        for (const char* bias : {"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias"})
            if (g.tensor(b + bias)) throw std::runtime_error("sapma (bias) tensoru beklenmiyordu: " + b + bias);
        LayerTensors L;
        L.wq = &g.require_tensor(b + "attn_q.weight");      expect_shape(*L.wq, {E, QD});
        L.wk = &g.require_tensor(b + "attn_k.weight");      expect_shape(*L.wk, {E, KD});
        L.wv = &g.require_tensor(b + "attn_v.weight");      expect_shape(*L.wv, {E, KD});
        L.wo = &g.require_tensor(b + "attn_output.weight"); expect_shape(*L.wo, {QD, E});
        L.gate = &g.require_tensor(b + "ffn_gate.weight");  expect_shape(*L.gate, {E, F});
        L.up = &g.require_tensor(b + "ffn_up.weight");      expect_shape(*L.up, {E, F});
        L.down = &g.require_tensor(b + "ffn_down.weight");  expect_shape(*L.down, {F, E});
        L.attn_norm = &g.require_tensor(b + "attn_norm.weight");  expect_shape(*L.attn_norm, {E});
        L.ffn_norm = &g.require_tensor(b + "ffn_norm.weight");    expect_shape(*L.ffn_norm, {E});
        L.q_norm = &g.require_tensor(b + "attn_q_norm.weight");   expect_shape(*L.q_norm, {D});
        L.k_norm = &g.require_tensor(b + "attn_k_norm.weight");   expect_shape(*L.k_norm, {D});
        for (const Tensor* t : {L.wq, L.wk, L.wv, L.wo, L.gate, L.up, L.down, L.attn_norm, L.ffn_norm, L.q_norm,
                                L.k_norm})
            if (!supported_type(t->type))
                throw std::runtime_error(std::string("tensor turu henuz desteklenmiyor: ") + type_name(t->type) +
                                         " (" + t->name + ")");
        m.layers.push_back(L);
    }
    return m;
}

std::vector<double> rope_inv_freq(const Config& c) {
    std::vector<double> f(size_t(c.rope_dim / 2));
    for (int i = 0; i < c.rope_dim / 2; ++i) f[size_t(i)] = std::pow(c.rope_base, -2.0 * i / c.rope_dim);
    return f;
}

// ---------------------------------------------------------------- CPU arka ucu (fp32 başvuru)

namespace {

class CpuModel final : public Backend {
public:
    CpuModel(const Gguf& g, int n_ctx) : c_(read_config(g)), n_ctx_(n_ctx) {
        if (n_ctx_ <= 0) throw std::runtime_error("baglam uzunlugu pozitif olmali");
        t_ = read_tensors(g, c_);
        output_norm_ = to_f32(*t_.output_norm);
        for (const LayerTensors& L : t_.layers)
            norms_.push_back({to_f32(*L.attn_norm), to_f32(*L.ffn_norm), to_f32(*L.q_norm), to_f32(*L.k_norm)});
        inv_freq_ = rope_inv_freq(c_);

        const size_t E = size_t(c_.n_embd), F = size_t(c_.n_ff);
        const size_t QD = size_t(c_.head_dim) * c_.n_head, KD = size_t(c_.head_dim) * c_.n_head_kv;
        const size_t kv = size_t(c_.n_layer) * size_t(n_ctx_) * KD;
        kcache_.assign(kv, 0.f);
        vcache_.assign(kv, 0.f);
        x_.resize(E); h_.resize(E); q_.resize(QD); k_.resize(KD); v_.resize(KD); att_.resize(QD); ao_.resize(E);
        g_.resize(F); u_.resize(F); scores_.resize(size_t(c_.n_head) * size_t(n_ctx_));
    }

    const Config& cfg() const override { return c_; }
    int n_ctx() const override { return n_ctx_; }
    const char* name() const override { return "cpu"; }

    void eval(const int32_t* tokens, int n, int pos0, float* logits) override {
        for (int i = 0; i < n; ++i) forward(tokens[i], pos0 + i, i + 1 == n ? logits : nullptr);
    }

private:
    struct Norms { std::vector<float> attn, ffn, q, k; };

    void forward(int32_t token, int pos, float* logits);

    Config c_;
    int n_ctx_;
    ModelTensors t_;
    std::vector<float> output_norm_;
    std::vector<Norms> norms_;
    std::vector<double> inv_freq_;
    std::vector<float> kcache_, vcache_;  // [katman][konum][kv_kafa][head_dim]
    std::vector<float> x_, h_, q_, k_, v_, att_, ao_, g_, u_, scores_;
};

void CpuModel::forward(int32_t token, int pos, float* logits) {
    if (token < 0 || token >= c_.n_vocab) throw std::out_of_range("token id sinir disi");
    if (pos < 0 || pos >= n_ctx_) throw std::out_of_range("baglam penceresi doldu");

    const int E = c_.n_embd, D = c_.head_dim, H = c_.n_head, HKV = c_.n_head_kv, F = c_.n_ff;
    const int KD = D * HKV, group = H / HKV;
    const float scale = 1.0f / std::sqrt(float(D));

    get_row(*t_.tok_embd, uint64_t(token), x_.data());

    for (int l = 0; l < c_.n_layer; ++l) {
        const LayerTensors& L = t_.layers[size_t(l)];
        const Norms& N = norms_[size_t(l)];

        // --- dikkat
        rmsnorm(h_.data(), x_.data(), N.attn.data(), E, c_.eps);
        matvec(*L.wq, h_.data(), q_.data());
        matvec(*L.wk, h_.data(), k_.data());
        matvec(*L.wv, h_.data(), v_.data());
        for (int hd = 0; hd < H; ++hd) {
            float* q = q_.data() + hd * D;
            rmsnorm(q, q, N.q.data(), D, c_.eps);
            rope(q, pos, c_.rope_dim, inv_freq_);
        }
        for (int hd = 0; hd < HKV; ++hd) {
            float* k = k_.data() + hd * D;
            rmsnorm(k, k, N.k.data(), D, c_.eps);
            rope(k, pos, c_.rope_dim, inv_freq_);
        }
        float* kc = kcache_.data() + size_t(l) * n_ctx_ * KD;
        float* vc = vcache_.data() + size_t(l) * n_ctx_ * KD;
        std::copy(k_.begin(), k_.end(), kc + size_t(pos) * KD);
        std::copy(v_.begin(), v_.end(), vc + size_t(pos) * KD);

#pragma omp parallel for schedule(static)
        for (int hd = 0; hd < H; ++hd) {
            const float* q = q_.data() + hd * D;
            const int kvh = hd / group;
            float* sc = scores_.data() + size_t(hd) * n_ctx_;
            float mx = -INFINITY;
            for (int t = 0; t <= pos; ++t) {
                const float* k = kc + size_t(t) * KD + size_t(kvh) * D;
                float s = 0.f;
                for (int i = 0; i < D; ++i) s += q[i] * k[i];
                sc[t] = s * scale;
                if (sc[t] > mx) mx = sc[t];
            }
            double sum = 0.0;
            for (int t = 0; t <= pos; ++t) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
            const float inv = float(1.0 / sum);
            float* out = att_.data() + hd * D;
            for (int i = 0; i < D; ++i) out[i] = 0.f;
            for (int t = 0; t <= pos; ++t) {
                const float* v = vc + size_t(t) * KD + size_t(kvh) * D;
                const float p = sc[t] * inv;
                for (int i = 0; i < D; ++i) out[i] += p * v[i];
            }
        }
        matvec(*L.wo, att_.data(), ao_.data());
        for (int i = 0; i < E; ++i) x_[size_t(i)] += ao_[size_t(i)];

        // --- ileri besleme (SwiGLU)
        rmsnorm(h_.data(), x_.data(), N.ffn.data(), E, c_.eps);
        matvec(*L.gate, h_.data(), g_.data());
        matvec(*L.up, h_.data(), u_.data());
        for (int i = 0; i < F; ++i) {
            const float a = g_[size_t(i)];
            g_[size_t(i)] = a / (1.0f + std::exp(-a)) * u_[size_t(i)];
        }
        matvec(*L.down, g_.data(), ao_.data());
        for (int i = 0; i < E; ++i) x_[size_t(i)] += ao_[size_t(i)];
    }

    if (logits) {
        rmsnorm(h_.data(), x_.data(), output_norm_.data(), E, c_.eps);
        matvec(*t_.output, h_.data(), logits);
    }
}

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(const Gguf& g, int n_ctx) {
    return std::make_unique<CpuModel>(g, n_ctx);
}

}  // namespace el
