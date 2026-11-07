#include "ops.h"

#include <immintrin.h>
#include <omp.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "quant.h"

namespace el {

static int g_threads = 0;

void set_threads(int n) {
    g_threads = n > 0 ? n : 0;
    if (g_threads) omp_set_num_threads(g_threads);
}
int threads() { return g_threads ? g_threads : omp_get_max_threads(); }

float f16_to_f32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}

float bf16_to_f32(uint16_t h) {
    uint32_t u = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool supported_type(GgmlType t) {
    return t == GgmlType::F32 || t == GgmlType::F16 || t == GgmlType::BF16 || t == GgmlType::Q8_0 ||
           t == GgmlType::Q4_K || t == GgmlType::Q6_K;
}

static void need_supported(const Tensor& t) {
    if (!supported_type(t.type))
        throw std::runtime_error(std::string("tensor turu henuz desteklenmiyor: ") + type_name(t.type) + " (" + t.name + ")");
}

static void convert_row(const Tensor& w, const uint8_t* src, uint64_t n, float* out) {
    switch (w.type) {
        case GgmlType::F32:
            std::memcpy(out, src, n * 4);
            break;
        case GgmlType::F16: {
            const auto* h = reinterpret_cast<const uint16_t*>(src);
            uint64_t j = 0;
            for (; j + 8 <= n; j += 8)
                _mm256_storeu_ps(out + j, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(h + j))));
            for (; j < n; ++j) out[j] = f16_to_f32(h[j]);
            break;
        }
        case GgmlType::BF16: {
            const auto* h = reinterpret_cast<const uint16_t*>(src);
            for (uint64_t j = 0; j < n; ++j) out[j] = bf16_to_f32(h[j]);
            break;
        }
        case GgmlType::Q8_0: dequant_q8_0(src, out, int64_t(n)); break;
        case GgmlType::Q4_K: dequant_q4_K(src, out, int64_t(n)); break;
        case GgmlType::Q6_K: dequant_q6_K(src, out, int64_t(n)); break;
        default:
            need_supported(w);
    }
}

std::vector<float> to_f32(const Tensor& t) {
    need_supported(t);
    std::vector<float> out(size_t(t.numel));
    convert_row(t, t.data, t.numel, out.data());
    return out;
}

void get_row(const Tensor& w, uint64_t row, float* out) {
    need_supported(w);
    if (row >= w.rows()) throw std::out_of_range("satir sinir disi: " + w.name);
    const uint64_t n = w.ne[0];
    const uint64_t row_bytes = w.nbytes / w.rows();
    convert_row(w, w.data + row * row_bytes, n, out);
}

static inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

static float dot_f16(const uint16_t* w, const float* x, int64_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 32 <= n; j += 32) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j))), _mm256_loadu_ps(x + j), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j + 8))), _mm256_loadu_ps(x + j + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j + 16))), _mm256_loadu_ps(x + j + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j + 24))), _mm256_loadu_ps(x + j + 24), a3);
    }
    for (; j + 8 <= n; j += 8)
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j))), _mm256_loadu_ps(x + j), a0);
    float s = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; j < n; ++j) s += f16_to_f32(w[j]) * x[j];
    return s;
}

static float dot_f32(const float* w, const float* x, int64_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 16 <= n; j += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j), _mm256_loadu_ps(x + j), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j + 8), _mm256_loadu_ps(x + j + 8), a1);
    }
    float s = hsum(_mm256_add_ps(a0, a1));
    for (; j < n; ++j) s += w[j] * x[j];
    return s;
}

static float dot_bf16(const uint16_t* w, const float* x, int64_t n) {
    float s = 0.f;
    for (int64_t j = 0; j < n; ++j) s += bf16_to_f32(w[j]) * x[j];
    return s;
}

void matvec(const Tensor& w, const float* x, float* y) {
    need_supported(w);
    const int64_t n = int64_t(w.ne[0]);
    const int64_t rows = int64_t(w.rows());
    const GgmlType t = w.type;
    const uint8_t* base = w.data;
    const int64_t row_bytes = int64_t(w.nbytes) / rows;
    if (t != GgmlType::F16 && t != GgmlType::F32 && t != GgmlType::BF16) {
        // Nicemli: satır iş parçacığının tamponuna çözülür, sonra fp32 iç çarpım (CPU başvurusu; hız ikincil)
#pragma omp parallel
        {
            std::vector<float> buf(static_cast<size_t>(n));
#pragma omp for schedule(static)
            for (int64_t r = 0; r < rows; ++r) {
                convert_row(w, base + r * row_bytes, uint64_t(n), buf.data());
                y[r] = dot_f32(buf.data(), x, n);
            }
        }
        return;
    }
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* p = base + r * row_bytes;
        if (t == GgmlType::F16) y[r] = dot_f16(reinterpret_cast<const uint16_t*>(p), x, n);
        else if (t == GgmlType::F32) y[r] = dot_f32(reinterpret_cast<const float*>(p), x, n);
        else y[r] = dot_bf16(reinterpret_cast<const uint16_t*>(p), x, n);
    }
}

void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float inv = float(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

}  // namespace el
