// CUDA arka ucu: Qwen3 ileri geçişi GPU'da.
//
// Ağırlıklar dosyadaki biçimde (F16, Q8_0, Q4_K, Q6_K) VRAM'e bir kez yüklenir ve çarpım sırasında çözülür;
// etkinleştirmeler, KV önbelleği ve tüm toplamalar fp32'dir (CPU başvurusuyla aynı sayısal sözleşme — hız için
// hassasiyetten ödün verilmez). Her nicemli ağırlık ggml'in formülü ve işlem sırasıyla fp32'ye çözülür.
//
// Çekirdekler:
//   gemv      tek token (çözme): her warp bir satır, x paylaşılan bellekte; ağırlık türüne göre şablonlu
//   gemm_q    toplu istem, tensor core (mma.sync m16n8k8): fp32 girdi iki fp16 parçaya bölünür (x ≈ hi + lo, satır
//             başına 2'nin kuvveti ölçekle); tensor core yalnızca nicemli ağırlığın tamsayı kısmıyla (fp16'da tam)
//             16'lık dilim toplamlarını hesaplar, dilim ölçeği ve min terimi fp32'de uygulanır
//   gemm_tc   toplu istem, F16 ağırlık, WMMA 16x16x16 (aynı hi+lo yöntemi)
//   gemm      toplu istem, F16 ağırlık, klasik fp32 karo çarpımı (karşılaştırma: EL_GEMM=simt)
//   attn      çevrimiçi softmax (flash tarzı): istemde karolu, çözmede parçalı (flash-decoding)
//   headnorm_rope  kafa başına RMSNorm + RoPE (NEOX) tek çekirdekte
// Q/K/V ve gate/up ağırlıkları yüklemede birleştirilir (aynı türdeki komşular tek matris). Gömme satırları CPU'da
// çözülüp kopyalanır (tablo VRAM'e yalnızca bağlı çıkış katmanı olarak girer). Tek tokenlik çözme adımı bir kez
// CUDA Graph olarak yakalanır ve her tokende tek çağrıyla yeniden oynatılır (konum aygıt belleğinden okunur).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "ops.h"
#include "quant.h"

namespace el {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA hatasi (") + what + "): " + cudaGetErrorString(e));
}
#define CK(x) ck((x), #x)
#define CK_LAUNCH() ck(cudaGetLastError(), "cekirdek baslatma")

constexpr int kHeadDim = 128;  // Qwen3 ailesinin tümünde 128; dikkat çekirdeği bu boyuta göre derlenir

// ---------------------------------------------------------------- yardımcılar

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// blockDim.x 32'nin katı olmalı; sonucu tüm iş parçacıklarına döndürür. MAX: en büyük değer (girdiler >= 0 olmalı)
template <bool MAX = false>
__device__ float block_reduce(float v) {
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = MAX ? warp_max(v) : warp_sum(v);
    __syncthreads();  // part[] önceki çağrıda kullanılmış olabilir
    if (lane == 0) part[warp] = v;
    __syncthreads();
    float s = lane < nw ? part[lane] : 0.f;
    return MAX ? warp_max(s) : warp_sum(s);
}

// ---------------------------------------------------------------- küçük çekirdekler

__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int n,
                          float eps) {
    const float* xr = x + size_t(blockIdx.x) * n;
    float* o = out + size_t(blockIdx.x) * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_reduce(s);
    const float inv = 1.0f / sqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = xr[i] * inv * w[i];
}

// grid (B, kafa sayısı), blockDim = head_dim. buf satırı b: buf + b*stride; kafa h: + h*D. Yerinde çalışır.
// cs: [konum][rope_dim/2] (cos, sin), çift hassasiyette hesaplanıp float'a yuvarlanmış tablo.
__global__ void k_headnorm_rope(float* buf, int stride, const float* __restrict__ w, const float2* __restrict__ cs,
                                const int* __restrict__ pos_p, int rope_dim, float eps) {
    extern __shared__ float sh[];
    const int D = blockDim.x, i = threadIdx.x;
    float* v = buf + size_t(blockIdx.x) * stride + size_t(blockIdx.y) * D;
    float a = v[i];
    const float ss = block_reduce(a * a);
    a = a * (1.0f / sqrtf(ss / D + eps)) * w[i];
    sh[i] = a;
    __syncthreads();
    const int half = rope_dim / 2;
    float r = a;
    if (i < rope_dim) {
        const int j = i < half ? i : i - half;
        const float2 c = cs[size_t(*pos_p + blockIdx.x) * half + j];
        r = i < half ? a * c.x - sh[i + half] * c.y : sh[i - half] * c.y + a * c.x;
    }
    v[i] = r;
}

// qkv satırı [Q | K | V]; K ve V, önbellekte pos+b satırına kopyalanır
__global__ void k_store_kv(const float* __restrict__ qkv, int qkv_stride, int qd, int kd, float* __restrict__ kc,
                           float* __restrict__ vc, const int* __restrict__ pos_p) {
    const int b = blockIdx.x;
    const float* src = qkv + size_t(b) * qkv_stride + qd;
    const size_t dst = size_t(*pos_p + b) * kd;
    for (int i = threadIdx.x; i < kd; i += blockDim.x) {
        kc[dst + i] = src[i];
        vc[dst + i] = src[kd + i];
    }
}

// gu satırı [gate | up] (2F); act = silu(gate) * up
__global__ void k_silu_mul(const float* __restrict__ gu, float* __restrict__ act, int F, size_t total) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t b = i / F, j = i % F;
    const float a = gu[b * 2 * F + j];
    act[i] = a / (1.0f + expf(-a)) * gu[b * 2 * F + F + j];
}

// ---------------------------------------------------------------- ağırlık türleri ve çözme

enum WType : int { W_F16 = 0, W_Q8_0 = 1, W_Q4_K = 2, W_Q6_K = 3 };

__device__ __forceinline__ float h2f(uint32_t h) { return __half2float(__ushort_as_half(uint16_t(h))); }

// Q4_K'nın 12 baytlık sıkıştırılmış ölçek alanı üç 32 bitlik sözcükte (a, b, c); q4k_scale_min ile aynı açma, ama
// çalışma anı indeksiyle yerel belleğe düşmeden yazmaçlarda.
__device__ __forceinline__ int sbyte(uint32_t a, uint32_t b, uint32_t c, int i) {
    const uint32_t w = i < 4 ? a : (i < 8 ? b : c);
    return int((w >> (8 * (i & 3))) & 0xffu);
}
__device__ __forceinline__ void q4k_sm(uint32_t a, uint32_t b, uint32_t c, int j, int& sc, int& m) {
    if (j < 4) {
        sc = sbyte(a, b, c, j) & 63;
        m = sbyte(a, b, c, j + 4) & 63;
    } else {
        sc = (sbyte(a, b, c, j + 4) & 0xF) | ((sbyte(a, b, c, j - 4) >> 6) << 4);
        m = (sbyte(a, b, c, j + 4) >> 4) | ((sbyte(a, b, c, j) >> 6) << 4);
    }
}

// Q6_K, yüklemede hizalı düzene çevrilir: blok = ql[128] qh[64] scales[16] (208 bayt, 16'nın katı), d ayrı (aux).
constexpr int kQ6Bytes = 208;

// Bir ağırlık satırının x ile iç çarpımına bu lane'in katkısı (x paylaşılan bellekte, warp'ın 32 lane'i satırı böler).
// Alt blok ölçeği ve min ggml'deki gibi fp32'de hesaplanır (d·sc, dmin·m) ve toplamın dışına alınır:
//   Σ x·(d1·q − m1) = d1·Σ x·q − m1·Σ x      (tensor core çarpımıyla aynı ayrıştırma)
// Bant genişliği için lane'ler ardışık baytları okur (birleşik erişim), x'i float4 ile bank çakışmasız okur ve 4 blok
// birden yüklenir (warp başına uçuşta daha çok bayt).
template <int T>
__device__ __forceinline__ float row_dot(const uint8_t* __restrict__ row, const __half* __restrict__ aux,
                                         const float* xs, int n, int lane);

template <>
__device__ __forceinline__ float row_dot<W_F16>(const uint8_t* __restrict__ row, const __half* __restrict__,
                                                const float* xs, int n, int lane) {
    const uint4* w4 = reinterpret_cast<const uint4*>(row);
    const float4* x4 = reinterpret_cast<const float4*>(xs);
    float s = 0.f;
#pragma unroll 4
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 u = __ldg(w4 + j);
        const __half2* h = reinterpret_cast<const __half2*>(&u);
        const float4 xa = x4[2 * j], xb = x4[2 * j + 1];
        const float2 f0 = __half22float2(h[0]), f1 = __half22float2(h[1]);
        const float2 f2 = __half22float2(h[2]), f3 = __half22float2(h[3]);
        s += f0.x * xa.x + f0.y * xa.y + f1.x * xa.z + f1.y * xa.w;
        s += f2.x * xb.x + f2.y * xb.y + f3.x * xb.z + f3.y * xb.w;
    }
    return s;
}

// Bayt → float, tam ve ucuz: 0x4B0000vv bit deseni 2^23 + vv'dir; 2^23 (ya da 2^23 + sapma) çıkarılınca vv − sapma
// tam olarak elde edilir. Bir PRMT (tamsayı birimi) + bir FADD — Turing'de çeyrek hızlı I2F dönüşümünden kaçınır.
__device__ __forceinline__ float byte_f(uint32_t w, int i, float bias) {
    return __int_as_float(int(__byte_perm(w, 0x4B000000u, 0x7440u + unsigned(i)))) - bias;
}
constexpr float kMagic = 8388608.0f;  // 2^23

__device__ __forceinline__ float dot_b4(uint32_t w, float4 x, float bias) {
    return byte_f(w, 0, bias) * x.x + byte_f(w, 1, bias) * x.y + byte_f(w, 2, bias) * x.z + byte_f(w, 3, bias) * x.w;
}

// Q8_0, yüklemede yeniden düzenlenmiş: row = int8[n], aux = satırın fp16 ölçekleri [n/32]. Lane başına 4 değer;
// işaretli bayt ^0x80 ile 0..255'e kaydırılır (sapma 128).
template <>
__device__ __forceinline__ float row_dot<W_Q8_0>(const uint8_t* __restrict__ row, const __half* __restrict__ aux,
                                                 const float* xs, int n, int lane) {
    const unsigned int* w32 = reinterpret_cast<const unsigned int*>(row);
    const float4* x4 = reinterpret_cast<const float4*>(xs);
    const int nc = n / 4;
    float s = 0.f;
    for (int c0 = 0; c0 < nc; c0 += 4 * 32) {
        uint32_t q[4] = {};
        float d[4] = {};
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int c = c0 + 32 * u + lane;
            if (c < nc) {
                q[u] = __ldg(w32 + c);
                d[u] = __half2float(aux[c / 8]);
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int c = c0 + 32 * u + lane;
            if (c < nc) s = fmaf(d[u], dot_b4(q[u] ^ 0x80808080u, x4[c], kMagic + 128.0f), s);
        }
    }
    return s;
}

// Q4_K: bloğa 8 lane; lane c, qs[16c..16c+15] okur → 64'lük parça j = c/2 içinde 16 alt-nibble (alt blok 2j) ve 16
// üst-nibble (alt blok 2j+1). Warp bir adımda 4 blok işler. x okumaları j kadar döndürülür (bank çakışması olmasın);
// alt blok başına Σx, x paylaşılan belleğe yüklenirken 16'lık dilimler hâlinde hesaplanmıştır (xs + n).
template <>
__device__ __forceinline__ float row_dot<W_Q4_K>(const uint8_t* __restrict__ row, const __half* __restrict__,
                                                 const float* xs, int n, int lane) {
    const int c = lane & 7, j = c >> 1, l0 = (c & 1) * 16, nb = n / QK_K;
    const float* xs16 = xs + n;
    float s = 0.f;
    for (int bi = lane >> 3; bi < nb; bi += 4) {
        const uint8_t* blk = row + size_t(bi) * sizeof(BlockQ4_K);
        const uint4 hd = __ldg(reinterpret_cast<const uint4*>(blk));  // d, dmin, scales[12]
        const uint4 qv = __ldg(reinterpret_cast<const uint4*>(blk + 16 + 32 * j + l0));
        const uint32_t w[4] = {qv.x, qv.y, qv.z, qv.w};
        const float* xp = xs + bi * QK_K + 64 * j + l0;
        float qa = 0.f, qb = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int r = (k + j) & 3;
            const uint32_t v = r == 0 ? w[0] : r == 1 ? w[1] : r == 2 ? w[2] : w[3];
            const float4 xa = *reinterpret_cast<const float4*>(xp + 4 * r);
            const float4 xb = *reinterpret_cast<const float4*>(xp + 32 + 4 * r);
            qa += dot_b4(v & 0x0F0F0F0Fu, xa, kMagic);
            qb += dot_b4((v >> 4) & 0x0F0F0F0Fu, xb, kMagic);
        }
        int s1, m1, s2, m2;
        q4k_sm(hd.y, hd.z, hd.w, 2 * j, s1, m1);
        q4k_sm(hd.y, hd.z, hd.w, 2 * j + 1, s2, m2);
        const float d = h2f(hd.x & 0xffffu), dmin = h2f(hd.x >> 16);
        const int x16 = (bi * QK_K + 64 * j + l0) / 16;
        s += __fmul_rn(d, float(s1)) * qa - __fmul_rn(dmin, float(m1)) * xs16[x16];
        s += __fmul_rn(d, float(s2)) * qb - __fmul_rn(dmin, float(m2)) * xs16[x16 + 2];
    }
    return s;
}

// Q6_K (hizalı düzen): yarım blok hf = lane/16 içinde ql[o..o+3] (o = 4·(lane%16)). ql baytının alt nibble'ı yarıdaki
// o konumunun, üst nibble'ı o+64'ün düşük 4 bitidir; üst 2 bit qh[o%32]'de, kaydırma 2·(konum/32). Konum p'nin ölçeği
// scales[8·hf + p/16]. 6 bitlik değerler sözcük düzeyinde birleştirilir (4 bayt birden), sapma 32.
template <>
__device__ __forceinline__ float row_dot<W_Q6_K>(const uint8_t* __restrict__ row, const __half* __restrict__ aux,
                                                 const float* xs, int n, int lane) {
    const int hf = lane >> 4, o = (lane & 15) * 4, nb = n / QK_K;
    const int sh_lo = 2 * (o / 32), sh_hi = sh_lo + 4;
    float s = 0.f;
    for (int b0 = 0; b0 < nb; b0 += 4) {
        uint32_t ql[4] = {}, qh[4] = {};
        int sa[4] = {}, sb[4] = {};
        float d[4] = {};
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (b0 + u < nb) {
                const uint8_t* blk = row + size_t(b0 + u) * kQ6Bytes;
                ql[u] = __ldg(reinterpret_cast<const unsigned int*>(blk + 64 * hf + o));
                qh[u] = __ldg(reinterpret_cast<const unsigned int*>(blk + 128 + 32 * hf + (o & 31)));
                const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192 + 8 * hf + o / 16);
                sa[u] = __ldg(sc);
                sb[u] = __ldg(sc + 4);
                d[u] = __half2float(aux[b0 + u]);
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (b0 + u >= nb) break;
            const float* xp = xs + (b0 + u) * QK_K + 128 * hf + o;
            const float4 xa = *reinterpret_cast<const float4*>(xp), xb = *reinterpret_cast<const float4*>(xp + 64);
            const uint32_t va = (ql[u] & 0x0F0F0F0Fu) | (((qh[u] >> sh_lo) & 0x03030303u) << 4);
            const uint32_t vb = ((ql[u] >> 4) & 0x0F0F0F0Fu) | (((qh[u] >> sh_hi) & 0x03030303u) << 4);
            const float ta = dot_b4(va, xa, kMagic + 32.0f), tb = dot_b4(vb, xb, kMagic + 32.0f);
            s += __fmul_rn(d[u], float(sa[u])) * ta + __fmul_rn(d[u], float(sb[u])) * tb;
        }
    }
    return s;
}

// ---------------------------------------------------------------- matris çarpımları

// y[r] (+)= W[r,:] · x. Her warp bir satır; ızgara SM sayısına göre sınırlı tutulur ve warp'lar satırlar üzerinde
// döner (x paylaşılan belleğe blok başına bir kez yüklenir).
template <int T, bool ACC>
__global__ void __launch_bounds__(1024) k_gemv(const uint8_t* __restrict__ W, const __half* __restrict__ aux, size_t row_bytes,
                       int aux_per_row, const float* __restrict__ x, float* y, int rows, int n) {
    extern __shared__ float xs[];  // x [n]; Q4_K'da ardından 16'lık dilim toplamları [n/16]
    for (int j = threadIdx.x; j < n; j += blockDim.x) xs[j] = x[j];
    if constexpr (T == W_Q4_K) {
        __syncthreads();
        for (int c = threadIdx.x; c < n / 16; c += blockDim.x) {
            float t = 0.f;
            for (int i = 0; i < 16; ++i) t += xs[16 * c + i];
            xs[n + c] = t;
        }
    }
    __syncthreads();
    const int warps = blockDim.x >> 5, lane = threadIdx.x & 31;
    for (int r = blockIdx.x * warps + (threadIdx.x >> 5); r < rows; r += gridDim.x * warps) {
        float s = row_dot<T>(W + size_t(r) * row_bytes, aux + size_t(r) * aux_per_row, xs, n, lane);
        s = warp_sum(s);
        if (lane == 0) y[r] = ACC ? y[r] + s : s;
    }
}

// Klasik fp32 karo çarpımı: Y[B][ldy] (+)= X[B][n] · W[rows][n]ᵀ  (W: F16)
constexpr int TM = 64, TN = 64, TK = 16;
template <bool ACC>
__global__ void __launch_bounds__(256) k_gemm(const float* __restrict__ X, const __half* __restrict__ W, float* Y,
                                              int ldy, int B, int rows, int n) {
    __shared__ float As[TK][TM + 4];
    __shared__ float Bs[TK][TN + 4];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int m0 = blockIdx.y * TM, n0 = blockIdx.x * TN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < n; k0 += TK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + i * 256, r = idx / TK, k = idx % TK, gk = k0 + k;
            const int gm = m0 + r, gn = n0 + r;
            As[k][r] = (gm < B && gk < n) ? X[size_t(gm) * n + gk] : 0.f;
            Bs[k][r] = (gn < rows && gk < n) ? __half2float(W[size_t(gn) * n + gk]) : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TK; ++k) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { a[i] = As[k][ty * 4 + i]; b[i] = Bs[k][tx * 4 + i]; }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gm = m0 + ty * 4 + i;
        if (gm >= B) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int gn = n0 + tx * 4 + j;
            if (gn >= rows) continue;
            float* p = Y + size_t(gm) * ldy + gn;
            *p = ACC ? *p + acc[i][j] : acc[i][j];
        }
    }
}

// X satırını iki fp16 parçaya böler: x·s ≈ hi + lo, s = 2^k (satırın en büyük |x|'i [2^14, 2^15) aralığına gelir).
// 2'nin kuvvetiyle ölçekleme kayıpsızdır; hi ~11, lo ~11 bit taşır → toplam ~22 bit (fp32: 24 bit).
// xsum16 boş değilse 16'lık dilim toplamlarını da yazar (ölçekli): nicemli çarpımda min terimi için
//   Σ x·(sc·q − mn) = sc·Σ x·q − mn·Σ x
__global__ void k_split(const float* __restrict__ X, __half* __restrict__ hi, __half* __restrict__ lo,
                        float* __restrict__ inv_scale, float* __restrict__ xsum16, int n) {
    const size_t off = size_t(blockIdx.x) * n;
    float m = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, fabsf(X[off + i]));
    m = block_reduce<true>(m);
    int e = 0;
    if (m > 0.f) { frexpf(m, &e); }   // m = f·2^e, f ∈ [0.5, 1)
    const int k = m > 0.f ? 15 - e : 0;  // m·2^k ∈ [2^14, 2^15)
    const float s = ldexpf(1.0f, k);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = X[off + i] * s;
        const __half h = __float2half_rn(v);
        hi[off + i] = h;
        lo[off + i] = __float2half_rn(v - __half2float(h));
    }
    if (threadIdx.x == 0) inv_scale[blockIdx.x] = ldexpf(1.0f, -k);
    if (xsum16) {
        for (int c = threadIdx.x; c < n / 16; c += blockDim.x) {
            float t = 0.f;
            for (int i = 0; i < 16; ++i) t += X[off + 16 * c + i] * s;
            xsum16[size_t(blockIdx.x) * (n / 16) + c] = t;
        }
    }
}

// Tensor core çarpımı (F16 ağırlık): Y[B][ldy] (+)= (Xhi + Xlo)[B][n] · W[rows][n]ᵀ · inv_scale[b].  n % GK == 0.
// Blok: 64 (token) x 128 (çıktı satırı), 8 warp (2x4), warp başına 32x32 = 2x2 WMMA parçası.
namespace wm = nvcuda::wmma;
constexpr int GM = 64, GN = 128, GK = 64, GPAD = 8;
template <bool ACC>
__global__ void __launch_bounds__(256) k_gemm_tc(const __half* __restrict__ Xh, const __half* __restrict__ Xl,
                                                 const float* __restrict__ inv_scale, const __half* __restrict__ W,
                                                 float* Y, int ldy, int B, int rows, int n) {
    __shared__ __align__(32) __half Ah[GM][GK + GPAD];
    __shared__ __align__(32) __half Al[GM][GK + GPAD];
    __shared__ __align__(32) __half Bs[GN][GK + GPAD];
    __shared__ __align__(32) float stage[8][16][16 + 4];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm_ = warp / 4, wn = warp % 4;  // warp'ın 32x32 karosu: satır wm_*32, sütun wn*32
    const int m0 = blockIdx.y * GM, n0 = blockIdx.x * GN;

    wm::fragment<wm::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wm::fill_fragment(acc[i][j], 0.f);

    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (int k0 = 0; k0 < n; k0 += GK) {
        // Karolar 16 baytlık (8 half) parçalarla yüklenir: satır başına GK/8 parça
        constexpr int CPR = GK / 8;
#pragma unroll
        for (int i = 0; i < GM * CPR / 256; ++i) {
            const int idx = tid + i * 256, r = idx / CPR, c = (idx % CPR) * 8, gm = m0 + r;
            const size_t g = size_t(gm) * n + k0 + c;
            *reinterpret_cast<uint4*>(&Ah[r][c]) = gm < B ? *reinterpret_cast<const uint4*>(Xh + g) : zero;
            *reinterpret_cast<uint4*>(&Al[r][c]) = gm < B ? *reinterpret_cast<const uint4*>(Xl + g) : zero;
        }
#pragma unroll
        for (int i = 0; i < GN * CPR / 256; ++i) {
            const int idx = tid + i * 256, r = idx / CPR, c = (idx % CPR) * 8, gn = n0 + r;
            *reinterpret_cast<uint4*>(&Bs[r][c]) =
                gn < rows ? __ldg(reinterpret_cast<const uint4*>(W + size_t(gn) * n + k0 + c)) : zero;
        }
        __syncthreads();
        // Tensor core yalnızca bu GK'lik dilimi toplar (iç toplaması IEEE fp32 kadar hassas değildir); dilim sonuçları
        // aşağıda olağan fp32 toplamayla ana biriktiriciye eklenir. hi ve lo ayrı tutulur ki lo'nun küçük katkısı
        // büyük hi toplamına hizalanırken kesilmesin.
        wm::fragment<wm::accumulator, 16, 16, 16, float> th[2][2], tl[2][2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) { wm::fill_fragment(th[i][j], 0.f); wm::fill_fragment(tl[i][j], 0.f); }
#pragma unroll
        for (int kk = 0; kk < GK; kk += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> ah[2], al[2];
            wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> bf[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                wm::load_matrix_sync(ah[i], &Ah[wm_ * 32 + i * 16][kk], GK + GPAD);
                wm::load_matrix_sync(al[i], &Al[wm_ * 32 + i * 16][kk], GK + GPAD);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) wm::load_matrix_sync(bf[j], &Bs[wn * 32 + j * 16][kk], GK + GPAD);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wm::mma_sync(th[i][j], ah[i], bf[j], th[i][j]);
                    wm::mma_sync(tl[i][j], al[i], bf[j], tl[i][j]);
                }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < acc[i][j].num_elements; ++e) acc[i][j].x[e] += th[i][j].x[e] + tl[i][j].x[e];
        __syncthreads();
    }

    // Çıkış: her parça warp'ın kendi ara tamponundan ölçekle ve sınır denetimiyle yazılır
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wm::store_matrix_sync(&stage[warp][0][0], acc[i][j], 16 + 4, wm::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int rr = e / 16, cc = e % 16;
                const int gm = m0 + wm_ * 32 + i * 16 + rr, gn = n0 + wn * 32 + j * 16 + cc;
                if (gm < B && gn < rows) {
                    const float v = stage[warp][rr][cc] * inv_scale[gm];
                    float* p = Y + size_t(gm) * ldy + gn;
                    *p = ACC ? *p + v : v;
                }
            }
            __syncwarp();
        }
}

// ---------------------------------------------------------------- nicemli tensor core çarpımı

// mma.sync m16n8k8 (sm_75): A 16x8 satır düzeni, B 8x8 sütun düzeni, fp16 girdi, fp32 toplayıcı. Parça düzeni PTX
// belgesindeki gibidir: g = lane/4, t = lane%4;  a0 = A[g][2t..2t+1], a1 = A[g+8][2t..2t+1];  b0 = B[2t..2t+1][g];
// c0,c1 = C[g][2t, 2t+1], c2,c3 = C[g+8][2t, 2t+1]. WMMA'dan farkı: toplayıcının düzeni bilindiği için her çıkış
// sütununa (ağırlık satırına) kendi dilim ölçeği uygulanabilir.
__device__ __forceinline__ void mma_k8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

// 16 tamsayıyı (fp16'da tam temsil edilir) iki 16 baytlık sözcük olarak paylaşılan belleğe yazar
__device__ __forceinline__ void store16(__half* dst, const int (&q)[16]) {
    uint32_t w[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __half2 h = __halves2half2(__int2half_rn(q[2 * i]), __int2half_rn(q[2 * i + 1]));
        w[i] = *reinterpret_cast<const uint32_t*>(&h);
    }
    reinterpret_cast<uint4*>(dst)[0] = make_uint4(w[0], w[1], w[2], w[3]);
    reinterpret_cast<uint4*>(dst)[1] = make_uint4(w[4], w[5], w[6], w[7]);
}

// Bir ağırlık satırının k..k+15 değerlerinin tamsayı kısmını fp16 olarak yazar ve o 16'lık dilimin ölçeğiyle min'ini
// döndürür:  w = sc·q − mn  (sc ve mn ggml'deki gibi fp32'de hesaplanır). F16'da q = w, sc = 1.
template <int T>
__device__ __forceinline__ void decode16(const uint8_t* __restrict__ row, const __half* __restrict__ aux, int k,
                                         __half* dst, float& sc, float& mn);

template <>
__device__ __forceinline__ void decode16<W_F16>(const uint8_t* __restrict__ row, const __half* __restrict__, int k,
                                                __half* dst, float& sc, float& mn) {
    const uint4* src = reinterpret_cast<const uint4*>(row + 2 * size_t(k));
    reinterpret_cast<uint4*>(dst)[0] = __ldg(src);
    reinterpret_cast<uint4*>(dst)[1] = __ldg(src + 1);
    sc = 1.f;
    mn = 0.f;
}

template <>
__device__ __forceinline__ void decode16<W_Q8_0>(const uint8_t* __restrict__ row, const __half* __restrict__ aux,
                                                 int k, __half* dst, float& sc, float& mn) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(row + k));
    const int8_t* q8 = reinterpret_cast<const int8_t*>(&u);
    int q[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) q[i] = q8[i];
    store16(dst, q);
    sc = __half2float(aux[k / QK8_0]);
    mn = 0.f;
}

template <>
__device__ __forceinline__ void decode16<W_Q4_K>(const uint8_t* __restrict__ row, const __half* __restrict__, int k,
                                                 __half* dst, float& sc, float& mn) {
    const uint8_t* blk = row + size_t(k / QK_K) * sizeof(BlockQ4_K);
    const int o = k % QK_K, is = o / 32, l0 = o % 32;
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(blk + 16 + 32 * (is / 2) + l0));
    const uint8_t* qb = reinterpret_cast<const uint8_t*>(&u);
    const int shift = (is & 1) * 4;
    int q[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) q[i] = (qb[i] >> shift) & 0xF;
    store16(dst, q);
    const uint4 hd = __ldg(reinterpret_cast<const uint4*>(blk));  // d, dmin, scales[12]
    int s6, m6;
    q4k_sm(hd.y, hd.z, hd.w, is, s6, m6);
    sc = __fmul_rn(h2f(hd.x & 0xffffu), float(s6));
    mn = __fmul_rn(h2f(hd.x >> 16), float(m6));
}

template <>
__device__ __forceinline__ void decode16<W_Q6_K>(const uint8_t* __restrict__ row, const __half* __restrict__ aux,
                                                 int k,
                                                 __half* dst, float& sc, float& mn) {
    const uint8_t* blk = row + size_t(k / QK_K) * kQ6Bytes;
    const int o = k % QK_K, hf = o / 128, grp = (o % 128) / 32, l0 = o % 32;
    const uint8_t* ql = blk + 64 * hf + 32 * (grp & 1) + l0;
    const uint8_t* qh = blk + 128 + 32 * hf + l0;
    const int nshift = grp >= 2 ? 4 : 0, hshift = 2 * grp;
    int q[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        q[i] = int(((__ldg(ql + i) >> nshift) & 0xF) | (((__ldg(qh + i) >> hshift) & 3) << 4)) - 32;
    store16(dst, q);
    const float d = __half2float(aux[k / QK_K]);
    sc = __fmul_rn(d, float(__ldg(reinterpret_cast<const int8_t*>(blk + 192) + 8 * hf + l0 / 16 + 2 * grp)));
    mn = 0.f;
}

// Y[b][ldy] (+)= Σ_k (Xhi + Xlo)[b][k] · W[r][k] · inv_scale[b]
// Tensor core yalnızca 16'lık dilimlerin tamsayı ağırlıklı toplamlarını (Σ x·q) hesaplar; dilim ölçeği ve min terimi
// fp32'de uygulanıp ana toplayıcıya eklenir — ağırlık hiçbir aşamada fp16'ya yuvarlanmaz. Dilim başına hi ve lo
// ayrı toplanır (Turing tensor core'unun kesik iç toplaması lo'nun küçük katkısını yutmasın).
// Blok 64 (token) x 128 (ağırlık satırı), 8 warp (2x4), warp başına 32x32 = 2 (m16) x 4 (n8) parça; K adımı 32.
constexpr int QM = 64, QN = 128, QKS = 32, QPAD = 8;
template <int T, bool ACC>
__global__ void __launch_bounds__(256) k_gemm_q(const __half* __restrict__ Xh, const __half* __restrict__ Xl,
                                                const float* __restrict__ inv_scale, const float* __restrict__ xsum16,
                                                const uint8_t* __restrict__ W, const __half* __restrict__ aux,
                                                size_t row_bytes, int aux_per_row, float* Y, int ldy, int B, int rows,
                                                int n) {
    __shared__ __align__(16) __half Ah[QM][QKS + QPAD];
    __shared__ __align__(16) __half Al[QM][QKS + QPAD];
    __shared__ __align__(16) __half Bq[QN][QKS + QPAD];
    // Dilim ölçekleri [QN][2]; Q4_K'da ardından min'ler [QN][2] ve x'in dilim toplamları [QM][2]
    constexpr bool kMin = T == W_Q4_K;
    __shared__ float Ssc[QN * 2 + (kMin ? QN * 2 + QM * 2 : 0)];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm_ = warp / 4, wn = warp % 4;
    const int m0 = blockIdx.y * QM, n0 = blockIdx.x * QN;
    const uint4 zero = make_uint4(0, 0, 0, 0);

    float acc[2][4][4] = {};
    for (int k0 = 0; k0 < n; k0 += QKS) {
        {   // A: 64 satır x 32 half = satır başına 4 parça (16 bayt), iş parçacığı başına bir parça
            const int r = tid >> 2, c = (tid & 3) * 8, gm = m0 + r;
            const size_t gi = size_t(gm) * n + k0 + c;
            *reinterpret_cast<uint4*>(&Ah[r][c]) = gm < B ? *reinterpret_cast<const uint4*>(Xh + gi) : zero;
            *reinterpret_cast<uint4*>(&Al[r][c]) = gm < B ? *reinterpret_cast<const uint4*>(Xl + gi) : zero;
        }
        if constexpr (kMin) {
            if (tid < QM * 2) {
                const int r = tid >> 1, h = tid & 1, gm = m0 + r;
                Ssc[4 * QN + 2 * r + h] = gm < B ? xsum16[size_t(gm) * (n / 16) + k0 / 16 + h] : 0.f;
            }
        }
        {   // B: iş parçacığı başına bir ağırlık satırının bir 16'lık yarısı
            const int r = tid >> 1, h = tid & 1, gr = n0 + r;
            float sc = 0.f, mn = 0.f;
            if (gr < rows) {
                decode16<T>(W + size_t(gr) * row_bytes, aux + size_t(gr) * aux_per_row, k0 + 16 * h, &Bq[r][16 * h],
                            sc, mn);
            } else {
                reinterpret_cast<uint4*>(&Bq[r][16 * h])[0] = zero;
                reinterpret_cast<uint4*>(&Bq[r][16 * h])[1] = zero;
            }
            Ssc[2 * r + h] = sc;
            if constexpr (kMin) Ssc[2 * QN + 2 * r + h] = mn;
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float th[2][4][4] = {}, tl[2][4][4] = {};
#pragma unroll
            for (int k8 = 0; k8 < 2; ++k8) {
                const int kk = 16 * h + 8 * k8 + 2 * t;
                uint32_t ah[2][2], al[2][2], bf[4];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    const int R = wm_ * 32 + mi * 16 + g;
                    ah[mi][0] = *reinterpret_cast<const uint32_t*>(&Ah[R][kk]);
                    ah[mi][1] = *reinterpret_cast<const uint32_t*>(&Ah[R + 8][kk]);
                    al[mi][0] = *reinterpret_cast<const uint32_t*>(&Al[R][kk]);
                    al[mi][1] = *reinterpret_cast<const uint32_t*>(&Al[R + 8][kk]);
                }
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    bf[ni] = *reinterpret_cast<const uint32_t*>(&Bq[wn * 32 + ni * 8 + g][kk]);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni) {
                        mma_k8(th[mi][ni], ah[mi][0], ah[mi][1], bf[ni]);
                        mma_k8(tl[mi][ni], al[mi][0], al[mi][1], bf[ni]);
                    }
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int col = wn * 32 + ni * 8 + 2 * t + (e & 1);
                        float v = Ssc[2 * col + h] * (th[mi][ni][e] + tl[mi][ni][e]);
                        if constexpr (kMin)
                            v -= Ssc[2 * QN + 2 * col + h] * Ssc[4 * QN + 2 * (wm_ * 32 + mi * 16 + g + (e >= 2 ? 8 : 0)) + h];
                        acc[mi][ni][e] += v;
                    }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int gm = m0 + wm_ * 32 + mi * 16 + g + (e >= 2 ? 8 : 0);
                const int gn = n0 + wn * 32 + ni * 8 + 2 * t + (e & 1);
                if (gm < B && gn < rows) {
                    const float v = acc[mi][ni][e] * inv_scale[gm];
                    float* p = Y + size_t(gm) * ldy + gn;
                    *p = ACC ? *p + v : v;
                }
            }
}

// ---------------------------------------------------------------- nicemli tamsayı tensor core çarpımı

// x satırı satır başına s = 2^k ölçekle sabit noktaya çevrilir ve NL işaretli 8 bitlik parçaya bölünür:
//   X = rint(x·s) = Σ_l a_l·2^(8l),   satırın en büyük |x·s|'i [2^(8NL−3), 2^(8NL−2)) aralığında
// Parçalar tamsayıda alttan yukarı dengeli basamaklarla çıkarılır: a_l ∈ [−128, 127], en üstteki ∈ [−65, 65]
// (NL ≤ 4: |X| ≤ 2^30 int32'ye sığar).
// Parça l, a8 + l·limb_stride'da [B][n] düzeninde. xsum32 boş değilse 32'lik dilimlerin Σ X'i (fp32) yazılır.
template <int NL>
__global__ void k_split8(const float* __restrict__ X, int8_t* __restrict__ a8, size_t limb_stride,
                         float* __restrict__ inv_scale, float* __restrict__ xsum32, int n) {
    const size_t off = size_t(blockIdx.x) * n;
    float m = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, fabsf(X[off + i]));
    m = block_reduce<true>(m);
    int e = 0;
    if (m > 0.f) { frexpf(m, &e); }                // m = f·2^e, f ∈ [0.5, 1)
    const int k = m > 0.f ? 8 * NL - 2 - e : 0;     // m·2^k ∈ [2^(8NL−3), 2^(8NL−2))
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int v = __float2int_rn(ldexpf(X[off + i], k));
#pragma unroll
        for (int l = 0; l < NL - 1; ++l) {
            const int a = ((v + 128) & 255) - 128;
            a8[l * limb_stride + off + i] = int8_t(a);
            v = (v - a) >> 8;
        }
        a8[(NL - 1) * limb_stride + off + i] = int8_t(v);
    }
    if (threadIdx.x == 0) inv_scale[blockIdx.x] = ldexpf(1.0f, -k);
    if (xsum32) {
        for (int c = threadIdx.x; c < n / 32; c += blockDim.x) {
            float t = 0.f;
            for (int i = 0; i < 32; ++i) t += float(__float2int_rn(ldexpf(X[off + 32 * c + i], k)));
            xsum32[size_t(blockIdx.x) * (n / 32) + c] = t;
        }
    }
}

// mma.sync m8n8k16 s8 (sm_75): A 8x16 satır düzeni, B 16x8 sütun düzeni, int32 toplayıcı — çarpım ve toplama TAM.
// g = lane/4, t = lane%4;  a = A[g][4t..4t+3], b = B[4t..4t+3][g];  c0,c1 = C[g][2t, 2t+1].
__device__ __forceinline__ void mma_i8(int (&c)[2], uint32_t a, uint32_t b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a), "r"(b));
}

// ldmatrix (sm_75): lane grubu j (lane 8j..8j+7) j. matrisin satır adreslerini verir (16 bayt hizalı)
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

// |v| < 2^22 olan tamsayı → float, tam (1.5·2^23 sihirli sayısı; çeyrek hızlı I2F yerine)
__device__ __forceinline__ float i2f_small(int v) {
    return __int_as_float(0x4B400000 + v) - 12582912.0f;
}

// Y[b][ldy] (+)= Σ_k x[b][k] · w[r][k]
// Tensor core yalnızca tamsayıları çarpar: x'in NL 8 bitlik parçası × ağırlığın tamsayı kısmı q (Q4_K 0..15,
// Q6_K −32..31, Q8_0 −128..127). Alt blok ölçekleri de tamsayı olan K türlerinde (Q4_K 6 bit, Q6_K int8) toplam
// 256'lık blok boyunca int32'de kalır ve blok sonunda bir kez fp32'ye çevrilip d ile çarpılır:
//   Σ x·w = d · Σ_alt sc·(Σ X·q) − Σ_alt (dmin·m)·Σ X        (Q4_K; Q6_K'da min yok)
// Q8_0'da ölçek (fp16 d) her 32'lik blokta değişir; blok toplamı fp32'ye çevrilip d ile çarpılır.
// Blok 64 (token) x 128 (ağırlık satırı), 16 warp (4x4), warp başına 16x32 = 2 (m8) x 4 (n8) parça; K adımı 128.
constexpr int IM = 64, IN = 128, IKS = 128, IPAD = 16, KSB = IKS / 32, kLimbs = 4;
constexpr int IT = 512, WMI = 2, WNI = 4, WGN = IN / (8 * WNI);
// Dinamik paylaşılan bellek (48 KB'ı aşar): A [NL][IM][IKS+IPAD], B [IN][IKS+IPAD] int8; alt blok ölçekleri
// Ssc [2·KSB][IN] int, Sf [KSB][IN], Sd [IN], Xs [IM][KSB] float (ölçekler devrik: bir iş parçacığının yan yana iki
// çıktı sütunu tek 64 bitlik okuma). Satır adımı 144 bayt = 36 sözcük ≡ 4 (mod 32): ldmatrix'in 8 satırı (16'şar bayt)
// 32 bankın hepsine bir kez düşer, bank çakışmasız.
constexpr size_t i8_smem(int nl) {
    return size_t(nl) * IM * (IKS + IPAD) + size_t(IN) * (IKS + IPAD) + size_t(IN) * 2 * KSB * 4 +
           size_t(IN) * KSB * 4 + size_t(IN) * 4 + size_t(IM) * KSB * 4;
}
template <int T, int NL, bool ACC>
__global__ void __launch_bounds__(IT, 1) k_gemm_i8(const int8_t* __restrict__ a8, size_t limb_stride,
                                                    const float* __restrict__ inv_scale,
                                                    const float* __restrict__ xsum32, const uint8_t* __restrict__ W,
                                                    const __half* __restrict__ aux, size_t row_bytes, int aux_per_row,
                                                    float* Y, int ldy, int B, int rows, int n) {
    extern __shared__ __align__(16) uint8_t i8sm[];
    auto As = reinterpret_cast<int8_t (*)[IM][IKS + IPAD]>(i8sm);
    auto Bs = reinterpret_cast<int8_t (*)[IKS + IPAD]>(i8sm + size_t(NL) * IM * (IKS + IPAD));
    auto Ssc = reinterpret_cast<int (*)[IN]>(i8sm + size_t(NL) * IM * (IKS + IPAD) + size_t(IN) * (IKS + IPAD));
    auto Sf = reinterpret_cast<float (*)[IN]>(reinterpret_cast<uint8_t*>(Ssc) + size_t(IN) * 2 * KSB * 4);
    float* Sd = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(Sf) + size_t(IN) * KSB * 4);
    [[maybe_unused]] auto Xs = reinterpret_cast<float (*)[KSB]>(Sd + IN);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm_ = warp / WGN, wn = warp % WGN;
    const int m0 = blockIdx.y * IM, n0 = blockIdx.x * IN;
    const uint4 zero = make_uint4(0, 0, 0, 0);

    // [mi][ni][parça][e]: tamsayı toplam. Q6_K: parça başına, 256'lık blok boyunca. Q4_K: parçalar ikişer
    // birleşik (P0 + 256·P1, P2 [+ 256·P3]), iki alt dilimde bir boşaltılır: |2·63·61440·257| < 2^31.
    constexpr int NI = T == W_Q4_K ? (NL + 1) / 2 : NL;
    static_assert(NL >= 2 && NL <= 4, "2, 3 ya da 4 parça");
    [[maybe_unused]] int I[WMI][WNI][NI][2] = {};
    float acc[WMI][WNI][2] = {};
    constexpr int TPR = IT / IN;
    static_assert(KSB == TPR, "iş parçacığı başına tam bir B alt dilimi");
    const int r_b = tid / TPR, sbi_b = tid % TPR, gr_b = n0 + r_b;
    [[maybe_unused]] uint4 rb_q0 = zero, rb_q1 = zero, rb_hd = zero, rb_h0 = zero, rb_h1 = zero;
    [[maybe_unused]] __half rb_d = __float2half(0.f);
    [[maybe_unused]] uint32_t rb_sc = 0;
    auto load_b = [&](int kk) {
        if (gr_b >= rows) return;
        const uint8_t* row = W + size_t(gr_b) * row_bytes;
        [[maybe_unused]] const __half* ax = aux + size_t(gr_b) * aux_per_row;
        const int k = kk + 32 * sbi_b;
        if constexpr (T == W_Q8_0) {
            rb_q0 = __ldg(reinterpret_cast<const uint4*>(row + k));
            rb_q1 = __ldg(reinterpret_cast<const uint4*>(row + k + 16));
            rb_d = ax[k / QK8_0];
        } else if constexpr (T == W_Q4_K) {
            const uint8_t* blk = row + size_t(k / QK_K) * sizeof(BlockQ4_K);
            const int is = (k % QK_K) / 32;
            rb_hd = __ldg(reinterpret_cast<const uint4*>(blk));
            const uint4* qs = reinterpret_cast<const uint4*>(blk + 16 + 32 * (is / 2));
            rb_q0 = __ldg(qs);
            rb_q1 = __ldg(qs + 1);
        } else {
            const uint8_t* blk = row + size_t(k / QK_K) * kQ6Bytes;
            const int o = k % QK_K, hf = o / 128, grp = (o % 128) / 32;
            const uint4* ql = reinterpret_cast<const uint4*>(blk + 64 * hf + 32 * (grp & 1));
            const uint4* qh = reinterpret_cast<const uint4*>(blk + 128 + 32 * hf);
            rb_q0 = __ldg(ql);
            rb_q1 = __ldg(ql + 1);
            rb_h0 = __ldg(qh);
            rb_h1 = __ldg(qh + 1);
            rb_sc = __ldg(reinterpret_cast<const uint16_t*>(blk + 192 + 8 * hf + 2 * grp));
            rb_d = ax[k / QK_K];
        }
    };
    load_b(0);
    for (int k0 = 0; k0 < n; k0 += IKS) {
        constexpr int CPR = IKS / 16;  // A: satır başına 16 baytlık parça
#pragma unroll
        for (int i = 0; i < NL * IM * CPR / IT; ++i) {
            const int idx = tid + IT * i, l = idx / (IM * CPR), r = (idx % (IM * CPR)) / CPR, c = (idx % CPR) * 16;
            const int gm = m0 + r;
            *reinterpret_cast<uint4*>(&As[l][r][c]) =
                gm < B ? *reinterpret_cast<const uint4*>(a8 + l * limb_stride + size_t(gm) * n + k0 + c) : zero;
        }
        if constexpr (T == W_Q4_K) {
            if (tid < IM * KSB) {
                const int r = tid / KSB, h = tid % KSB, gm = m0 + r;
                Xs[r][h] = gm < B ? xsum32[size_t(gm) * (n / 32) + k0 / 32 + h] : 0.f;
            }
        }
        // B: iş parçacığı başına bir (satır, 32'lik alt dilim); ham baytlar bir adım önce yazmaçlara çekildi (rb_*)
        if (gr_b < rows) {
            uint4 q0, q1;
            if constexpr (T == W_Q8_0) {
                q0 = rb_q0;
                q1 = rb_q1;
                Sf[sbi_b][r_b] = __half2float(rb_d);
            } else if constexpr (T == W_Q4_K) {
                const int is = ((k0 + 32 * sbi_b) % QK_K) / 32, sh = (is & 1) * 4;
                q0 = make_uint4((rb_q0.x >> sh) & 0x0F0F0F0Fu, (rb_q0.y >> sh) & 0x0F0F0F0Fu, (rb_q0.z >> sh) & 0x0F0F0F0Fu,
                                (rb_q0.w >> sh) & 0x0F0F0F0Fu);
                q1 = make_uint4((rb_q1.x >> sh) & 0x0F0F0F0Fu, (rb_q1.y >> sh) & 0x0F0F0F0Fu, (rb_q1.z >> sh) & 0x0F0F0F0Fu,
                                (rb_q1.w >> sh) & 0x0F0F0F0Fu);
                int s6, m6;
                q4k_sm(rb_hd.y, rb_hd.z, rb_hd.w, is, s6, m6);
                Ssc[sbi_b][r_b] = s6;
                Sf[sbi_b][r_b] = __fmul_rn(h2f(rb_hd.x >> 16), float(m6));
                if (sbi_b == 0) Sd[r_b] = h2f(rb_hd.x & 0xffffu);
            } else {  // Q6_K (hizalı düzen): rb_q0/q1 = ql, rb_h0/h1 = qh
                const int o = (k0 + 32 * sbi_b) % QK_K, grp = (o % 128) / 32;
                const int ns = grp >= 2 ? 4 : 0, hs = 2 * grp;
                auto six = [&](uint32_t lw, uint32_t hw) {
                    return __vsub4(((lw >> ns) & 0x0F0F0F0Fu) | (((hw >> hs) & 0x03030303u) << 4), 0x20202020u);
                };
                q0 = make_uint4(six(rb_q0.x, rb_h0.x), six(rb_q0.y, rb_h0.y), six(rb_q0.z, rb_h0.z), six(rb_q0.w, rb_h0.w));
                q1 = make_uint4(six(rb_q1.x, rb_h1.x), six(rb_q1.y, rb_h1.y), six(rb_q1.z, rb_h1.z), six(rb_q1.w, rb_h1.w));
                Ssc[2 * sbi_b][r_b] = int(int8_t(rb_sc & 0xff));
                Ssc[2 * sbi_b + 1][r_b] = int(int8_t(rb_sc >> 8));
                if (sbi_b == 0) Sd[r_b] = __half2float(rb_d);
            }
            *reinterpret_cast<uint4*>(&Bs[r_b][32 * sbi_b]) = q0;
            *reinterpret_cast<uint4*>(&Bs[r_b][32 * sbi_b + 16]) = q1;
        } else {
            Ssc[2 * sbi_b][r_b] = 0;
            Ssc[2 * sbi_b + 1][r_b] = 0;
            Sf[sbi_b][r_b] = 0.f;
            if (sbi_b == 0) Sd[r_b] = 0.f;
            *reinterpret_cast<uint4*>(&Bs[r_b][32 * sbi_b]) = zero;
            *reinterpret_cast<uint4*>(&Bs[r_b][32 * sbi_b + 16]) = zero;
        }
        __syncthreads();
        if (k0 + IKS < n) load_b(k0 + IKS);  // sonraki adımın ağırlıkları: hesap sürerken yolda
#pragma unroll 1
        for (int sb = 0; sb < KSB; ++sb) {  // 32'lik alt dilim
            // B parçaları alt dilim boyunca, A parçaları her m8 satırı için yüklenir (az yazmaç, aynı sayıda okuma)
            // ldmatrix: 8x8'lik b16 matrisi = 8 satır x 16 bayt; iş parçacığı satır lane/4'ün 4t..4t+3 baytlarını alır, bu
            // tam olarak m8n8k16 s8 parçası (A satır düzeni, B'nin Bs[n][k] düzeni). Dört matris tek komutta.
            const int lj = lane >> 3, lr = lane & 7;
            uint32_t bf[2][4];  // [k16][ni]
            ldsm_x4(bf[0], &Bs[wn * 32 + lj * 8 + lr][32 * sb]);
            ldsm_x4(bf[1], &Bs[wn * 32 + lj * 8 + lr][32 * sb + 16]);
            [[maybe_unused]] int scs[4][2][1];  // [ni][e]: Q4_K alt blok tamsayı ölçeği (Q6_K'da ölçek kullanıldığı yerde okunur)
            [[maybe_unused]] float fs[4][2];    // [ni][e]: Q4_K dmin·m, Q8_0 d
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                [[maybe_unused]] const int c = wn * 32 + ni * 8 + 2 * t;  // çift: 8 bayt hizalı
                if constexpr (T == W_Q4_K) {
                    const int2 s = *reinterpret_cast<const int2*>(&Ssc[sb][c]);
                    scs[ni][0][0] = s.x;
                    scs[ni][1][0] = s.y;
                }
                if constexpr (T != W_Q6_K) {
                    const float2 f = *reinterpret_cast<const float2*>(&Sf[sb][c]);
                    fs[ni][0] = f.x;
                    fs[ni][1] = f.y;
                }
            }
#pragma unroll
            for (int mi = 0; mi < WMI; ++mi) {
                uint32_t af[2][NL];  // matris q = k16·NL + l; ldmatrix'te lane grubu lj (ve 4 + lj) bir matrisin satırlarını verir
                const int arow = wm_ * (8 * WMI) + mi * 8 + lr;
                {
                    uint32_t r[4];
                    ldsm_x4(r, &As[lj % NL][arow][32 * sb + 16 * (lj / NL)]);
#pragma unroll
                    for (int q = 0; q < 4; ++q) af[q / NL][q % NL] = r[q];
                }
                if constexpr (NL == 4) {
                    uint32_t r[4];
                    ldsm_x4(r, &As[lj][arow][32 * sb + 16]);
#pragma unroll
                    for (int q = 0; q < 4; ++q) af[1][q] = r[q];
                } else if constexpr (NL == 3) {  // kalan iki matris (k16 = 1, l = 1, 2); x2'de yalnız lane 0..15 adres verir
                    uint32_t r[2];
                    ldsm_x2(r, &As[1 + (lj & 1)][arow][32 * sb + 16]);
                    af[1][1] = r[0];
                    af[1][2] = r[1];
                }
                [[maybe_unused]] float xsm = 0.f;
                if constexpr (T == W_Q4_K) xsm = Xs[wm_ * (8 * WMI) + mi * 8 + g][sb];
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    if constexpr (T == W_Q6_K) {
#pragma unroll
                        for (int k16 = 0; k16 < 2; ++k16) {
                            int P[NL][2] = {};
#pragma unroll
                            for (int l = 0; l < NL; ++l) mma_i8(P[l], af[k16][l], bf[k16][ni]);
                            const int2 s = *reinterpret_cast<const int2*>(&Ssc[2 * sb + k16][wn * 32 + ni * 8 + 2 * t]);
#pragma unroll
                            for (int l = 0; l < NL; ++l) {
                                I[mi][ni][l][0] += s.x * P[l][0];
                                I[mi][ni][l][1] += s.y * P[l][1];
                            }
                        }
                    } else {
                        int P[NL][2] = {};
#pragma unroll
                        for (int k16 = 0; k16 < 2; ++k16)
#pragma unroll
                            for (int l = 0; l < NL; ++l) mma_i8(P[l], af[k16][l], bf[k16][ni]);
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            if constexpr (T == W_Q4_K) {
#pragma unroll
                                for (int h = 0; h < NI; ++h)
                                    I[mi][ni][h][e] +=
                                        scs[ni][e][0] * (P[2 * h][e] + (2 * h + 1 < NL ? 256 * P[min(2 * h + 1, NL - 1)][e] : 0));
                                acc[mi][ni][e] -= fs[ni][e] * xsm;
                            } else {  // Q8_0: |P| < 2^19
                                float f = i2f_small(P[NL - 1][e]);
#pragma unroll
                                for (int l = NL - 2; l >= 0; --l) f = fmaf(f, 256.0f, i2f_small(P[l][e]));
                                acc[mi][ni][e] += fs[ni][e] * f;
                            }
                        }
                    }
                }
            }
            if constexpr (T == W_Q4_K) {
                if (sb & 1) {  // iki alt dilim bitti: fp32'ye çevir, d ile çarp
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const float d = Sd[wn * 32 + ni * 8 + 2 * t + e];
#pragma unroll
                            for (int mi = 0; mi < WMI; ++mi) {
                                if constexpr (NI == 2)
                                    acc[mi][ni][e] += d * fmaf(float(I[mi][ni][NI - 1][e]), 65536.0f, float(I[mi][ni][0][e]));
                                else
                                    acc[mi][ni][e] += d * float(I[mi][ni][0][e]);
#pragma unroll
                                for (int h = 0; h < NI; ++h) I[mi][ni][h][e] = 0;
                            }
                        }
                }
            }
        }
        if constexpr (T == W_Q6_K) {
            if ((k0 + IKS) % QK_K == 0) {  // blok sonu: tamsayı toplamı fp32'ye çevir, d ile çarp
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float d = Sd[wn * 32 + ni * 8 + 2 * t + e];
#pragma unroll
                        for (int mi = 0; mi < WMI; ++mi) {
                            float f = float(I[mi][ni][NL - 1][e]);
#pragma unroll
                            for (int l = NL - 2; l >= 0; --l) f = fmaf(f, 256.0f, float(I[mi][ni][l][e]));
                            acc[mi][ni][e] += d * f;
#pragma unroll
                            for (int l = 0; l < NL; ++l) I[mi][ni][l][e] = 0;
                        }
                    }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < WMI; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int gm = m0 + wm_ * (8 * WMI) + mi * 8 + g;
                const int gn = n0 + wn * 32 + ni * 8 + 2 * t + e;
                if (gm < B && gn < rows) {
                    const float v = acc[mi][ni][e] * inv_scale[gm];
                    float* p = Y + size_t(gm) * ldy + gn;
                    *p = ACC ? *p + v : v;
                }
            }
}

// ---------------------------------------------------------------- dikkat

// Dikkat. q: satır b, kafa h → q + b*q_stride + h*D.  kc/vc: bu katmanın önbelleği, [konum][HKV][D].
// Sorgu b, pos+b konumundadır ve 0..pos+b anahtarlarını görür (nedensel).

// Çözme (tek sorgu), "flash-decoding": anahtarlar kAttnSplit parçaya bölünür; grid (H, kAttnSplit), blockDim = D.
// Her blok kendi parçası için normalize edilmemiş (m, l, acc) yazar; k_attn_combine birleştirir. Parça sayısı sabit
// olduğundan ızgara konumdan bağımsızdır (CUDA Graph ile uyumlu).
constexpr int kAttnSplit = 16;
__global__ void __launch_bounds__(kHeadDim) k_attn_split(const float* __restrict__ q, const float* __restrict__ kc,
                                                         const float* __restrict__ vc, float* __restrict__ part_acc,
                                                         float2* __restrict__ part_ml, const int* __restrict__ pos_p,
                                                         int H, int HKV, float scale) {
    constexpr int D = kHeadDim, KT = 32, PER_WARP = KT / (D / 32);
    __shared__ float qs[D];
    __shared__ float sc[KT];
    const int h = blockIdx.x, split = blockIdx.y, kvh = h / (H / HKV);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t kstride = size_t(HKV) * D;
    qs[tid] = q[size_t(h) * D + tid];
    __syncthreads();

    const int T = *pos_p + 1;
    const int chunk = (T + kAttnSplit - 1) / kAttnSplit;
    const int t_begin = split * chunk, t_end = min(T, t_begin + chunk);
    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (int t0 = t_begin; t0 < t_end; t0 += KT) {
        for (int j = 0; j < PER_WARP; ++j) {
            const int kk = warp * PER_WARP + j, t = t0 + kk;
            float s = -INFINITY;
            if (t < t_end) {
                const float* k = kc + size_t(t) * kstride + size_t(kvh) * D;
                float p = 0.f;
#pragma unroll
                for (int i = lane; i < D; i += 32) p += qs[i] * k[i];
                s = warp_sum(p) * scale;
            }
            if (lane == 0) sc[kk] = s;
        }
        __syncthreads();
        float tm = -INFINITY;
#pragma unroll
        for (int j = 0; j < KT; ++j) tm = fmaxf(tm, sc[j]);
        const float mn = fmaxf(m, tm);
        const float corr = expf(m - mn);  // m = -inf iken 0
        l *= corr;
        acc *= corr;
        const int lim = min(KT, t_end - t0);
        for (int j = 0; j < lim; ++j) {
            const float p = expf(sc[j] - mn);
            l += p;
            acc += p * vc[size_t(t0 + j) * kstride + size_t(kvh) * D + tid];
        }
        m = mn;
        __syncthreads();
    }
    const size_t slot = size_t(h) * kAttnSplit + split;
    part_acc[slot * D + tid] = acc;
    if (tid == 0) part_ml[slot] = make_float2(m, l);
}

// grid H, blockDim = D
__global__ void k_attn_combine(const float* __restrict__ part_acc, const float2* __restrict__ part_ml,
                               float* __restrict__ out) {
    const int h = blockIdx.x, tid = threadIdx.x, D = blockDim.x;
    float M = -INFINITY;
    for (int s = 0; s < kAttnSplit; ++s) M = fmaxf(M, part_ml[h * kAttnSplit + s].x);
    float l = 0.f, o = 0.f;
    for (int s = 0; s < kAttnSplit; ++s) {
        const float2 ml = part_ml[h * kAttnSplit + s];
        if (ml.x == -INFINITY) continue;  // boş parça
        const float w = expf(ml.x - M);
        l += w * ml.y;
        o += w * part_acc[(size_t(h) * kAttnSplit + s) * D + tid];
    }
    out[size_t(h) * D + tid] = o / l;
}

// Toplu istem, eski (EL_ATTN=tile32): grid (H, ceil(B/AQ)), 256 iş parçacığı (8 warp x 4 sorgu). Q karosu ve 32'lik K/V karoları paylaşılan
// bellekte; K satırları 129'a dolgulu (bank çakışması yok). Puan aşamasında lane = anahtar, çıktıda lane = boyut.
constexpr int AQ = 32, AK = 32;
constexpr size_t kAttnPrefillSmem = (size_t(AQ) * kHeadDim + size_t(AK) * (kHeadDim + 1) + size_t(AK) * kHeadDim) *
                                    sizeof(float);
__global__ void __launch_bounds__(256) k_attn_prefill(const float* __restrict__ q, int q_stride,
                                                      const float* __restrict__ kc, const float* __restrict__ vc,
                                                      float* __restrict__ out, const int* __restrict__ pos_p, int B,
                                                      int H, int HKV, float scale) {
    constexpr int D = kHeadDim, QW = AQ / 8;  // warp başına sorgu
    extern __shared__ float smem[];
    float* Qs = smem;                      // [AQ][D]
    float* Ks = Qs + AQ * D;               // [AK][D+1]
    float* Vs = Ks + AK * (D + 1);         // [AK][D]
    const int h = blockIdx.x, b0 = blockIdx.y * AQ, kvh = h / (H / HKV);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int pos0 = *pos_p;
    const size_t kstride = size_t(HKV) * D;

    for (int i = tid; i < AQ * D; i += 256) {
        const int r = i / D, d = i % D, b = b0 + r;
        Qs[i] = b < B ? q[size_t(b) * q_stride + size_t(h) * D + d] : 0.f;
    }
    float m[QW], l[QW], acc[QW][D / 32];
#pragma unroll
    for (int qi = 0; qi < QW; ++qi) {
        m[qi] = -INFINITY;
        l[qi] = 0.f;
#pragma unroll
        for (int i = 0; i < D / 32; ++i) acc[qi][i] = 0.f;
    }
    const int T = pos0 + min(B, b0 + AQ);  // bloktaki son sorgunun gördüğü anahtar sayısı
    for (int t0 = 0; t0 < T; t0 += AK) {
        __syncthreads();
        for (int i = tid; i < AK * D; i += 256) {
            const int r = i / D, d = i % D, t = t0 + r;
            float kv = 0.f, vv = 0.f;
            if (t < T) {
                const size_t o = size_t(t) * kstride + size_t(kvh) * D + d;
                kv = kc[o];
                vv = vc[o];
            }
            Ks[r * (D + 1) + d] = kv;
            Vs[r * D + d] = vv;
        }
        __syncthreads();

        float s[QW];
#pragma unroll
        for (int qi = 0; qi < QW; ++qi) s[qi] = 0.f;
        const float* kr = Ks + lane * (D + 1);
        for (int d = 0; d < D; ++d) {
            const float kd = kr[d];
#pragma unroll
            for (int qi = 0; qi < QW; ++qi) s[qi] += Qs[(warp * QW + qi) * D + d] * kd;
        }
        float p[QW];
#pragma unroll
        for (int qi = 0; qi < QW; ++qi) {
            const int b = b0 + warp * QW + qi, t = t0 + lane;
            const float sv = (b < B && t <= pos0 + b) ? s[qi] * scale : -INFINITY;
            const float mn = fmaxf(m[qi], warp_max(sv));
            const float base = mn == -INFINITY ? 0.f : mn;  // tümü maskeliyse NaN üretme
            const float corr = expf(m[qi] - base);
            p[qi] = expf(sv - base);
            l[qi] = l[qi] * corr + warp_sum(p[qi]);
#pragma unroll
            for (int i = 0; i < D / 32; ++i) acc[qi][i] *= corr;
            m[qi] = mn;
        }
        for (int j = 0; j < AK; ++j) {
            float v[D / 32];
#pragma unroll
            for (int i = 0; i < D / 32; ++i) v[i] = Vs[j * D + lane + 32 * i];
#pragma unroll
            for (int qi = 0; qi < QW; ++qi) {
                const float pj = __shfl_sync(0xffffffffu, p[qi], j);
#pragma unroll
                for (int i = 0; i < D / 32; ++i) acc[qi][i] += pj * v[i];
            }
        }
    }
#pragma unroll
    for (int qi = 0; qi < QW; ++qi) {
        const int b = b0 + warp * QW + qi;
        if (b >= B) continue;
#pragma unroll
        for (int i = 0; i < D / 32; ++i) out[(size_t(b) * H + h) * D + lane + 32 * i] = acc[qi][i] / l[qi];
    }
}

// Toplu istem, yazmaç döşemeli (varsayılan; yukarıdaki eski 32'lik çekirdek EL_ATTN=tile32 ile seçilir): grid (H, ceil(B/PQ)), 256 iş
// parçacığı = 16 x 16 (tx, ty). Blok bir kafanın 64 sorgusunu işler: her K/V karosu 64 sorguya hizmet eder (eskide 32;
// 6800 tokenlık istemde süre dikkatte, K/V trafiği yarıya iner). İş parçacığı satırları ty*4..+3; puanda anahtarlar
// tx ve tx+16 (4x2), çıktıda boyutlar tx*4..+3 ve 64+tx*4..+3 (4x8), hepsi yazmaçta. Q ve K/V satır düzeninde 132'ye
// dolgulu (float4 hizalı, okumalarda bank çakışması yok), P devrik [anahtar][sorgu] 68'e dolgulu. K ve V aynı tamponu
// sırayla kullanır; sıradaki karo hesap sürerken yazmaçlara yüklenir. Hesap fp32; yalnız toplama sırası farklı.
constexpr int PQ = 64, PK = 32, PQS = kHeadDim + 4, PPS = PQ + 4;
constexpr size_t kAttnPrefill64Smem = (size_t(PQ) * PQS + size_t(PK) * PQS + size_t(PK) * PPS) * sizeof(float);
__global__ void __launch_bounds__(256, 1) k_attn_prefill64(const float* __restrict__ q, int q_stride,
                                                           const float* __restrict__ kc, const float* __restrict__ vc,
                                                           float* __restrict__ out, const int* __restrict__ pos_p,
                                                           int B, int H, int HKV, float scale) {
    constexpr int D = kHeadDim;
    extern __shared__ float4 smem4[];
    float* Qs = reinterpret_cast<float*>(smem4);  // [PQ][PQS]
    float* KV = Qs + PQ * PQS;                     // [PK][PQS]: önce K, sonra V
    float* Pt = KV + PK * PQS;                     // [PK][PPS]
    const int h = blockIdx.x, b0 = blockIdx.y * PQ, kvh = h / (H / HKV);
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int pos0 = *pos_p;
    const size_t kstride = size_t(HKV) * D;
    const float* kbase = kc + size_t(kvh) * D;
    const float* vbase = vc + size_t(kvh) * D;

    for (int i = tid; i < PQ * D / 4; i += 256) {
        const int r = i / (D / 4), c = (i % (D / 4)) * 4, b = b0 + r;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (b < B) v = *reinterpret_cast<const float4*>(q + size_t(b) * q_stride + size_t(h) * D + c);
        *reinterpret_cast<float4*>(Qs + r * PQS + c) = v;
    }
    const int T = pos0 + min(B, b0 + PQ);  // bloktaki son sorgunun gördüğü anahtar sayısı
    float4 pre[4];                         // sıradaki K ya da V karosu (32 x 128 = 1024 float4, iş parçacığı başına 4)
    auto fetch = [&](const float* base, int t0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + 256 * j, t = t0 + (i >> 5);
            pre[j] = t < T ? *reinterpret_cast<const float4*>(base + size_t(t) * kstride + (i & 31) * 4)
                           : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    auto stash = [&]() {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + 256 * j;
            *reinterpret_cast<float4*>(KV + (i >> 5) * PQS + (i & 31) * 4) = pre[j];
        }
    };

    float m[4], l[4], o[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.f;
#pragma unroll
        for (int c = 0; c < 8; ++c) o[i][c] = 0.f;
    }
    fetch(kbase, 0);
    for (int t0 = 0; t0 < T; t0 += PK) {
        __syncthreads();  // önceki karonun V'si ve P'si okundu (ilk turda: Q yazıldı)
        stash();
        __syncthreads();
        fetch(vbase, t0);  // V, puanlar hesaplanırken gelir

        float s[4][2];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = 0.f;
        const float* q0 = Qs + ty * 4 * PQS;
        const float* k0 = KV + tx * PQS;
#pragma unroll 4
        for (int d = 0; d < D; d += 4) {
            const float4 ka = *reinterpret_cast<const float4*>(k0 + d);
            const float4 kb = *reinterpret_cast<const float4*>(k0 + 16 * PQS + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 qv = *reinterpret_cast<const float4*>(q0 + i * PQS + d);
                s[i][0] = fmaf(qv.w, ka.w, fmaf(qv.z, ka.z, fmaf(qv.y, ka.y, fmaf(qv.x, ka.x, s[i][0]))));
                s[i][1] = fmaf(qv.w, kb.w, fmaf(qv.z, kb.z, fmaf(qv.y, kb.y, fmaf(qv.x, kb.x, s[i][1]))));
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int b = b0 + ty * 4 + i;
            const bool row = b < B;
#pragma unroll
            for (int j = 0; j < 2; ++j) s[i][j] = row && t0 + tx + 16 * j <= pos0 + b ? s[i][j] * scale : -INFINITY;
            float mx = fmaxf(s[i][0], s[i][1]);
#pragma unroll
            for (int off = 8; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));  // satırın 16 iş parçacığı
            const float mn = fmaxf(m[i], mx);
            const float base = mn == -INFINITY ? 0.f : mn;  // tümü maskeliyse NaN üretme
            const float corr = expf(m[i] - base);
            s[i][0] = expf(s[i][0] - base);
            s[i][1] = expf(s[i][1] - base);
            l[i] = l[i] * corr + (s[i][0] + s[i][1]);  // iş parçacığının kısmi toplamı; sonda satırca toplanır
#pragma unroll
            for (int c = 0; c < 8; ++c) o[i][c] *= corr;
            m[i] = mn;
        }
#pragma unroll
        for (int j = 0; j < 2; ++j)
            *reinterpret_cast<float4*>(Pt + (tx + 16 * j) * PPS + ty * 4) = make_float4(s[0][j], s[1][j], s[2][j], s[3][j]);
        __syncthreads();  // K okundu, P yazıldı
        stash();
        __syncthreads();
        if (t0 + PK < T) fetch(kbase, t0 + PK);  // sıradaki K, çıktı toplanırken gelir

#pragma unroll 4
        for (int k = 0; k < PK; ++k) {
            const float4 p = *reinterpret_cast<const float4*>(Pt + k * PPS + ty * 4);
            const float4 va = *reinterpret_cast<const float4*>(KV + k * PQS + tx * 4);
            const float4 vb = *reinterpret_cast<const float4*>(KV + k * PQS + 64 + tx * 4);
            const float pr[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                o[i][0] = fmaf(pr[i], va.x, o[i][0]);
                o[i][1] = fmaf(pr[i], va.y, o[i][1]);
                o[i][2] = fmaf(pr[i], va.z, o[i][2]);
                o[i][3] = fmaf(pr[i], va.w, o[i][3]);
                o[i][4] = fmaf(pr[i], vb.x, o[i][4]);
                o[i][5] = fmaf(pr[i], vb.y, o[i][5]);
                o[i][6] = fmaf(pr[i], vb.z, o[i][6]);
                o[i][7] = fmaf(pr[i], vb.w, o[i][7]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float li = l[i];
#pragma unroll
        for (int off = 8; off; off >>= 1) li += __shfl_xor_sync(0xffffffffu, li, off);
        const int b = b0 + ty * 4 + i;
        if (b >= B) continue;
        float* dst = out + (size_t(b) * H + h) * D;
        *reinterpret_cast<float4*>(dst + tx * 4) = make_float4(o[i][0] / li, o[i][1] / li, o[i][2] / li, o[i][3] / li);
        *reinterpret_cast<float4*>(dst + 64 + tx * 4) =
            make_float4(o[i][4] / li, o[i][5] / li, o[i][6] / li, o[i][7] / li);
    }
}

// Eski tek-blok dikkat (karşılaştırma için tutulur, EL_ATTN=naive): grid (H, B), blockDim = D.
__global__ void __launch_bounds__(kHeadDim) k_attn(const float* __restrict__ q, int q_stride,
                                                   const float* __restrict__ kc, const float* __restrict__ vc,
                                                   float* __restrict__ out, const int* __restrict__ pos_p, int H,
                                                   int HKV, float scale) {
    constexpr int D = kHeadDim, KT = 32, PER_WARP = KT / (D / 32);
    __shared__ float qs[D];
    __shared__ float sc[KT];
    const int h = blockIdx.x, b = blockIdx.y, kvh = h / (H / HKV);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t kstride = size_t(HKV) * D;
    qs[tid] = q[size_t(b) * q_stride + size_t(h) * D + tid];
    __syncthreads();

    const int T = *pos_p + b + 1;
    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (int t0 = 0; t0 < T; t0 += KT) {
        for (int j = 0; j < PER_WARP; ++j) {
            const int kk = warp * PER_WARP + j, t = t0 + kk;
            float s = -INFINITY;
            if (t < T) {
                const float* k = kc + size_t(t) * kstride + size_t(kvh) * D;
                float p = 0.f;
#pragma unroll
                for (int i = lane; i < D; i += 32) p += qs[i] * k[i];
                s = warp_sum(p) * scale;
            }
            if (lane == 0) sc[kk] = s;
        }
        __syncthreads();
        float tm = -INFINITY;
#pragma unroll
        for (int j = 0; j < KT; ++j) tm = fmaxf(tm, sc[j]);
        const float mn = fmaxf(m, tm);
        const float corr = expf(m - mn);  // m = -inf iken 0
        l *= corr;
        acc *= corr;
        const int lim = min(KT, T - t0);
        for (int j = 0; j < lim; ++j) {
            const float p = expf(sc[j] - mn);
            l += p;
            acc += p * vc[size_t(t0 + j) * kstride + size_t(kvh) * D + tid];
        }
        m = mn;
        __syncthreads();
    }
    out[(size_t(b) * H + h) * D + tid] = acc / l;
}

// ---------------------------------------------------------------- bellek

struct DevBuf {
    void* p = nullptr;
    DevBuf() = default;
    explicit DevBuf(size_t n) { CK(cudaMalloc(&p, n)); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p(o.p) { o.p = nullptr; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) { release(); p = o.p; o.p = nullptr; }
        return *this;
    }
    ~DevBuf() { release(); }
    void release() { if (p) cudaFree(p); p = nullptr; }
    template <class T> T* as() const { return static_cast<T*>(p); }
};

DevBuf upload(const void* src, size_t bytes) {
    DevBuf b(bytes);
    CK(cudaMemcpy(b.p, src, bytes, cudaMemcpyHostToDevice));
    return b;
}

DevBuf upload_f32(const Tensor& t) {
    std::vector<float> v = to_f32(t);
    return upload(v.data(), v.size() * sizeof(float));
}

WType wtype_of(const Tensor& t) {
    switch (t.type) {
        case GgmlType::F16: return W_F16;
        case GgmlType::Q8_0: return W_Q8_0;
        case GgmlType::Q4_K: return W_Q4_K;
        case GgmlType::Q6_K: return W_Q6_K;
        default:
            throw std::runtime_error(std::string("CUDA yolu bu agirlik turunu desteklemiyor: ") + t.name + " " +
                                     type_name(t.type));
    }
}

// VRAM'deki ağırlık matrisi: rows satır, satır uzunluğu n. F16 ve Q4_K dosyadaki bayt düzeninde. Hizalı okuma için
// Q8_0 int8 [rows][n] + aux = fp16 d [rows][n/32]; Q6_K 208 baytlık bloklar + aux = fp16 d [rows][n/256] olarak yeniden
// düzenlenir (bitler aynı, yalnızca yerleri değişir).
struct QMat {
    WType type = W_F16;
    int rows = 0, n = 0;
    size_t row_bytes = 0;
    int aux_per_row = 0;
    DevBuf w, aux;
};
// Doğrusal katman: çıktıda col sütunundan başlayan parçalar (aynı türdeki komşu tensörler tek parça)
struct Seg {
    QMat m;
    int col = 0;
};
using Lin = std::vector<Seg>;

QMat upload_group(const std::vector<const Tensor*>& parts) {
    QMat m;
    m.type = wtype_of(*parts[0]);
    m.n = int(parts[0]->ne[0]);
    uint64_t total = 0;
    for (const Tensor* t : parts) {
        m.rows += int(t->rows());
        total += t->nbytes;
    }
    if (m.n % 32 != 0) throw std::runtime_error("satir uzunlugu 32'nin kati olmali: " + parts[0]->name);
    if (m.type == W_Q8_0) {
        const size_t nb = size_t(m.n) / QK8_0;
        std::vector<int8_t> q(size_t(m.rows) * m.n);
        std::vector<uint16_t> d(size_t(m.rows) * nb);
        size_t r = 0;
        for (const Tensor* t : parts) {
            const auto* blk = reinterpret_cast<const BlockQ8_0*>(t->data);
            for (uint64_t i = 0; i < t->rows(); ++i, ++r)
                for (size_t b = 0; b < nb; ++b) {
                    const BlockQ8_0& x = blk[i * nb + b];
                    d[r * nb + b] = x.d;
                    std::memcpy(&q[r * m.n + b * QK8_0], x.qs, QK8_0);
                }
        }
        m.row_bytes = size_t(m.n);
        m.aux_per_row = int(nb);
        m.w = upload(q.data(), q.size());
        m.aux = upload(d.data(), d.size() * sizeof(uint16_t));
    } else if (m.type == W_Q6_K) {
        const size_t nb = size_t(m.n) / QK_K;
        std::vector<uint8_t> q(size_t(m.rows) * nb * kQ6Bytes);
        std::vector<uint16_t> d(size_t(m.rows) * nb);
        size_t r = 0;
        for (const Tensor* t : parts) {
            const auto* blk = reinterpret_cast<const BlockQ6_K*>(t->data);
            for (uint64_t i = 0; i < t->rows(); ++i, ++r)
                for (size_t b = 0; b < nb; ++b) {
                    const BlockQ6_K& x = blk[i * nb + b];
                    std::memcpy(&q[(r * nb + b) * kQ6Bytes], &x, kQ6Bytes);  // ql, qh, scales (d hariç)
                    d[r * nb + b] = x.d;
                }
        }
        m.row_bytes = nb * kQ6Bytes;
        m.aux_per_row = int(nb);
        m.w = upload(q.data(), q.size());
        m.aux = upload(d.data(), d.size() * sizeof(uint16_t));
    } else {
        m.row_bytes = size_t(total / uint64_t(m.rows));
        m.w = DevBuf(size_t(total));
        size_t off = 0;
        for (const Tensor* t : parts) {
            CK(cudaMemcpy(static_cast<char*>(m.w.p) + off, t->data, size_t(t->nbytes), cudaMemcpyHostToDevice));
            off += size_t(t->nbytes);
        }
    }
    return m;
}

// Çıktıları yan yana gelen tensörler (ör. Q|K|V); aynı türdeki ardışık olanlar birleştirilir
Lin upload_lin(std::initializer_list<const Tensor*> parts) {
    Lin out;
    std::vector<const Tensor*> group;
    int col = 0;
    auto flush = [&] {
        if (group.empty()) return;
        Seg s;
        s.m = upload_group(group);
        s.col = col;
        col += s.m.rows;
        out.push_back(std::move(s));
        group.clear();
    };
    for (const Tensor* t : parts) {
        if (!group.empty() && (group[0]->type != t->type || group[0]->ne[0] != t->ne[0])) flush();
        group.push_back(t);
    }
    flush();
    return out;
}

bool env_is(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, value) == 0;
}

// ---------------------------------------------------------------- model

class CudaModel final : public Backend {
public:
    CudaModel(const Gguf& g, int n_ctx, int max_batch) : c_(read_config(g)), n_ctx_(n_ctx), max_batch_(max_batch) {
        if (n_ctx_ <= 0) throw std::runtime_error("baglam uzunlugu pozitif olmali");
        ModelTensors t = read_tensors(g, c_);
        if (c_.head_dim != kHeadDim) throw std::runtime_error("CUDA yolu su an yalnizca head_dim=128 destekliyor");
        use_tc_ = !env_is("EL_GEMM", "simt");
        f16_mma_ = env_is("EL_GEMM", "mma");
        q_hilo_ = env_is("EL_GEMM", "hilo");
        if (const char* v = std::getenv("EL_NL3")) nl3_ = std::atoi(v);
        if (const char* v = std::getenv("EL_NL2")) nl2_ = std::atoi(v);
        use_graph_ = !env_is("EL_NO_GRAPH", "1");
        naive_attn_ = env_is("EL_ATTN", "naive");
        tile32_attn_ = env_is("EL_ATTN", "tile32");
        cudaDeviceProp prop{};
        CK(cudaGetDeviceProperties(&prop, 0));
        sm_count_ = prop.multiProcessorCount;
        CK(cudaFuncSetAttribute(k_attn_prefill, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kAttnPrefillSmem)));
        CK(cudaFuncSetAttribute(k_attn_prefill64, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kAttnPrefill64Smem)));
        for (const void* f : {(const void*)k_gemm_i8<W_Q8_0, 4, false>, (const void*)k_gemm_i8<W_Q8_0, 4, true>,
                              (const void*)k_gemm_i8<W_Q4_K, 4, false>, (const void*)k_gemm_i8<W_Q4_K, 4, true>,
                              (const void*)k_gemm_i8<W_Q6_K, 4, false>, (const void*)k_gemm_i8<W_Q6_K, 4, true>})
            CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, int(i8_smem(4))));
        for (const void* f : {(const void*)k_gemm_i8<W_Q8_0, 2, false>, (const void*)k_gemm_i8<W_Q8_0, 2, true>,
                              (const void*)k_gemm_i8<W_Q4_K, 2, false>, (const void*)k_gemm_i8<W_Q4_K, 2, true>,
                              (const void*)k_gemm_i8<W_Q6_K, 2, false>, (const void*)k_gemm_i8<W_Q6_K, 2, true>})
            CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, int(i8_smem(2))));
        for (const void* f : {(const void*)k_gemm_i8<W_Q8_0, 3, false>, (const void*)k_gemm_i8<W_Q8_0, 3, true>,
                              (const void*)k_gemm_i8<W_Q4_K, 3, false>, (const void*)k_gemm_i8<W_Q4_K, 3, true>,
                              (const void*)k_gemm_i8<W_Q6_K, 3, false>, (const void*)k_gemm_i8<W_Q6_K, 3, true>})
            CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, int(i8_smem(3))));

        const size_t E = size_t(c_.n_embd), F = size_t(c_.n_ff), D = size_t(c_.head_dim);
        const size_t QD = D * c_.n_head, KD = D * c_.n_head_kv, QKV = QD + 2 * KD, B = size_t(max_batch_);
        const size_t kv_bytes = size_t(c_.n_layer) * size_t(n_ctx_) * KD * sizeof(float);
        const size_t maxn = std::max({E, QD, F});
        if ((maxn + maxn / 16) * sizeof(float) > 48 * 1024)
            throw std::runtime_error("gemv: satir uzunlugu paylasilan bellege sigmiyor (en fazla 12288)");

        // VRAM yetecek mi? (ağırlık + KV + tamponlar; sürücüye pay bırak). Gömme tablosu yalnızca bağlı çıkış
        // katmanı olarak VRAM'e girer (t.output bağlıysa tok_embd'dir).
        uint64_t need = kv_bytes * 2 + (B * (2 * E + QKV + QD + 3 * F) + c_.n_vocab) * sizeof(float);
        need += B * maxn * 2 * sizeof(__half) + B * (maxn / 16) * sizeof(float);
        need += size_t(kLimbs) * B * maxn + B * (maxn / 32) * sizeof(float);
        need += t.output->nbytes;
        for (const LayerTensors& L : t.layers)
            for (const Tensor* w : {L.wq, L.wk, L.wv, L.wo, L.gate, L.up, L.down}) need += w->nbytes;
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        if (need + (256ull << 20) > free_b) {
            char msg[200];
            std::snprintf(msg, sizeof msg, "GPU bellegi yetersiz: %.0f MB gerekli, %.0f MB bos (--ctx kucultun)",
                          need / 1048576.0, free_b / 1048576.0);
            throw std::runtime_error(msg);
        }

        CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        emb_ = t.tok_embd;
        lm_head_ = upload_lin({t.output});
        output_norm_ = upload_f32(*t.output_norm);
        for (const LayerTensors& L : t.layers) {
            Layer d;
            d.attn_norm = upload_f32(*L.attn_norm);
            d.ffn_norm = upload_f32(*L.ffn_norm);
            d.q_norm = upload_f32(*L.q_norm);
            d.k_norm = upload_f32(*L.k_norm);
            d.qkv = upload_lin({L.wq, L.wk, L.wv});
            d.wo = upload_lin({L.wo});
            d.gu = upload_lin({L.gate, L.up});
            d.down = upload_lin({L.down});
            layers_.push_back(std::move(d));
        }

        // RoPE tablosu: açı çift hassasiyette (CPU başvurusuyla aynı), sonra float
        const std::vector<double> inv = rope_inv_freq(c_);
        const size_t half = inv.size();
        std::vector<float2> cs(size_t(n_ctx_) * half);
        for (size_t p = 0; p < size_t(n_ctx_); ++p)
            for (size_t i = 0; i < half; ++i) {
                const double th = double(p) * inv[i];
                cs[p * half + i] = make_float2(float(std::cos(th)), float(std::sin(th)));
            }
        rope_ = upload(cs.data(), cs.size() * sizeof(float2));

        kcache_ = DevBuf(kv_bytes);
        vcache_ = DevBuf(kv_bytes);
        x_ = DevBuf(B * E * 4); h_ = DevBuf(B * E * 4);
        qkv_ = DevBuf(B * QKV * 4); att_ = DevBuf(B * QD * 4);
        gu_ = DevBuf(B * 2 * F * 4); act_ = DevBuf(B * F * 4);
        xh_ = DevBuf(B * maxn * 2); xl_ = DevBuf(B * maxn * 2); xs_ = DevBuf(B * 4);
        xsum_ = DevBuf(B * (maxn / 16) * 4);
        x8_ = DevBuf(size_t(kLimbs) * B * maxn); xs8_ = DevBuf(B * 4); xsum32_ = DevBuf(B * (maxn / 32) * 4);
        logits_ = DevBuf(size_t(c_.n_vocab) * 4);
        part_acc_ = DevBuf(size_t(c_.n_head) * kAttnSplit * D * 4);
        part_ml_ = DevBuf(size_t(c_.n_head) * kAttnSplit * sizeof(float2));
        pos_ = DevBuf(sizeof(int));
        CK(cudaMallocHost(&pos_host_, sizeof(int)));
        CK(cudaMallocHost(&emb_host_, B * E * sizeof(float)));
    }

    ~CudaModel() override {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (emb_host_) cudaFreeHost(emb_host_);
        if (pos_host_) cudaFreeHost(pos_host_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    const Config& cfg() const override { return c_; }
    int n_ctx() const override { return n_ctx_; }
    const char* name() const override { return "cuda"; }

    void eval(const int32_t* tokens, int n, int pos0, float* logits) override {
        if (n <= 0) throw std::invalid_argument("bos token dizisi");
        if (pos0 < 0 || pos0 + n > n_ctx_) throw std::out_of_range("baglam penceresi doldu");
        for (int i = 0; i < n; ++i)
            if (tokens[i] < 0 || tokens[i] >= c_.n_vocab) throw std::out_of_range("token id sinir disi");
        for (int done = 0; done < n;) {
            const int b = std::min(max_batch_, n - done);
            chunk(tokens + done, b, pos0 + done, done + b == n ? logits : nullptr);
            done += b;
        }
    }

private:
    struct Layer {
        DevBuf attn_norm, ffn_norm, q_norm, k_norm;
        Lin qkv, wo, gu, down;
    };

    template <int T>
    void gemv_t(const QMat& m, const float* x, float* y, bool acc) {
        // x paylaşılan bellekte; büyükse (ffn_down) SM'ye tek blok sığar → blok 1024 iş parçacığı olur ki SM yine 32 warp
        // çalıştırsın. Küçükse 256'lık bloklardan SM başına 4.
        const int threads = m.n * int(sizeof(float)) > 16 * 1024 ? 1024 : 256, rows_per = threads / 32;
        const dim3 grid(std::min((m.rows + rows_per - 1) / rows_per, sm_count_ * (1024 / threads)));
        const size_t sh = size_t(m.n + (T == W_Q4_K ? m.n / 16 : 0)) * sizeof(float);
        const uint8_t* w = m.w.as<uint8_t>();
        const __half* aux = m.aux.as<__half>();
        if (acc) k_gemv<T, true><<<grid, threads, sh, stream_>>>(w, aux, m.row_bytes, m.aux_per_row, x, y, m.rows, m.n);
        else k_gemv<T, false><<<grid, threads, sh, stream_>>>(w, aux, m.row_bytes, m.aux_per_row, x, y, m.rows, m.n);
    }

    template <int T>
    void gemm_q_t(const QMat& m, float* y, int ldy, int B, bool acc) {
        const dim3 grid((m.rows + QN - 1) / QN, (B + QM - 1) / QM);
        const __half *xh = xh_.as<__half>(), *xl = xl_.as<__half>(), *aux = m.aux.as<__half>();
        const float *xs = xs_.as<float>(), *xsum = xsum_.as<float>();
        const uint8_t* w = m.w.as<uint8_t>();
        if (acc) k_gemm_q<T, true><<<grid, 256, 0, stream_>>>(xh, xl, xs, xsum, w, aux, m.row_bytes, m.aux_per_row, y,
                                                              ldy, B, m.rows, m.n);
        else k_gemm_q<T, false><<<grid, 256, 0, stream_>>>(xh, xl, xs, xsum, w, aux, m.row_bytes, m.aux_per_row, y,
                                                           ldy, B, m.rows, m.n);
    }

    template <int T, int NL>
    void gemm_i8_t(const QMat& m, float* y, int ldy, int B, bool acc) {
        const dim3 grid((m.rows + IN - 1) / IN, (B + IM - 1) / IM);
        const size_t ls = size_t(B) * m.n;
        const int8_t* a8 = x8_.as<int8_t>();
        const float *xs = xs8_.as<float>(), *xsum = xsum32_.as<float>();
        const uint8_t* w = m.w.as<uint8_t>();
        const __half* aux = m.aux.as<__half>();
        if (acc) k_gemm_i8<T, NL, true><<<grid, IT, i8_smem(NL), stream_>>>(a8, ls, xs, xsum, w, aux, m.row_bytes, m.aux_per_row, y,
                                                               ldy, B, m.rows, m.n);
        else k_gemm_i8<T, NL, false><<<grid, IT, i8_smem(NL), stream_>>>(a8, ls, xs, xsum, w, aux, m.row_bytes, m.aux_per_row, y,
                                                            ldy, B, m.rows, m.n);
    }

    // Y[B][ldy] (+)= X · Wᵀ ; B=1 ise gemv, değilse tensor core (F16'da isteğe göre fp32 karo) çarpımı
    void linear(const float* X, const Lin& W, float* Y, int B, int ldy, int n, bool acc, int nl = kLimbs) {
        if (B == 1) {
            for (const Seg& s : W) {
                switch (s.m.type) {
                    case W_F16: gemv_t<W_F16>(s.m, X, Y + s.col, acc); break;
                    case W_Q8_0: gemv_t<W_Q8_0>(s.m, X, Y + s.col, acc); break;
                    case W_Q4_K: gemv_t<W_Q4_K>(s.m, X, Y + s.col, acc); break;
                    case W_Q6_K: gemv_t<W_Q6_K>(s.m, X, Y + s.col, acc); break;
                }
                CK_LAUNCH();
            }
            return;
        }
        bool split = false, need_xsum = false, split8 = false;
        for (const Seg& s : W) need_xsum |= s.m.type == W_Q4_K;
        for (const Seg& s : W) {
            const QMat& m = s.m;
            float* y = Y + s.col;
            if (m.type != W_F16 && !q_hilo_ && n % IKS == 0) {  // nicemli: tamsayı tensor core
                if (!split8) {
                    float* xsum = need_xsum ? xsum32_.as<float>() : nullptr;
                    if (nl == 2) k_split8<2><<<B, 256, 0, stream_>>>(X, x8_.as<int8_t>(), size_t(B) * n, xs8_.as<float>(), xsum, n);
                    else if (nl == 3) k_split8<3><<<B, 256, 0, stream_>>>(X, x8_.as<int8_t>(), size_t(B) * n, xs8_.as<float>(), xsum, n);
                    else k_split8<4><<<B, 256, 0, stream_>>>(X, x8_.as<int8_t>(), size_t(B) * n, xs8_.as<float>(), xsum, n);
                    CK_LAUNCH();
                    split8 = true;
                }
                switch (m.type * 8 + nl) {
                    case W_Q8_0 * 8 + 2: gemm_i8_t<W_Q8_0, 2>(m, y, ldy, B, acc); break;
                    case W_Q4_K * 8 + 2: gemm_i8_t<W_Q4_K, 2>(m, y, ldy, B, acc); break;
                    case W_Q6_K * 8 + 2: gemm_i8_t<W_Q6_K, 2>(m, y, ldy, B, acc); break;
                    case W_Q8_0 * 8 + 3: gemm_i8_t<W_Q8_0, 3>(m, y, ldy, B, acc); break;
                    case W_Q8_0 * 8 + 4: gemm_i8_t<W_Q8_0, 4>(m, y, ldy, B, acc); break;
                    case W_Q4_K * 8 + 3: gemm_i8_t<W_Q4_K, 3>(m, y, ldy, B, acc); break;
                    case W_Q4_K * 8 + 4: gemm_i8_t<W_Q4_K, 4>(m, y, ldy, B, acc); break;
                    case W_Q6_K * 8 + 3: gemm_i8_t<W_Q6_K, 3>(m, y, ldy, B, acc); break;
                    case W_Q6_K * 8 + 4: gemm_i8_t<W_Q6_K, 4>(m, y, ldy, B, acc); break;
                    default: break;
                }
                CK_LAUNCH();
                continue;
            }
            if (m.type == W_F16 && !use_tc_) {
                const dim3 grid((m.rows + TN - 1) / TN, (B + TM - 1) / TM);
                if (acc) k_gemm<true><<<grid, 256, 0, stream_>>>(X, m.w.as<__half>(), y, ldy, B, m.rows, n);
                else k_gemm<false><<<grid, 256, 0, stream_>>>(X, m.w.as<__half>(), y, ldy, B, m.rows, n);
                CK_LAUNCH();
                continue;
            }
            if (!split) {  // hi/lo bölmesi bir kez, tüm parçalar için
                k_split<<<B, 256, 0, stream_>>>(X, xh_.as<__half>(), xl_.as<__half>(), xs_.as<float>(),
                                                need_xsum ? xsum_.as<float>() : nullptr, n);
                CK_LAUNCH();
                split = true;
            }
            switch (m.type) {
                case W_F16:
                    if (!f16_mma_ && n % GK == 0) {
                        const dim3 grid((m.rows + GN - 1) / GN, (B + GM - 1) / GM);
                        const __half *xh = xh_.as<__half>(), *xl = xl_.as<__half>(), *w = m.w.as<__half>();
                        if (acc) k_gemm_tc<true><<<grid, 256, 0, stream_>>>(xh, xl, xs_.as<float>(), w, y, ldy, B,
                                                                            m.rows, n);
                        else k_gemm_tc<false><<<grid, 256, 0, stream_>>>(xh, xl, xs_.as<float>(), w, y, ldy, B,
                                                                         m.rows, n);
                    } else {
                        gemm_q_t<W_F16>(m, y, ldy, B, acc);
                    }
                    break;
                case W_Q8_0: gemm_q_t<W_Q8_0>(m, y, ldy, B, acc); break;
                case W_Q4_K: gemm_q_t<W_Q4_K>(m, y, ldy, B, acc); break;
                case W_Q6_K: gemm_q_t<W_Q6_K>(m, y, ldy, B, acc); break;
            }
            CK_LAUNCH();
        }
    }

    // Tüm katmanlar (+ istenirse son tokenin logit'leri). x_ gömmelerle dolu gelir; konum aygıt belleğinden okunur.
    void body(int B, bool want_logits) {
        const int E = c_.n_embd, F = c_.n_ff, D = c_.head_dim, H = c_.n_head, HKV = c_.n_head_kv;
        const int QD = D * H, KD = D * HKV, QKV = QD + 2 * KD;
        const float scale = 1.0f / std::sqrt(float(D));
        float *x = x_.as<float>(), *h = h_.as<float>(), *qkv = qkv_.as<float>(), *att = att_.as<float>();
        float *gu = gu_.as<float>(), *act = act_.as<float>();
        const float2* cs = rope_.as<float2>();
        const int* pos = pos_.as<int>();

        for (int l = 0; l < c_.n_layer; ++l) {
            const Layer& L = layers_[size_t(l)];
            float* kc = kcache_.as<float>() + size_t(l) * n_ctx_ * KD;
            float* vc = vcache_.as<float>() + size_t(l) * n_ctx_ * KD;

            k_rmsnorm<<<B, 256, 0, stream_>>>(x, L.attn_norm.as<float>(), h, E, c_.eps);
            CK_LAUNCH();
            linear(h, L.qkv, qkv, B, QKV, E, false, limbs(1));
            k_headnorm_rope<<<dim3(B, H), D, D * sizeof(float), stream_>>>(qkv, QKV, L.q_norm.as<float>(), cs, pos,
                                                                             c_.rope_dim, c_.eps);
            CK_LAUNCH();
            k_headnorm_rope<<<dim3(B, HKV), D, D * sizeof(float), stream_>>>(qkv + QD, QKV, L.k_norm.as<float>(), cs,
                                                                               pos, c_.rope_dim, c_.eps);
            CK_LAUNCH();
            k_store_kv<<<B, 256, 0, stream_>>>(qkv, QKV, QD, KD, kc, vc, pos);
            CK_LAUNCH();
            if (naive_attn_) {
                k_attn<<<dim3(H, B), kHeadDim, 0, stream_>>>(qkv, QKV, kc, vc, att, pos, H, HKV, scale);
            } else if (B == 1) {
                k_attn_split<<<dim3(H, kAttnSplit), kHeadDim, 0, stream_>>>(
                    qkv, kc, vc, part_acc_.as<float>(), part_ml_.as<float2>(), pos, H, HKV, scale);
                CK_LAUNCH();
                k_attn_combine<<<H, kHeadDim, 0, stream_>>>(part_acc_.as<float>(), part_ml_.as<float2>(), att);
            } else if (tile32_attn_) {
                k_attn_prefill<<<dim3(H, (B + AQ - 1) / AQ), 256, kAttnPrefillSmem, stream_>>>(qkv, QKV, kc, vc, att,
                                                                                               pos, B, H, HKV, scale);
            } else {
                k_attn_prefill64<<<dim3(H, (B + PQ - 1) / PQ), 256, kAttnPrefill64Smem, stream_>>>(
                    qkv, QKV, kc, vc, att, pos, B, H, HKV, scale);
            }
            CK_LAUNCH();
            linear(att, L.wo, x, B, E, QD, true, limbs(2));

            k_rmsnorm<<<B, 256, 0, stream_>>>(x, L.ffn_norm.as<float>(), h, E, c_.eps);
            CK_LAUNCH();
            linear(h, L.gu, gu, B, 2 * F, E, false, limbs(4));
            const size_t nf = size_t(B) * F;
            k_silu_mul<<<unsigned((nf + 255) / 256), 256, 0, stream_>>>(gu, act, F, nf);
            CK_LAUNCH();
            linear(act, L.down, x, B, E, F, true, limbs(8));
        }
        if (want_logits) {
            k_rmsnorm<<<1, 256, 0, stream_>>>(x + size_t(B - 1) * E, output_norm_.as<float>(), h, E, c_.eps);
            CK_LAUNCH();
            linear(h, lm_head_, logits_.as<float>(), 1, c_.n_vocab, E, false);
        }
    }

    void chunk(const int32_t* tokens, int B, int pos, float* logits) {
        // Gömme satırları CPU'da çözülür (tablo nicemli olabilir; satır başına yalnızca E değer)
        const size_t E = size_t(c_.n_embd);
        for (int i = 0; i < B; ++i) get_row(*emb_, uint64_t(tokens[i]), emb_host_ + size_t(i) * E);
        *pos_host_ = pos;
        CK(cudaMemcpyAsync(x_.p, emb_host_, size_t(B) * E * sizeof(float), cudaMemcpyHostToDevice, stream_));
        CK(cudaMemcpyAsync(pos_.p, pos_host_, sizeof(int), cudaMemcpyHostToDevice, stream_));

        if (B == 1 && use_graph_) {
            // Çözme adımı: ilk seferde yakala, sonra tek çağrıyla oynat (yüzlerce çekirdek başlatma yerine 1)
            if (!graph_exec_) {
                cudaGraph_t graph = nullptr;
                CK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
                body(1, true);
                CK(cudaStreamEndCapture(stream_, &graph));
                const cudaError_t e = cudaGraphInstantiate(&graph_exec_, graph, 0);
                cudaGraphDestroy(graph);
                CK(e);
            }
            CK(cudaGraphLaunch(graph_exec_, stream_));
        } else {
            body(B, logits != nullptr);
        }
        if (logits)
            CK(cudaMemcpyAsync(logits, logits_.p, size_t(c_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost,
                               stream_));
        CK(cudaStreamSynchronize(stream_));
    }

    Config c_;
    int n_ctx_, max_batch_;
    // 3 parçalı (22 bit) x kullanan doğrusal katmanlar, bit maskesi: 1 qkv, 2 wo, 4 gate/up, 8 down. wo ve gate/up
    // girdilerinde 3 parça 4 parçayla aynı doğrulukta; qkv'de hata ~1.7 kat, ffn_down girdisindeki aykırı değerlerde
    // (en büyük/ortanca ~36 bin) ~4 kat artıyor. EL_NL3 ile değiştirilebilir (0: her yerde 4 parça).
    int nl3_ = 2 | 4;
    // DENEYSEL, varsayılan kapalı: 2 parçalı (14 bit) x. fp32 doğruluk sözleşmesinin dışında; EL_NL2 bit maskesiyle.
    int nl2_ = 0;
    int limbs(int bit) const { return nl2_ & bit ? 2 : nl3_ & bit ? 3 : 4; }
    bool use_tc_ = true, f16_mma_ = false, q_hilo_ = false, use_graph_ = true, naive_attn_ = false, tile32_attn_ = false;
    int sm_count_ = 1;
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    const Tensor* emb_ = nullptr;   // gömme tablosu (bellek eşlemeli dosyada)
    float* emb_host_ = nullptr;     // sabitlenmiş (pinned) gömme satırları [B][E]
    int* pos_host_ = nullptr;       // sabitlenmiş konum
    Lin lm_head_;
    DevBuf output_norm_, rope_;
    std::vector<Layer> layers_;
    DevBuf kcache_, vcache_;
    DevBuf x_, h_, qkv_, att_, gu_, act_, xh_, xl_, xs_, xsum_, x8_, xs8_, xsum32_, logits_, pos_, part_acc_, part_ml_;
};

}  // namespace

bool cuda_available(std::string* why) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n == 0) {
        if (why) *why = e != cudaSuccess ? cudaGetErrorString(e) : "CUDA aygiti yok";
        return false;
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess || p.major < 7 || (p.major == 7 && p.minor < 5)) {
        if (why) *why = "GPU hesaplama yetenegi 7.5 ve uzeri olmali (mma.sync m16n8k8)";
        return false;
    }
    return true;
}

std::unique_ptr<Backend> make_cuda_backend(const Gguf& g, int n_ctx) {
    std::string why;
    if (!cuda_available(&why)) throw std::runtime_error("CUDA kullanilamiyor: " + why);
    return std::make_unique<CudaModel>(g, n_ctx, 512);
}

}  // namespace el
