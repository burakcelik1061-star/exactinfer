// ggml nicemleme blokları (GGUF'taki bayt düzeniyle birebir) ve satır çözücüler.
//
//   Q8_0  32 değer / 34 bayt:  y = d · q                         q ∈ [-128, 127]
//   Q4_K 256 değer / 144 bayt: y = (d·sc) · q − (dmin·m)          q ∈ [0, 15], 32'lik alt blok başına 6 bit sc, m
//   Q6_K 256 değer / 210 bayt: y = (d·sc) · (q − 32)              q ∈ [0, 63], 16'lık alt blok başına int8 sc
// Çözücüler ggml'in dequantize_row_* işlevleriyle aynı işlem sırasını izler (aynı fp32 yuvarlama).
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define EL_HD __host__ __device__ __forceinline__
#else
#define EL_HD inline
#endif

namespace el {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;

#pragma pack(push, 1)
struct BlockQ8_0 {
    uint16_t d;  // fp16
    int8_t qs[QK8_0];
};
struct BlockQ4_K {
    uint16_t d, dmin;     // fp16
    uint8_t scales[12];   // 8 alt blok için 6 bitlik ölçek ve min, sıkıştırılmış
    uint8_t qs[QK_K / 2]; // 64'lük her parçada: alt 4 bit ilk 32 değer, üst 4 bit sonraki 32
};
struct BlockQ6_K {
    uint8_t ql[QK_K / 2];   // alt 4 bit
    uint8_t qh[QK_K / 4];   // üst 2 bit
    int8_t scales[QK_K / 16];
    uint16_t d;             // fp16
};
#pragma pack(pop)
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 blok boyutu");
static_assert(sizeof(BlockQ4_K) == 144, "Q4_K blok boyutu");
static_assert(sizeof(BlockQ6_K) == 210, "Q6_K blok boyutu");

// Q4_K alt blok j (0..7) için 6 bitlik ölçek ve min
EL_HD void q4k_scale_min(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// n değerlik satırı çözer (n blok boyunun katı)
void dequant_q8_0(const void* src, float* y, int64_t n);
void dequant_q4_K(const void* src, float* y, int64_t n);
void dequant_q6_K(const void* src, float* y, int64_t n);

}  // namespace el
