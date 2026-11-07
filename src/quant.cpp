#include "quant.h"

#include "ops.h"

namespace el {

void dequant_q8_0(const void* src, float* y, int64_t n) {
    const auto* b = static_cast<const BlockQ8_0*>(src);
    for (int64_t i = 0; i < n / QK8_0; ++i) {
        const float d = f16_to_f32(b[i].d);
        for (int j = 0; j < QK8_0; ++j) y[i * QK8_0 + j] = b[i].qs[j] * d;
    }
}

void dequant_q4_K(const void* src, float* y, int64_t n) {
    const auto* b = static_cast<const BlockQ4_K*>(src);
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = f16_to_f32(b[i].d), dmin = f16_to_f32(b[i].dmin);
        const uint8_t* q = b[i].qs;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            int sc, m;
            q4k_scale_min(is + 0, b[i].scales, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            q4k_scale_min(is + 1, b[i].scales, sc, m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void dequant_q6_K(const void* src, float* y, int64_t n) {
    const auto* b = static_cast<const BlockQ6_K*>(src);
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = f16_to_f32(b[i].d);
        const uint8_t* ql = b[i].ql;
        const uint8_t* qh = b[i].qh;
        const int8_t* sc = b[i].scales;
        for (int h = 0; h < QK_K; h += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = int8_t((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = int8_t((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = int8_t((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = int8_t((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

}  // namespace el
