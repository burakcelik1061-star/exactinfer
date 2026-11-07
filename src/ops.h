// CPU hesap çekirdekleri (fp32 biriktirme). Ağırlıklar dosyadaki türünde kalır, okunurken fp32'ye çevrilir.
#pragma once

#include <cstdint>
#include <vector>

#include "gguf.h"

namespace el {

float f16_to_f32(uint16_t h);
float bf16_to_f32(uint16_t h);

bool supported_type(GgmlType t);

// Tensörün tamamını fp32 vektöre çevirir (küçük tensörler: norm ağırlıkları)
std::vector<float> to_f32(const Tensor& t);

// W'nin `row` numaralı satırını (ne[0] öğe) fp32 olarak yazar (gömme tablosu araması)
void get_row(const Tensor& w, uint64_t row, float* out);

// y[r] = sum_j W[r, j] * x[j],  r < W.rows(), j < W.ne[0]
void matvec(const Tensor& w, const float* x, float* y);

// out = x / rms(x) * w
void rmsnorm(float* out, const float* x, const float* w, int n, float eps);

void set_threads(int n);
int threads();

}  // namespace el
