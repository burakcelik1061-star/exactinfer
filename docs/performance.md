# Performance

All numbers below were measured on a laptop: RTX 2060 6 GB (sm_75, 336 GB/s), Ryzen 5 5600X, Windows 11. Same GGUF
file and same GPU for engine and Ollama. Medians; ranges are GPU clock variation between runs (see
[methodology](#measurement-methodology)).

```powershell
python tests\bench.py --model qwen3:4b-instruct      # engine and Ollama, pp512 / tg128
.\build\el.exe bench $m --pp 512 --tg 128             # engine only
```

## Summary

| Tokens/s | pp512 engine | pp512 Ollama | tg128 engine | tg128 Ollama |
|---|---|---|---|---|
| Qwen3-0.6B F16 | 5 280 – 5 800 | 6 900 – 8 200 | **164 – 186** | 122 – 138 |
| Qwen3-1.7B Q8_0 | 2 850 | 4 640 – 4 730 | **111 – 123** | 113 – 119 |
| Qwen3-4B Q4_K_M | 1 100 – 1 140 | 1 770 – 2 170 | **76 – 81** | 83 – 86 |

*pp = prompt processing (prefill), tg = token generation (decode). The 4B prompt number includes the later attention
and weight-prefetch work; the 1.7B one predates the prefetch, which measured +4.5% on it.*

Longer runs on 0.6B F16: pp2048 3 200 – 3 500 (Ollama 9 000), tg512 164 (Ollama 127). CPU backend (5600X, AVX2):
pp512 30, tg128 25.

## Token generation: bandwidth-bound, at or ahead of Ollama

Decoding one token reads every weight once, so speed is set by how close each GEMV gets to memory bandwidth.

- `k_gemv<type>` uses one row per warp, keeps x in shared memory and runs a persistent grid sized to the SM count. It
  decodes F16 / Q8_0 / Q4_K / Q6_K weights while reading, so the bytes read per kernel are as few as possible.
- Q/K/V and gate/up matrices are fused.
- The whole decode step (≈340 kernels) is captured once as a **CUDA Graph**. Position and token are read from device
  memory, so the step replays with one call per token.
- Decode attention is split-K (flash-decoding): keys go in 16 blocks, then a combine kernel.

The quantized GEMVs reach 70–80% of the card's bandwidth (Q4_K ~235 GB/s, Q6_K ~265 GB/s).

- **0.6B F16:** ~30% faster than Ollama.
- **1.7B:** on par.
- **4B:** ~5% behind. Ollama uses integer (dp4a) products there; the engine computes in fp32.

## Prompt processing: slower by design

On quantized models the engine is 1.65× (1.7B) to 2.1× (4B) slower than Ollama here.

On quantized models Ollama rounds each activation row to one 8-bit value and does one int8 product. The engine splits
it into 3–4 exact int8 limbs to keep fp32 accuracy ([accuracy.md](accuracy.md)), so it does the same product with
**3–4× more tensor-core work**. On F16 models the hi/lo split doubles the work the same way.

In wall-clock terms this costs little for an assistant: a 1 000-token prompt at 4B takes ~1 s, and a chat reuses the
KV prefix ([serve-protocol.md](serve-protocol.md)).

### Where the time goes
In a 2 048-token prompt at 4B, ~77% of the time is `k_gemm_i8` and 16% is prompt attention (Nsight Systems).
Ablation, with parts of the kernel disabled and kernel time measured:

| Disabled | Time saved |
|---|---|
| tensor-core multiply | 3% |
| global weight loads | 24% |

The kernel waits on weight loads; the tensor cores are not the bottleneck. GPU performance counters need
administrator rights, which the project does not use, so tuning relies on timings and ablation.

### What was done, and what was tried and dropped

| Change | Effect | Kept |
|---|---|---|
| int8 limb GEMM instead of the hi/lo fp16 path on quantized models | 4B pp512 615 → ~980 | yes (`EL_GEMM=hilo` restores the old path) |
| `k_attn_prefill64`: 64 queries per block, scores and output in registers, float4 shared-memory reads, next K/V tile prefetched | 6 800-token prompt: attention 7.5 s → 4.1 s (~3.3 TFLOPS, half the fp32 peak) | yes |
| Weight prefetch: raw bytes for the next K step (Q4_K 12, Q6_K 18 registers) loaded while the current step computes | 4B pp512 **+5–7%**, pp2048 **+3–6%**; 1.7B +4.5% / +3% | yes |
| `ldmatrix` (×4) + transposed scale layout | a third of the shared-memory instructions, no speed on its own | yes: lower register pressure lets the prefetch fit without spills |
| Two blocks per SM (64×64 tile, K step 64) | 20% slower: the A tile (3–4 limbs) is read twice per output | no |
| Unrolling the sub-slice loop (2, 4) | no gain, more spills | no |
| No prefetch for Q6_K | 2% worse | no |
| 2 activation limbs | pp512 +26%, but breaks accuracy | off (`EL_NL2`) |

Every kept change is **bit-identical** to the kernel before it. The check ran on 0.6B F16, 1.7B Q8_0 and 4B Q4_K at
13 / 1 293 / 3 848-token prompts, and the fp32 reference gave 11/11 on all three models.

Prompt attention, 4B:

| prompt tok/s | pp512 | pp2048 | pp6800 |
|---|---|---|---|
| old attention (`EL_ATTN=tile32`) | 905 | 775 | 499 (13.6 s) |
| `k_attn_prefill64` | 1 019 | 851 | 646 (10.5 s) |

The math is still fp32; only the summation order changed. `tests\attn_long.py` on a 6 023-token prompt: three
attention kernels give the same tokens (max \|Δlogprob\| 1.6e-5 vs the tiled kernel, 9.6e-5 vs the single-block one).

## Measurement methodology

On this laptop the GPU clock drifts with temperature: the same binary gave 1 130 and 1 010 tok/s ten minutes apart.
Sequential "before, then after" runs are therefore misleading; an early "+11%" for `ldmatrix` turned out to be drift,
and the real ratio was 0.99. Speed comparisons alternate the two binaries run by run (interleaved A/B) and compare
medians of the pairs.

Speed claims in this repository come with the command that produced them. Accuracy is re-checked after every kernel
change, so a speed number never comes from a kernel that fails the fp32 reference.

## Memory

| | KV cache per token | Context on a 6 GB card |
|---|---|---|
| 4B | 288 KB (fp32) | ~8k tokens |
| 1.7B / 0.6B | 224 KB (fp32) | more |

GPU memory is computed before loading; if the model and context do not fit, loading stops with a clear error. Loading
takes 3–6 s (CUDA context + copying weights from the mapped file to VRAM); an assistant loads once and stays resident.
An fp16/int8 KV cache would double or quadruple the context, but it changes the accuracy contract and has not been
done.
