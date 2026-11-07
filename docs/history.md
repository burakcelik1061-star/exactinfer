# Development history

The project was built in phases. The order was set on purpose: **correctness first, then speed, then scale.** Each
phase had to pass the previous phases' checks before it counted as done.

| Phase | Goal | Outcome |
|---|---|---|
| 1. Correctness (CPU, fp32) | GGUF reader, tokenizer, forward pass | Tokenizer 30 042/30 042 vs HF; forward pass 11/11 vs numpy fp32 (3.6e-5); 10/11 identical to Ollama end to end, 1 fp16 near-tie |
| 2. CUDA | kernels for the RTX 2060 without changing the accuracy contract | hi/lo fp16 split on tensor cores, sliced fp32 accumulation (2.0e-5); GEMV + CUDA Graph + flash-decoding: decode ~30% faster than Ollama on 0.6B |
| 3. Scaling and quantization | Qwen3-1.7B Q8_0, Qwen3-4B Q4_K_M in 6 GB | dequantizers bit-identical to `gguf`; int8 limb GEMM with exact int32 products; 11/11 vs fp32 on all models |
| 4. The guard | Rust tool-call guard | allow / confirm / deny, tickets, path jail, taint escalation, actions.log, emergency stop |
| 5. Constrained decoding | tool calls that are always schema-valid | byte-level matcher with a no-dead-end invariant; altered calls escalate to confirmation |
| 6. Integration | the engine as a real assistant's brain | stage 1 engine brain; stage 2 the guard as decision-maker; long-prompt attention; DRY; re-ask turn 98 s → 8 s |
| GEMM study (2026-11-07) | prompt speed without losing exactness | weight prefetch +3–7%, bit-identical; 2-limb mode measured and left off |

Details are in [accuracy.md](accuracy.md), [performance.md](performance.md), [bekci.md](bekci.md),
[tool-calls.md](tool-calls.md) and [integration.md](integration.md).

## Possible next steps

- Prefetch for the activation (A) tile, and fusing `k_split8` into the GEMM.
- An fp16 or int8 KV cache for longer context. This is only acceptable with a measured accuracy cost, as an explicit
  opt-in.
- More architectures beyond Qwen3 dense.
- A Linux build.
