# Accuracy

exactinfer's design constraint is that **quantization changes how weights are stored, not what is computed**. Weights
stay in their GGUF format (F16, Q8_0, Q4_K, Q6_K) in VRAM. Activations, the KV cache and every accumulation are fp32,
and every kernel is checked against an independent fp32 implementation that shares no code with the engine.

This page explains how that contract is kept on hardware that has no fp32 matrix units, and how it is measured.

## The checks

| Check | What it compares | Result |
|---|---|---|
| GGUF reader | metadata and tensor table vs Ollama's report | 751.63M parameters, 311 tensors, 28 metadata keys: identical |
| Tokenizer | byte-level BPE + qwen2 pre-tokenizer vs HuggingFace `tokenizers` (Rust) | **30 042 / 30 042 strings, ~1M tokens identical** (Turkish, code, emoji, CJK, Unicode whitespace, random fuzz) |
| Dequantizers | random rows of every tensor vs the `gguf` Python package | 1.7B Q8_0: 7.4M values **bit-identical**; 4B Q4_K_M: 15.4M values (Q4_K + Q6_K) **bit-identical** |
| Forward pass | teacher-forced logprobs vs an independent numpy fp32 implementation, 11 prompts (one of 1 109 tokens, crossing the 512-token chunk boundary) | **11/11 on every model, all tokens identical** |
| End to end | greedy generation vs Ollama (llama.cpp) | 0 failures; only near-ties, each resolved by the fp32 reference in the engine's favour |

Max \|Δlogprob\| against the fp32 reference:

| | CPU backend | CUDA backend |
|---|---|---|
| Qwen3-0.6B F16 | 3.6e-5 | 2.0e-5 |
| Qwen3-1.7B Q8_0 | — | 2.7e-5 |
| Qwen3-4B Q4_K_M | — | 1.6e-5 |

Every later optimisation (prompt attention, weight prefetch, `ldmatrix`) re-ran this check: 11/11 on all three
models, ≤ 2.3e-5.

```powershell
python tests\parity_tokenizer.py
python tests\parity_dequant.py --model qwen3:4b-instruct
python tests\parity_reference.py --backend cuda --model qwen3:4b-instruct
python tests\parity_model.py --backend cuda --model qwen3:4b-instruct
```

## Why Ollama's numbers differ

Ollama (llama.cpp) runs F16 models with fp16 math and an fp16 KV cache, so logprobs differ from fp32 by ~1e-2 and a
long context sees near-ties of up to ~0.1. On quantized models it also **rounds each activation row to 8 bits
(`q8_1`) before the matrix multiply**, so logprobs differ by up to ~0.2 and a few near-ties go the other way.

`parity_model.py` counts a divergence as a near-tie only when both sides' choices are within `--tie` logprob of each
other. At every such divergence, `parity_reference.py` (fp32) agrees with the engine.

| Ollama comparison | 0.6B F16 | 1.7B Q8_0 | 4B Q4_K_M |
|---|---|---|---|
| failures | 0 | 0 | 0 |
| near-ties | 1 | 3 | 2 |

This does not make Ollama wrong in practice: on the assistant's 62-scenario test it picks the right tool as often
(see [integration.md](integration.md)). What it means is that exactinfer's output is a *reproducible function of the
weights*: the same as a straightforward fp32 implementation, not one more approximation of it.

## F16 weights on fp16 tensor cores: the hi/lo split

Turing tensor cores only take fp16 inputs. Each activation row is scaled by a per-row power of two (lossless) and split
into two fp16 parts, `x·s ≈ hi + lo`, which together carry ~22 bits of mantissa. The weights are already F16, so

```
X·Wᵀ = hi·Wᵀ + lo·Wᵀ        (two tensor-core passes, fp32 accumulation)
```

Turing's tensor-core internal accumulation is not as precise as IEEE fp32: measured over long chains, \|Δlogprob\|
was 2.9e-4. So only **64-wide K slices** go through the tensor core and the slices are combined with ordinary fp32
adds. The result is 2.0e-5, better than a classic fp32 tiled GEMM on the same data (2.6e-5).

## Quantized weights

### Decode (one token at a time)
Each weight is decoded in fp32 using ggml's formula (sub-block scale `d·sc`, min `dmin·m`). Scales and mins are
pulled out of the inner sum:

```
Σ x·(d₁·q − m₁) = d₁·Σ x·q − m₁·Σ x
```

4–6-bit integers become floats with a magic-number trick instead of I2F: `0x4B0000vv` is the float 2²³ + vv, so the
conversion is one byte permute and one subtract, both at full rate.

### Prompt: exact int8 tensor-core products with limbs
The integer part of a quantized weight is already int8 (Q4_K 0..15, Q6_K −32..31, Q8_0 −128..127). The activation
row is converted to fixed point with a power-of-two scale 2ᵏ, chosen so the row maximum lands in
2^(8·NL−3)..2^(8·NL−2). It is then split into **NL signed 8-bit limbs**:

```
X = Σ_l a_l · 2^(8l)
```

Raw PTX `mma.sync.m8n8k16.s8.s8.s32` computes each `Σ a_l·q` **exactly** in int32. The integer sub-block scales of
Q4_K/Q6_K are multiplied and accumulated in int32 too. The result moves to fp32 only once per 64 (Q4_K) or 256 (Q6_K)
values, where `d` and Q4_K's min term are applied.

- **Overflow is ruled out by construction.** The bounds follow from the scale ranges. For example, in Q4_K two limbs
  share one int32 and `2·63·61440·257 < 2³¹`.
- **The only rounding is the fixed-point conversion of the activation:** 30 bits with NL = 4, 22 bits with NL = 3.
- **Limb count per matrix:** `qkv` and `ffn_down` inputs use 4 limbs, because they carry outliers. `wo` and gate/up
  inputs use 3, with measured error equal to 4 limbs. `EL_NL3=<mask>` overrides this: 1 qkv, 2 wo, 4 gate/up,
  8 down; default 6.

### Aligned re-layout
Q8_0 (34-byte) and Q6_K (210-byte) blocks are not 4/16-byte aligned. At load time they are rewritten bit for bit into
an aligned layout (Q6_K: a 208-byte block plus a separate scale array) so vector loads work. `parity_dequant.py`
checks the result.

## The trade-off, measured: the 2-limb experiment

Fewer limbs is the main reason llama.cpp's quantized prompt path is faster, because it does the product once with
8-bit activations. A 2-limb (14-bit) mode exists in the code behind `EL_NL2=<mask>` and is **off by default**:

| Setting | Prompt speed vs default | fp32 reference | max \|Δlogprob\| |
|---|---|---|---|
| default (3–4 limbs) | baseline | 11/11 | 2.3e-5 |
| 2 limbs on `wo` and gate/up | not reported | 8/11 | 5.5e-3 |
| 2 limbs everywhere | pp512 +26%, pp2048 +14–21% | 2/11 | 5.8e-2 (~2 500× worse) |

It breaks the accuracy contract, so it stays off. The experiment shows where the speed gap with llama.cpp comes from
and what it would take to close it.
