# exactinfer

A from-scratch LLM inference engine that runs open-weight **Qwen3** models straight from GGUF files, with no ML
libraries (no cuBLAS, no cuDNN, no llama.cpp). The core is C++ and CUDA with every kernel hand-written; a separate
**Rust process, the "bekci" (Turkish for *guard*)**, validates and gates every tool call the model produces.

The engine's design constraint is **exactness**: quantized weights are stored as-is in VRAM, but activations, the KV
cache and all accumulations stay in fp32, and every kernel is checked against an independent fp32 reference. It is
the local "brain" of a private voice desktop assistant, which is how the tool-calling and guard features were
developed and measured.

> No model is trained here. The engine runs the open weights published by the Qwen team (Apache-2.0). The tokenizer,
> GGUF parser, transformer forward pass, CUDA kernels, sampling, constrained decoding and the guard are written in
> this repository. Source comments and the notes in `bekci/` are in Turkish; this README is the English reference.

## Highlights

| | |
|---|---|
| Correctness | 11/11 prompts, 455+ tokens identical to an independent numpy fp32 implementation on 0.6B F16, 1.7B Q8_0 and 4B Q4_K_M; max \|Δlogprob\| ≤ 2.7e-5 |
| Tokenizer | 30 042 / 30 042 test strings (~1M tokens) identical to HuggingFace `tokenizers`, including Turkish, code, emoji, CJK |
| Quantized GEMM | int8 tensor-core path (`mma.sync m8n8k16`) with activations split into 3–4 exact int8 limbs, so quantized prompt processing keeps fp32-level accuracy |
| Decode speed | 4B Q4_K_M: 76–81 tok/s on an RTX 2060 6 GB (Ollama 83–86); 0.6B F16: 164–186 vs 122–138 |
| Tool-call safety | constrained decoding guarantees every tool call is schema-valid; a Rust guard decides allow / confirm / deny per call |
| Hardware | developed on Windows 11, RTX 2060 6 GB (sm_75), Ryzen 5 5600X |

## Status

### Phase 1 — correctness (CPU, fp32) ✅

| Component | Checked against | Result |
|---|---|---|
| GGUF reader | Ollama's report | 751.63M parameters, 311 tensors, 28 metadata keys — identical |
| Tokenizer (byte-level BPE, qwen2 pre-tokenizer, special tokens) | HuggingFace `tokenizers` (Rust) | **30 042 / 30 042 strings, ~1M tokens identical** (Turkish, code, emoji, CJK, Unicode whitespace, random fuzz) |
| Forward pass (Qwen3-0.6B) | independent numpy fp32 implementation (teacher forcing) | **11/11 prompts, all 455 tokens identical; max \|Δlogprob\| 3.6e-5** |
| End-to-end generation | Ollama (llama.cpp, GPU fp16) | 10/11 prompts identical, same EOS points; 1 prompt had an fp16-induced near-tie (0.084) that the fp32 reference resolves in the engine's favour |

### Phase 2 — CUDA (RTX 2060 6 GB, sm_75) ✅

The accuracy contract is unchanged: weights F16, **all activations, KV cache and accumulations fp32**.

| Check | Result |
|---|---|
| Independent numpy fp32 implementation | **11/11 prompts (including a 1109-token prompt), all 455 tokens identical; max \|Δlogprob\| 2.0e-5** |
| Ollama | same table as the CPU backend (10/11 exact, 1 fp16 near-tie) |

Speed (tokens/s, median, same model file, same GPU; `python tests\bench.py`):

| | pp512 | tg128 | pp2048 | tg512 |
|---|---|---|---|---|
| **engine (CUDA, fp32 math)** | 4 400 – 5 400 | **158 – 173** | 3 200 – 3 500 | **164** |
| Ollama (CUDA, fp16 math) | 7 600 – 8 300 | 126 – 128 | 9 000 | 127 |
| engine (CPU, 5600X) | 30 | 25 | — | — |

*pp = prompt processing, tg = token generation. Ranges are GPU clock variation between runs.*

- **Token generation is ~30% faster than Ollama.** This phase is memory-bandwidth bound: a GEMV that minimises
  bytes read per kernel, fused Q/K/V and gate/up matrices, a CUDA Graph, and split-K (flash-decoding) attention.
- **Prompt processing is 1.7–2.8× slower than Ollama.** Ollama rounds activations to fp16 and runs tensor cores at
  full rate; the engine keeps fp32 accuracy by running each product as two tensor-core passes (below). For an
  assistant, a 1–2k-token prompt still takes 0.3–0.6 s.

#### fp32 accuracy on tensor cores
Tensor cores only accept fp16 inputs. Each activation row is split with a per-row power-of-two scale (lossless)
into two fp16 parts: `x·s ≈ hi + lo` (~22 bits). Because the weights are already F16,
`X·Wᵀ = hi·Wᵀ + lo·Wᵀ` is computed in two passes with fp32 accumulation. Turing's tensor-core internal accumulation
is not as precise as IEEE fp32 (measured: \|Δlogprob\| 2.9e-4 over long chains), so only 64-wide slices go through
the tensor core and slices are combined with ordinary fp32 adds → 2.0e-5, better than a classic fp32 tiled GEMM
(2.6e-5).

### Phase 3 — scaling and quantization (Qwen3-1.7B Q8_0, Qwen3-4B Q4_K_M) ✅

Weights stay in their quantized format in VRAM and are decoded during the multiply; the accuracy contract is
unchanged (activations, KV cache and accumulation in fp32). The 4B model (2.4 GB) fits a 6 GB card comfortably.

| Check | 0.6B F16 | 1.7B Q8_0 | 4B Q4_K_M |
|---|---|---|---|
| Dequantizers vs the `gguf` package (random rows from every tensor) | — | 7.4M values **bit-identical** | 15.4M values (Q4_K + Q6_K) **bit-identical** |
| Independent numpy fp32 implementation (11 prompts, incl. 1109 tokens) | 11/11, \|Δlp\| ≤ 2.0e-5 | 11/11, \|Δlp\| ≤ 2.7e-5 | 11/11, \|Δlp\| ≤ 1.6e-5 |
| Ollama (end to end) | 0 failures, 1 near-tie | 0 failures, 3 near-ties | 0 failures, 2 near-ties |

Ollama (llama.cpp) also rounds activations to 8 bits before the matrix multiply on quantized models, so logprobs
differ by up to ~0.2 and a few near-ties go the other way. At every disagreement the independent fp32
implementation confirms the engine's choice.

Speed (tokens/s, median; `python tests\bench.py --model ...`):

| | pp512 | tg128 |
|---|---|---|
| **engine, 4B Q4_K_M** | 900 – 980 (after the later attention and weight-prefetch work: 1 100 – 1 140) | **76 – 81** |
| Ollama, 4B Q4_K_M | 1 770 – 2 170 | 83 – 86 |
| **engine, 1.7B Q8_0** | 2 850 | **111 – 123** |
| Ollama, 1.7B Q8_0 | 4 640 – 4 730 | 113 – 119 |
| **engine, 0.6B F16** | 5 280 – 5 800 | **164 – 186** |
| Ollama, 0.6B F16 | 6 900 – 8 200 | 122 – 138 |

- **Token generation** is bandwidth bound: the quantized GEMVs use 70–80% of the card's 336 GB/s (Q4_K ~235,
  Q6_K ~265 GB/s). On par with Ollama at 1.7B, ~5% behind at 4B; Ollama uses integer (dp4a) products where the
  engine computes in fp32.
- **Prompt processing:** Ollama is ~2.1× faster at 4B and ~1.65× at 1.7B. Quantized models use int8 tensor cores
  (below) and split activations into 3–4 int8 limbs to keep fp32 accuracy; Ollama rounds activations to a single
  8-bit value, i.e. it does the same product with 3–4× less work. A 1 000-token prompt at 4B takes ~1 s. The old
  hi/lo fp16 path (`EL_GEMM=hilo`, ~615 tok/s at 4B) is kept for comparison.

#### fp32 accuracy with quantized weights
- **Decoding (single token):** every weight is decoded in fp32 using ggml's formula (sub-block scale `d·sc`, min
  `dmin·m`); scales and mins are pulled outside the sum: `Σ x·(d₁·q − m₁) = d₁·Σ x·q − m₁·Σ x`. 4–6-bit integers
  are converted to float with a magic number instead of I2F (`0x4B0000vv` = 2²³ + vv: one byte permute and one
  subtract, both full rate).
- **Prompt (int8 tensor cores):** the integer part of the quantized weight (Q4_K 0..15, Q6_K −32..31, Q8_0
  −128..127) is already int8. The activation row is converted to fixed point with a power-of-two scale 2ᵏ (the row
  maximum lands in 2^(8·NL−3)..2^(8·NL−2)) and split into NL signed 8-bit limbs: `X = Σ a_l·2^(8l)`. Raw PTX
  `mma.sync.m8n8k16.s8.s8.s32` computes `Σ a_l·q` **exactly** in int32 per limb. The integer sub-block scales of
  Q4_K/Q6_K are also multiplied and accumulated in int32; the result moves to fp32 only once per 64 (Q4_K) or 256
  (Q6_K) values, where `d` and Q4_K's min term are applied. Overflow bounds follow from the scale ranges (in Q4_K
  two limbs share one int32: `2·63·61440·257 < 2³¹`). The only rounding is the fixed-point conversion of the
  activation: 30 bits with NL = 4, 22 bits with NL = 3. `ffn_down` and `qkv` inputs use 4 limbs; `wo` and gate/up
  inputs (no outliers) use 3, with measured error equal to 4 limbs (`EL_NL3` overrides).
- **Aligned re-layout:** Q8_0 (34-byte) and Q6_K (210-byte) blocks are not 4/16-byte aligned; at load time they are
  rewritten bit-for-bit into an aligned layout (Q6_K: 208-byte block plus a separate scale array) so vector loads work.

#### GEMM speed study (2026-11-07)

In a 2048-token prompt at 4B, ~77% of the time is `k_gemm_i8` and 16% is prompt attention. Ablation (parts disabled,
kernel time measured with nsys): removing the tensor-core multiply saves only 3%, removing global loads saves 24%.
The kernel is bound by waiting on weight loads, not by the tensor cores. What was done (all **bit-identical** to the
old kernel on 0.6B F16, 1.7B Q8_0 and 4B Q4_K at 13 / 1 293 / 3 848-token prompts; fp32 reference 11/11 on three models):

- **Weight prefetch:** the raw weight bytes for the next K step (Q4_K 12, Q6_K 18 registers) are loaded into
  registers while the current step computes; decode and the shared-memory store happen at the start of the next
  step. Interleaved A/B measurement: 4B pp512 **+5–7%**, pp2048 **+3–6%**; 1.7B +4.5% / +3%.
- `ldmatrix` (four matrices per instruction) and a transposed scale layout (two adjacent output columns in one
  64-bit read) cut shared-memory instructions to about a third but gave no measurable speed on their own; they
  stay because the lower register pressure lets the prefetch fit without spilling in Q4_K.
- Tried and dropped: two blocks per SM (64×64 tile, K step 64) was 20% slower because the A tile (3–4 limbs) is read
  twice per output; unrolling the sub-slice loop (2, 4) gave no gain and more spills; disabling the prefetch for
  Q6_K was 2% worse.
- **Measurement lesson:** on this laptop the GPU clock drifts with temperature (the same binary gave 1 130 and
  1 010 tok/s ten minutes apart). Speed comparisons must alternate the two binaries; sequential runs are
  misleading (an early "+11%" was an artefact of this).

**A 2-limb (14-bit) mode exists but is off by default (`EL_NL2` bitmask, like `EL_NL3`).** Fewer limbs is the main
reason Ollama's quantized prompt path is faster. Measured: 2 limbs everywhere gives pp512 +26%, pp2048 +14–21% but
the fp32 reference passes only 2/11 prompts with max \|Δlogprob\| 5.8e-2 (default: 2.3e-5, ~2 500× smaller);
2 limbs only on `wo` and gate/up passes 8/11 with 5.5e-3. It breaks the accuracy contract, so it is not used.

### Phase 4 — the bekci: a tool-call guard (Rust) ✅

The model never executes anything directly. Its text goes to the bekci, which extracts tool calls, validates them
against an allow-list policy file and schema, and returns **allow**, **confirm** or **deny** for each call. The
executor only runs calls the guard allowed, using the guard's normalised arguments. It is a separate process
speaking JSON lines over stdin/stdout; dependencies are only serde, serde_json, toml and windows-sys.

| Rule | How it is enforced |
|---|---|
| Only defined tools; no free-form commands | A tool not in the policy does not exist; a `blocked` tool is never shown to the model. A `kind = "shell"` parameter loads only with `shell_mode = true` and that tool always needs confirmation |
| Risky actions always confirm, default "No" | `confirm` tools and every network tool (enforced at load time). The guard can open the dialog itself (default button "No"); in client mode the client collects the answer but cannot grant execution. Arguments are bound to the confirmation ticket: they cannot change after approval; tickets are 128-bit random and single-use. No answer within the timeout counts as "No" |
| Every call, confirmation and result is logged | `actions.log` (JSONL, UTC timestamps, rotates at 5 MB). Secret-looking fields (`api_key`, `parola`, `sk-…`) are masked. If the log cannot be written, the action is denied |
| Emergency stop | Global hotkey (default Ctrl+Alt+End) or the `stop` op: pending confirmations drop, running jobs are reported for cancellation, every call is denied until `resume` |
| Not run as admin; parameters validated | Exits if started elevated. Path jail: absolute drive paths only; UNC/device namespace, `..`, alternate data streams (`:`), device names (CON, NUL…), trailing dots/spaces rejected; junctions and symlinks resolved to the real path (an escape through a junction inside an allowed folder is caught). Exact types (`true` is not a number, `1.0` is not an integer), extra/missing fields rejected, range and enum checks, invisible and bidi characters rejected, https-only URLs |
| Instructions inside content are not commands | After a tool whose output is third-party content (file, web, clipboard), every side-effecting tool is escalated to confirmation until the user gives a new instruction. Duplicate JSON keys (`{"name":"a","name":"b"}`) are rejected at every level; half-open or nested `<tool_call>` rejects the whole message. The guard's own policy and log folders are closed to tools |
| No secrets in code or files | A policy file with a key named like api_key/secret/password/token is rejected |

Verification: 20 unit tests, 8 attack scenarios (path traversal, junction escape, type confusion, ticket replay,
timeout, escalation after untrusted data, escalation of constraint-altered calls, emergency stop, log masking), and
3 process-level protocol tests. End to end (`tests\e2e_bekci.py`): the engine produces real calls from the tool list
the guard supplies; 6/6 scenarios got the expected decision (calculator → allow, file deletion and a web address →
confirm, `C:\Windows\System32\config\SAM` → deny, no shutdown tool shown to the model so no call).

### Phase 5 — constrained decoding ✅

`el generate --tools tools.json` (the tool list is the guard's `tools` output). When the model emits the
`<tool_call>` special token the engine switches to constrained mode: at each step it picks the highest-logit token
**that the matcher accepts**. The matcher is a byte-level state machine (`src/constrain.*`, schema in `src/json.*`)
and accepts only:

```
{"name": <tool in the list>, "arguments": {<the tool's parameters, each at most once, required ones included>}}
```

- Exact types: string / integer (range) / number (range, no exponent) / `true`-`false`; enums and tool names come
  from the list.
- Strings obey the same rules as the guard's validators: no invisible or bidi characters, no `\u` escapes, no raw
  control characters; `maxLength` in code points; `name` (letters, digits, space, `_ - .`, no leading/trailing
  space), `path` (absolute drive path; no `:*?"<>|`, device names, trailing dot/space components), `uri`
  (`https://`, no spaces, no `user@`) and `multiline` (the only kind where `\n` `\t` are allowed).
- **Invariant: every accepted prefix can be completed into a valid call** (no dead ends). That is why half UTF-8
  characters, integer prefixes (after `-`, range bounds as u64 magnitudes), path components (a root that can never
  become valid such as "con.") and the last character of a length limit are checked separately for completability.
- Once a call is complete only `</tool_call>` can be chosen; end-of-sequence inside a call is forbidden. In free
  text, `</tool_call>` and spelling the tags out as plain text are forbidden, so the constraint cannot be bypassed.
- If the constraint rejected the model's most likely token and picked another, the call is reported in
  `altered_calls`, and the guard **escalates it to confirmation** (unless the tool is read-only). Found by
  measurement: for "set the volume to 150 percent" the model would have written `150`, the range is 0..100, so the
  constraint closed the number at `15` — schema-valid but not what was asked. Constraints guarantee validity, not
  intent; an altered call never runs silently.
- The path jail (allowed root folders) depends on the file system, so it is not in the schema; the guard enforces it.

Verification (`tests\constrain_bekci.py`, no model needed): 82 hand-written attack/boundary cases (tool outside the
allow-list, extra/missing/duplicate fields, type confusion, `05` `-0` `1e1`, RLO character and `\u202e`,
`..`/UNC/ADS/`NUL`/`con.txt`/`COM1 .log`, `http://`, `https://user@…`) got the expected result; the guard also
rejected those invalid under the policy. `el constrain-test --fuzz` tries all 256 bytes at every step of a random
walk: 18 000 walks with no dead end; **all** 9 000 completed random calls under the policy schema were accepted by
the guard (6 444 allow, 1 113 confirm; 1 443 denied only because the path was outside the allowed folder).

On a model (`tests\constrain_compare.py`, Qwen3-0.6B fp16, CPU, 17 prompts, greedy):

| | Right tool + schema-valid | Generation speed |
|---|---|---|
| Unconstrained | 14/17 | 15.4 tok/s |
| Constrained | 15/17 (1 call changed by the constraint → escalated) | 15.0 tok/s |

At 0.6B the model already writes valid JSON on these prompts (inner quotes, single-backslash paths, multiline
notes included); the constraint changed one token in 17 prompts. What it adds is not a measured accuracy gain but a
**guarantee**: whatever the model emits, the tool call parses and fits the schema. The two remaining errors are
semantic ("start the file explorer" → `file_search`; "how is the computer doing?" → no call) and constraints do not
fix those. The overhead is within measurement noise.

### Phase 6 — assistant integration ✅

The engine runs inside a development copy of a private voice assistant as its local brain (`brain = "engine"`),
started as a persistent `el serve` process. Stage 1 kept the safety decision in the assistant's audited Python
pipeline; in stage 2 the bekci replaced that pipeline as the decision-maker.

**Stage 1: the engine becomes the brain**

- **The prompt matches Ollama token for token.** Ollama 0.40 builds the prompt with the GGUF's HF chat template
  (converting tools to its own Go structure: `additionalProperties` dropped, empty `description` written, empty
  `required` omitted, keys sorted); the engine brain builds the exact equivalent. Token counts equal Ollama's in 6/6
  cases (all 45 tools: 5 828 = 5 828; two consecutive tool results in one user turn).
- **Constrained decoding** runs with the assistant's 45-tool schema (type, enum, range, `maxLength`, format). A call
  the constraint altered goes to the tool runner with the `altered` flag and **always asks for confirmation**
  (including tools that normally need none and session-wide network grants).
- **Sampling:** in free text, the settings Ollama uses for this model (temperature 0.2, top_k 20, top_p 0.8); inside
  a tool call decoding is greedy. Greedy free text fell into repetition loops ("tell me a joke" → 1 024 tokens).
- An emergency stop kills the engine process on the spot; it restarts on the next request.

Measurement: the assistant's `--brain-test` (62 scenarios, side-effect free), Qwen3-4B Q4_K_M, RTX 2060, same day,
same machine:

| | Right tool | Median | Mean | Slowest |
|---|---|---|---|---|
| Ollama (llama.cpp) | 62/62 | 1.0 s | 1.3 s | 5.4 s |
| Engine | 61/62 (1 call altered by constraint → goes to confirmation) | 2.0 s | 2.7 s | 20.5 s |
| Engine, new prompt attention | 61/62 | 1.8 s | 2.3 s | 32.1 s (repetition loop, below) |
| Engine, + repetition penalty (DRY) | 61/62 | 1.9 s | 2.4 s | 17.4 s |
| Engine, + narrowed re-ask turn (below) | 61/62 | 1.7 s | — | — |

The only miss is "Open notepad please": the engine first calls `list_apps`, then `open_app` (the result is right; the
metric looks at the first call). **The speed gap is in long prompts.** The assistant's router picks a different
tool subset per question, so the system block changes and the prefix cache keeps only the system prompt (~750
tokens); a turn that re-asks a tool-less reply with all tools processes ~6 800 tokens. Making the prompt layout
cache-friendly breaks accuracy (showing all tools at once dropped it to 86%), so the kernels were optimised instead:

**Long-prompt attention.** Nsight Systems: in a 6 800-token prompt **55% of the time** was prompt attention (~7.5 s;
~28% of the fp32 peak). The new `k_attn_prefill64` (64 queries per block, scores and output in registers, float4
shared-memory reads, next K/V tile prefetched) cut attention to **4.1 s** (~3.3 TFLOPS, half the card's fp32 peak);
math is still fp32, only the summation order changed.

| 4B, prompt tok/s | pp512 | pp2048 | pp6800 |
|---|---|---|---|
| old attention | 905 | 775 | 499 (13.6 s) |
| new attention | 1 019 | 851 | 646 (10.5 s) |

Accuracy: fp32 reference 11/11 on 0.6B and 4B (≤ 2.3e-5), 0 failures against Ollama; `tests\attn_long.py` on a
6 023-token prompt: three kernels produced the same tokens (max \|Δlogprob\| 1.6e-5 vs the old tiled kernel, 9.6e-5
vs the single-block one).

**Repetition loops.** At temperature 0.2 (nearly greedy), "tell me a joke" once in three runs wrote the same
pattern for 1 024 tokens (~19 s). The fix is a repetition penalty in `el serve` (DRY, the equivalent of llama.cpp's
sampler; the assistant sends 0.8 / 1.75 / 2): if the last L tokens of the reply occurred earlier in the reply in
the same order, the logit of the token that followed that occurrence drops by `0.8 · 1.75^(L-2)`; newline, `:`, `"`
and `*` break matches. It applies only to the reply's own free text: verbatim quotes from the prompt (tool results,
the user's words) are not penalised and tool-call bodies are untouched. Measured: the joke finished normally in 6/6
runs (77–246 tokens); with the penalty off `serve_test` matches greedy exactly; with it on, the 0.6B "write it 12
times" repetition went from 7 to 1 (cost: an explicitly requested repetition is also broken).

**Re-ask turn narrowed.** The assistant re-asks a tool-less reply with all tools when it contains a number or a
first-person claim. Measured over 62 scenarios + 16 chat sentences: 8 re-asks used all 46 tools (~6 500 tokens,
~12 s) and **none** produced a tool call; all were plain chat whose reply merely contained a number or a verb suffix
("born in 1881", "I'm working"). Weak triggers are now re-asked with the same narrow list as the first prompt
(prefix in the engine's cache, ~0.6 s); the full list is used only for a strong claim ("I'm saving it", "I opened
it"). Total re-ask time 98 s → 8 s; "what year was Atatürk born?" 13.7 → 0.9 s; brain-test 61/62 (median 1.7 s).
A second KV slot (fp16 KV, which would have changed the accuracy contract) was not needed.

**Stage 2: the bekci is the assistant's decision-maker**

With `guard = "bekci"` the verdict on every tool call (allow / confirm / deny), the confirmation ticket, the
session network grant and the taint state live in the bekci process. The assistant collects the confirmation in its
own UI (`confirm_ui = "client"`), but permission to execute comes only from the guard's `exec` id, with the arguments
the guard normalised.

- **The policy is not hand-maintained:** the assistant generates it at every start from its tool registry and
  config: per-tool danger level, `readonly` (23 tools), `network`, `untrusted`; parameter kinds (path → `path`,
  note/clipboard text → `multiline`, other text → single line, 2 000 characters). With network off, network tools
  never enter the policy; with a cloud brain, mail/calendar tools do not either. Allowed roots are the assistant's;
  protected folders (its data, the Startup folder) are in `deny`.
- **The Python validation is a second wall:** arguments the guard allowed must also pass the assistant's older
  rules; if they do not, the ticket is closed with "No" or `result ok=false` is reported (the guard keeps no open job).
- **Taint rule:** if file/web/clipboard content was read in this request, *every* side-effecting tool (even adding a
  note or opening an app) escalates to confirmation; read-only tools are not asked. The Python pipeline only did
  this for the network session grant.
- **The session network grant moved into the guard** (`[bekci] network_session = true`): confirming the first
  read-only network query can grant a session; it is suspended while untrusted content is in the conversation
  (`turn {untrusted_history}` reports history too), dropped on emergency stop, and unusable by constraint-altered
  calls. A failed untrusted result also taints (the error text can carry content).
- **Fail closed:** no guard, no answer, or a dead process means no tool runs. The process is restarted only at the
  start of a new user request (a mid-turn restart would not know what was read, loosening the rule). Emergency stop
  goes to the guard as `stop`, the next request sends `resume`.
- **Deliberate differences from the Python pipeline:** a single-line text with a newline or invisible/bidi
  character is rejected without asking (Python escaped and asked); a *read-only* call altered by the constraint does
  not go to confirmation (it can only produce a wrong answer).

Verification: bekci 20 + 11 + 3 tests (new: session grant, ticket without offer, failed result tainting); 13 tests in
the assistant against the real `bekci.exe` (allow, confirm Yes/No, outside the roots, taint escalation, constraint,
session grant, emergency stop, dead process, missing exe, second wall, policy validity over 6 settings
combinations). The assistant's whole suite in both modes: Python pipeline 535/535; with the guard enabled
523 pass and 13 are skipped because of the differences above or because they write internal state directly.

End to end on the GPU (Qwen3-4B, engine + bekci + the real orchestrator; every confirmation answered "No", tools that
touch the screen stubbed): all 8 scenarios got the expected decision at 3–8 s per turn — after listing the desktop,
an `open_app` in the same request escalated to confirmation (the reason is shown in the dialog), deleting
`C:\Windows\win.ini` was denied without asking, weather / volume 150% / clipboard read went to confirmation. Also
tried from the HUD: dialog → No → deny; STOP → guard `stop`; new command → `resume` → allow. The trial surfaced two
brain issues, both fixed in the assistant: the model did not know real folder paths and invented
`C:\Users\User\Desktop` (the context now carries the full allowed paths), and it could report a refused action as
done (the tool result now says the action was NOT performed and a closing note is appended if the reply does not).

## Architecture

```
src/gguf.*        GGUF v2/v3 parser: memory-mapped, zero-copy; the file is treated as UNTRUSTED input
src/tokenizer.*   byte-level BPE + a hand-written implementation of the qwen2 pre-tokenizer regex
src/unicode_tables.h  \p{L} / \p{N} ranges (generated by tools/gen_unicode.py)
src/model.*       config + weight table validation, Backend interface, CPU backend (fp32 reference)
src/ops.*         CPU: AVX2/F16C matrix-vector product, RMSNorm; OpenMP
src/quant.*       ggml quantization blocks (Q8_0, Q4_K, Q6_K) and row decoders (bit-identical to ggml)
src/cuda/model_cuda.cu  CUDA backend (below)
src/json.*        strict JSON reader (tool schema; rejects duplicate keys, invalid UTF-8, deep nesting)
src/constrain.*   constrained decoding: tool schema + byte-level call matcher
src/main.cpp      el info | tokenize | generate | bench | dequant | constrain-test | serve  (JSON output: token, logprob, top-k)
tests/            parity, speed and end-to-end guard tests (Python; shares no code with the engine)
bekci/            Rust guard: policy (policy loading and rules), validate (path jail, text, URL),
                  jsonx (JSON that rejects duplicate keys), calls (<tool_call> extraction), guard (decision,
                  confirmation tickets, taint, stop), log (actions.log), server (JSON-lines protocol), sys (Win32)
```

CUDA kernels:

| Kernel | What it does |
|---|---|
| `k_gemv<type>` | Single token: one row per warp, x in shared memory, persistent grid sized to the SM count; decodes F16 / Q8_0 / Q4_K / Q6_K weights while reading |
| `k_split8<NL>` + `k_gemm_i8<type,NL>` | Quantized batched prompt: splits x into NL int8 limbs; raw PTX `mma.sync m8n8k16` int8 tensor core, 64×128 tile, 16 warps, K step 128 (63 KB dynamic shared memory), sub-block scales in int32; `ldmatrix` fragment loads; next step's weights prefetched into registers |
| `k_split` + `k_gemm_q<type>` | Old quantized batched prompt (`EL_GEMM=hilo`): `mma.sync m16n8k8` fp16, hi/lo split |
| `k_split` + `k_gemm_tc` | F16 batched prompt: WMMA 16×16×16 tensor core, hi/lo split, 64×128 tile, sliced fp32 accumulation |
| `k_gemm` | Classic fp32 tiled multiply (for comparison, `EL_GEMM=simt`) |
| `k_attn_prefill64` | Batched prompt attention: a block handles 64 queries of one head with 32-wide K/V tiles; scores (4×2) and output (4×8) in registers per thread, float4 shared-memory reads, next tile prefetched into registers; online softmax, causal skipping, fp32 |
| `k_attn_prefill` | Old batched prompt attention (32 queries × 32 keys, kept for comparison with `EL_ATTN=tile32`) |
| `k_attn_split` + `k_attn_combine` | Decode attention (flash-decoding): keys split into 16 blocks, then combined |
| `k_headnorm_rope` | Per-head RMSNorm + RoPE (NEOX), angles from a table computed in double precision |
| `k_rmsnorm`, `k_silu_mul`, `k_store_kv` | Helpers (embedding rows are decoded on the CPU and copied) |

The decode step (≈340 kernels) is captured once as a **CUDA Graph** at the first token; position and token are read
from device memory, so the graph is replayed with a single call per token.

The Qwen3 layer:

```
h = rmsnorm(x);  q,k,v = Wq·h, Wk·h, Wv·h
q,k = RoPE(rmsnorm_head(q)), RoPE(rmsnorm_head(k))       # Qwen3-specific per-head q/k norm
x += Wo · softmax(q·kᵀ/√d + mask) · v                     # 16 query heads, 8 KV heads (GQA)
h = rmsnorm(x);  x += Wdown · (silu(Wgate·h) ⊙ Wup·h)
```

### Security
- In the GGUF parser every length/count/offset is checked against the file bounds; overflow, out-of-range tensors,
  misaligned data, duplicate names and unknown types reject the file. Unreasonable sizes (giant arrays, giant
  strings) are rejected before any allocation.
- Invalid UTF-8 input is rejected; token ids and context positions are bounds-checked before they reach the GPU.
- GPU memory is computed before loading; if it does not fit, loading stops with a clear error.
- The model's tool calls pass through the Rust guard (Phase 4 above).

## Build

Requirements: Windows 10/11 x64, Visual Studio 2022+ (C++ desktop workload), CMake ≥ 3.24, Ninja, an AVX2 CPU.
For the GPU build: CUDA Toolkit 12+ and an NVIDIA card with compute capability 7.5+ (Turing or newer). Without CUDA
only the CPU build is produced.

```powershell
.\build.ps1          # build\el.exe  (finds CUDA via CUDA_PATH or the default install folder)
cargo build --release --manifest-path bekci\Cargo.toml   # bekci\target\release\bekci.exe (Rust 1.80+)
```

## Usage

```powershell
ollama pull qwen3:0.6b-fp16        # or qwen3:1.7b-q8_0, qwen3:4b-instruct; or point EL_MODEL at any Qwen3 GGUF
$m = (python -c "import sys; sys.path.insert(0,'tests'); from common import model_path; print(model_path())")
.\build\el.exe info $m
Set-Content prompt.txt "The most populous city in Turkey is" -Encoding UTF8 -NoNewline
.\build\el.exe generate $m -f prompt.txt -n 16 --top 3            # --backend auto|cpu|cuda (default auto)
.\build\el.exe bench $m --pp 512 --tg 128
cmd /c ".\bekci\target\release\bekci.exe check-policy bekci\policy.example.toml > tools.json"   # raw UTF-8
.\build\el.exe generate $m -f prompt.txt -n 120 --top 1 --tools tools.json   # constrained decoding
```

(Ollama is used here only as a convenient place to obtain the GGUF files and as a speed/accuracy reference; the
engine itself does not need it.) With `--tools` the output gains `calls`, `constrained` (tokens chosen inside the
call), `forced` (tokens the constraint changed; `"forced": true` on that token), `open_call` (the call was still open
when context/`-n` ran out) and `altered_calls` (call indices to pass to the guard as `parse {..., "altered": [...]}`).

### Persistent server: `el serve`

An assistant loads the model once and keeps it open; requests arrive on stdin as JSON lines and replies go to
stdout (stderr is log only).

```powershell
.\build\el.exe serve $m --backend cuda --ctx 8192
{"ready":true,"backend":"cuda","n_ctx":8192,"ms_load":4120}
{"id":1,"prompt":"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n","n":256,"tools":[...]}
{"ok":true,"hex":"...","tokens":12,"stop":"eog","prompt_tokens":14,"reused":0,"ms_prompt":31,"ms_gen":160,"id":1}
{"op":"quit"}
```

- `prompt` is the full prompt (the chat template is built by the client); `n` is 1..32768 (default 256); if `tools` is
  given, constrained decoding works as with `--tools` and the reply gains `calls`, `constrained`, `forced`,
  `open_call`, `altered_calls`. The output text is returned as `hex` (UTF-8) for byte exactness; `stop` is
  `eog | length | ctx`.
- **Prefix reuse:** the longest common token prefix with the previous prompt + reply stays in the KV cache and only
  the rest is processed (`reused`). In a chat each turn computes only the new messages; the output is identical to
  running the same prompt from scratch with `generate` (`tests\serve_test.py`).
- Sampling: `"temperature"` (0..2, default 0 = greedy), `"top_k"` (1..1000, default 20), `"top_p"` ((0,1], default
  0.8), `"seed"` (same seed, same output). The chain is top_k → top_p → temperature; it applies to free text only —
  the inside of a tool call is always greedy.
- Repetition penalty: `"dry_multiplier"` (0..5, default 0 = off), `"dry_base"` (1..4, 1.75), `"dry_allowed_length"`
  (1..20, 2); when on, the reply carries `dry_hits` (steps where the penalty applied). Free text of the reply only.
- A malformed request returns `{"ok":false,"error":...}` and the process keeps running; a request line is at most
  16 MB.

The prompt can also be piped through stdin, but the PowerShell 5.1 pipe corrupts non-ASCII characters and appends a
newline, so `-f` is recommended.

Environment variables for comparison: `EL_GEMM=simt` (fp32 multiply instead of the tensor core on F16),
`EL_GEMM=mma` (the `mma.sync` kernel on F16 too), `EL_GEMM=hilo` (hi/lo fp16 kernel instead of int8 on quantized
prompts), `EL_NL3=<mask>` (layers that use 3 int8 limbs: 1 qkv, 2 wo, 4 gate/up, 8 down; default 6, 0 = 4 limbs
everywhere), `EL_NL2=<mask>` (experimental 2 limbs, off by default, breaks the accuracy contract), `EL_ATTN=naive`
(old single-block attention), `EL_ATTN=tile32` (old 32-wide tiled prompt attention), `EL_NO_GRAPH=1` (CUDA Graph off).

## Tests

Requires Python 3.10+, `numpy`, `tokenizers`, `gguf` (the independent decoder for quantized models); Ollama must be
running for the Ollama comparisons. `--model` is an Ollama model name in every test (default `qwen3:0.6b-fp16`);
the tests unload Ollama from the GPU before the engine runs.

```powershell
python tests\parity_tokenizer.py                     # vs HF tokenizers (no Ollama needed)
python tests\parity_dequant.py --model qwen3:4b-instruct  # quantized decoders vs the gguf package
python tests\parity_reference.py --backend cuda      # vs the independent numpy fp32 implementation (cpu | cuda)
python tests\parity_model.py --backend cuda          # end to end vs Ollama
python tests\bench.py [--cpu]                        # speed: engine and Ollama
cargo test --manifest-path bekci\Cargo.toml          # guard: unit, attack and protocol tests
python tests\e2e_bekci.py [--backend cpu]            # the engine produces tool calls, the guard checks them
python tests\constrain_bekci.py                      # matcher: attack table + fuzzing, cross-checked with the guard
python tests\constrain_compare.py [--backend cpu]     # same prompts, constrained vs unconstrained
python tests\serve_test.py [--backend cpu]           # el serve = el generate (prefix cache, sampling, errors)
python tests\attn_long.py [--tokens 6000]            # long context: three attention kernels must give the same tokens
```

The guard on its own:

```powershell
.\bekci\target\release\bekci.exe check-policy bekci\policy.example.toml   # validate the policy, print the tool schema
.\bekci\target\release\bekci.exe serve --policy bekci\policy.example.toml
{"op":"parse","text":"<tool_call>{\"name\":\"open_app\",\"arguments\":{\"name\":\"Paint\"}}</tool_call>"}
{"calls":[{"args":{"name":"Paint"},"decision":"allow","exec":1,"tool":"open_app"}],"ok":true}
```

Ops: `hello`, `tools` (the tool schema to put into the model's prompt), `parse {text, altered?}`,
`check {call, altered?}` (`altered`: calls the engine changed through the constraint; escalates side-effecting
tools to confirmation), `confirm {ticket, answer, session?}` (only with `confirm_ui = "client"`; `session`: grant a
session network permission together with the confirmation, only if the ticket carried `"session": "offer"`),
`result {exec, ok, detail?}`, `turn {untrusted_history?}` (a new user instruction; if the history still holds
untrusted content the session grant stays suspended), `stop`, `resume`, `quit`. An allow reply may carry
`via: "session_grant"`, a confirm reply `session: "offer" | "suspended"`. Spontaneous events are `{"event": ...}`
lines (confirmation decided, emergency stop). The dialog-mode test opens a window on screen, so it only runs with
`BEKCI_DIALOG_TEST=1`.

## Known limitations

- Loading takes 3–6 s (CUDA context + copying weights from the mapped file to VRAM); not a problem for an assistant
  that loads once and stays resident.
- The KV cache is fp32: 288 KB per token at 4B, 224 KB at 1.7B/0.6B. With 4B on a 6 GB card about 8k tokens of
  context fit; longer context needs KV in fp16/int8, which changes the accuracy contract and must be measured.
- Prompt processing on quantized models is ~1.65–2.1× slower than Ollama (above; the price of fp32 accuracy). GPU
  performance counters need administrator rights, so kernel tuning relied on Nsight Systems timings and ablation.
- Constrained decoding fixes the key order (`name` first, `arguments` second; Qwen3 writes it that way) and forbids
  `\u` escapes in strings (characters are written directly as UTF-8). There are no nested object/array parameters
  (the policy has none either).
- Weight types: F16, Q8_0, Q4_K, Q6_K (all of Q4_K_M and Q8_0 files). Qwen3 dense architecture only, greedy
  selection in constrained segments; Windows/MSVC builds only.

## Roadmap

1. ~~Correctness: GGUF, tokenizer, CPU fp32 forward pass, parity tests~~ ✅
2. ~~CUDA kernels, batched prompt, CUDA Graph, speed measurement~~ ✅
3. ~~Scaling: Qwen3-1.7B / 4B, Q8_0 / Q4_K / Q6_K quantization (fitting 6 GB of VRAM)~~ ✅
4. ~~Rust guard: tool-call validation, allow-list, confirmation, `actions.log`; JSON IPC~~ ✅
5. ~~Constrained decoding: tool calls are always valid JSON / schema~~ ✅
6. Assistant integration: ~~stage 1 — engine brain, constrained decoding, Python safety pipeline~~ ✅;
   ~~stage 2 — the guard replaces the Python pipeline~~ ✅; ~~long-prompt speed (the re-ask turn)~~ ✅;
   ~~GEMM weight prefetch~~ ✅

## License

No license has been chosen yet; until one is added, all rights are reserved.
