# `el` command line and `el serve` protocol

## One-shot commands

```powershell
.\build\el.exe info $m                                         # GGUF metadata and tensor table
.\build\el.exe generate $m -f prompt.txt -n 16 --top 3          # --backend auto|cpu|cuda (default auto)
.\build\el.exe bench $m --pp 512 --tg 128
.\build\el.exe generate $m -f prompt.txt -n 120 --top 1 --tools tools.json   # constrained decoding
```

`generate` prints JSON with each token's bytes (`hex`), logprob and top-k alternatives. The parity tests consume
this output.

With `--tools`, the output also contains:

| Field | Meaning |
|---|---|
| `calls` | the tool calls in the reply |
| `constrained` | number of tokens chosen inside a call |
| `forced` | number of tokens the constraint changed; the token itself carries `"forced": true` |
| `open_call` | a call was still open when the context or `-n` ran out |
| `altered_calls` | call indices to pass to the guard as `parse {..., "altered": [...]}` |

Prefer `-f prompt.txt` to piping. The PowerShell 5.1 pipe corrupts non-ASCII characters and appends a newline.

## `el serve`: a persistent process

An assistant loads the model once and keeps it resident. Requests arrive on stdin as JSON lines and replies go to
stdout; stderr is for logging only.

```text
> .\build\el.exe serve $m --backend cuda --ctx 8192
{"ready":true,"backend":"cuda","n_ctx":8192,"ms_load":4120}
{"id":1,"prompt":"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n","n":256,"tools":[...]}
{"ok":true,"hex":"...","tokens":12,"stop":"eog","prompt_tokens":14,"reused":0,"ms_prompt":31,"ms_gen":160,"id":1}
{"op":"quit"}
```

- `prompt` is the full prompt; the client builds the chat template. `n` is 1..32768 (default 256). The output text
  comes back as `hex` (UTF-8) so it is byte-exact. `stop` is `eog | length | ctx`.
- If `tools` is given, constrained decoding runs as with `--tools`. The reply gains `calls`, `constrained`,
  `forced`, `open_call` and `altered_calls`.
- **Prefix reuse:** the longest common token prefix with the previous prompt + reply stays in the KV cache, and only
  the rest is processed (`reused` says how many tokens were kept). In a chat each turn computes only the new
  messages. The output is identical to running the same prompt from scratch with `generate`
  (`tests\serve_test.py` checks this).
- **Errors:** a malformed request returns `{"ok":false,"error":...}` and the process keeps running. A request line
  is at most 16 MB.

### Sampling

| Field | Range | Default |
|---|---|---|
| `temperature` | 0..2 | 0 (greedy) |
| `top_k` | 1..1000 | 20 |
| `top_p` | (0, 1] | 0.8 |
| `seed` | any | same seed → same output |

The chain is top_k → top_p → temperature. It applies to free text only; **the inside of a tool call is always
greedy**.

### Repetition penalty (DRY)

| Field | Range | Default |
|---|---|---|
| `dry_multiplier` | 0..5 | 0 (off) |
| `dry_base` | 1..4 | 1.75 |
| `dry_allowed_length` | 1..20 | 2 |

DRY is the equivalent of llama.cpp's sampler of the same name. Suppose the last L tokens of the reply already
occurred earlier in the reply, in the same order. Then the logit of the token that followed that earlier occurrence
drops by `multiplier · base^(L − allowed_length)`. Newline, `:`, `"` and `*` break matches.

The penalty applies only to the reply's own free text:
- verbatim quotes from the prompt (tool results, the user's words) are not penalised;
- tool-call bodies are untouched.

When it is on, the reply carries `dry_hits`, the number of steps where it applied. Why it exists is covered in
[integration.md](integration.md#repetition-loops).

## Environment variables

These are for comparison and experiments. The defaults are the measured, accuracy-preserving choices.

| Variable | Effect |
|---|---|
| `EL_MODEL=<path>` | GGUF file for the test scripts; takes precedence over their `--model` (an Ollama model name) |
| `EL_GEMM=simt` | fp32 tiled multiply instead of the tensor core on F16 |
| `EL_GEMM=mma` | the `mma.sync` kernel on F16 too |
| `EL_GEMM=hilo` | hi/lo fp16 kernel instead of int8 limbs on quantized prompts |
| `EL_NL3=<mask>` | matrices that use 3 int8 limbs: 1 qkv, 2 wo, 4 gate/up, 8 down; default 6, 0 = 4 limbs everywhere |
| `EL_NL2=<mask>` | experimental 2 limbs; **off by default, breaks the accuracy contract** ([accuracy.md](accuracy.md#the-trade-off-measured-the-2-limb-experiment)) |
| `EL_ATTN=naive` | old single-block attention |
| `EL_ATTN=tile32` | old 32-wide tiled prompt attention |
| `EL_NO_GRAPH=1` | CUDA Graph off |
