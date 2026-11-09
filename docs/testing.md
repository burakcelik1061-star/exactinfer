# Tests

The tests are Python scripts that share no code with the engine. The fp32 reference is an independent numpy
implementation, and the decoders it uses for quantized weights come from the `gguf` package. The guard has its own
Rust test suite.

Requirements:
- Python 3.10+ with `numpy`, `tokenizers` and `gguf`.
- For comparisons against Ollama, a running Ollama.
- `--model` is an Ollama model name in every script (default `qwen3:0.6b-fp16`). `EL_MODEL=<path>` overrides it with
  a GGUF file.
- Before the engine runs, the tests unload Ollama's copy from the GPU.

| Command | Needs | Checks | Expected |
|---|---|---|---|
| `python tests\parity_tokenizer.py` | — | tokenizer vs HF `tokenizers` | 30 042 / 30 042 identical |
| `python tests\parity_dequant.py --model qwen3:4b-instruct` | — | quantized row decoders vs `gguf` | bit-identical |
| `python tests\parity_reference.py --backend cuda` | GPU (or `--backend cpu`) | forward pass vs numpy fp32, 11 prompts | 11/11, \|Δlp\| ~1e-5 |
| `python tests\parity_model.py --backend cuda` | GPU, Ollama | greedy generation vs Ollama | 0 failures; near-ties allowed |
| `python tests\bench.py [--cpu]` | GPU, Ollama | speed, engine and Ollama | see [performance.md](performance.md) |
| `python tests\attn_long.py [--tokens 6000]` | GPU | three attention kernels on a long prompt | same tokens |
| `python tests\serve_test.py [--backend cpu]` | — | `el serve` = `el generate` (prefix cache, sampling, errors) | all pass |
| `python tests\constrain_bekci.py` | built guard | matcher attack table + fuzzing, cross-checked with the guard | 82/82; no dead end |
| `python tests\constrain_compare.py [--backend cpu]` | built guard | constrained vs unconstrained on 17 prompts | see [tool-calls.md](tool-calls.md) |
| `python tests\e2e_bekci.py [--backend cpu]` | built guard | engine writes real calls, guard judges them | 6/6 |
| `cargo test --manifest-path bekci\Cargo.toml` | Rust | guard unit, attack and protocol tests | all pass |

GPU speed results depend on temperature. To compare two builds, alternate them run by run
([performance.md](performance.md#measurement-methodology)).
