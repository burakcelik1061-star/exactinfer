# Real-world integration: a private voice assistant

The engine was built to be the local brain of a private Windows voice assistant. That assistant is not part of this
repository. It is where the tool-calling, the guard and the long-prompt work were driven by real use instead of
benchmarks. This page records what was measured and what it taught.

The assistant starts `el serve` as a persistent process (`brain = "engine"`). It has two stages:
- **Stage 1:** the safety decision stayed in the assistant's existing, audited Python pipeline.
- **Stage 2:** the bekci replaced that pipeline as the decision-maker (`guard = "bekci"`).

## Stage 1: the engine as the brain

- **The prompt matches Ollama token for token.** Ollama builds the prompt from the GGUF's HF chat template and
  converts the tools to its own Go structure:
  - `additionalProperties` is dropped;
  - an empty `description` is written out;
  - an empty `required` is omitted;
  - keys are sorted.

  The engine brain builds the exact equivalent. Token counts equal Ollama's in 6/6 cases, including all 45 tools
  (5 828 = 5 828) and two consecutive tool results in one user turn.
- **Constrained decoding** runs with the assistant's 45-tool schema (type, enum, range, `maxLength`, format). A
  call the constraint altered reaches the tool runner with the `altered` flag and always asks for confirmation.
- **Sampling:** free text uses the settings Ollama uses for this model (temperature 0.2, top_k 20, top_p 0.8).
  Inside a tool call decoding is greedy. Fully greedy free text fell into repetition loops.
- **Emergency stop** kills the engine process on the spot; it restarts on the next request.

### Brain test

The assistant has a 62-scenario `--brain-test` that checks the first tool the model chooses, with no side effects.
It ran on Qwen3-4B Q4_K_M on the RTX 2060, same day, same machine:

| | Right tool | Median | Mean | Slowest |
|---|---|---|---|---|
| Ollama (llama.cpp) | 62/62 | 1.0 s | 1.3 s | 5.4 s |
| Engine | 61/62 | 2.0 s | 2.7 s | 20.5 s |
| + new prompt attention | 61/62 | 1.8 s | 2.3 s | 32.1 s (a repetition loop) |
| + repetition penalty (DRY) | 61/62 | 1.9 s | 2.4 s | 17.4 s |
| + narrowed re-ask turn | 61/62 | **1.7 s** | — | — |

The only miss is "Open notepad please". The engine first calls `list_apps`, then `open_app`. The outcome is right,
but the metric looks at the first call. **In practice the two brains choose equally well; the engine is slower on
long prompts.**

### Lesson 1: the long prompt was the cost, so the kernels were fixed, not the prompt

The assistant's router picks a different tool subset per question. The system block therefore changes, and the
prefix cache keeps only the system prompt (~750 tokens). Making the prompt cache-friendly by showing all tools every
time dropped accuracy to 86%. That was rejected, and the prompt kernels were optimised instead.

Nsight Systems showed that in a 6 800-token prompt **55% of the time** was prompt attention (~7.5 s, ~28% of the fp32
peak). `k_attn_prefill64` cut it to 4.1 s with fp32 math unchanged ([performance.md](performance.md)).

### Lesson 2: measure the trigger before optimising the path

The assistant re-asked a tool-less reply with *all* tools when the reply contained a number or a first-person claim.
This guards against "I did it" replies with no actual call.

Measured over 62 scenarios plus 16 chat sentences, 8 re-asks used all 46 tools (~6 500 tokens, ~12 s each), and
**none** produced a tool call. All of them were plain chat whose reply happened to contain a year or a verb suffix.

Weak triggers are now re-asked with the same narrow tool list as the first prompt, which is still in the engine's KV
cache (~0.6 s). The full list is kept for strong claims ("I'm saving it", "I opened it").
- Total re-ask time: 98 s → 8 s.
- "What year was Atatürk born?": 13.7 s → 0.9 s.
- Brain test still 61/62.

A second KV slot, which would have needed an fp16 KV cache and so a change to the accuracy contract, was not needed.

### Repetition loops

At temperature 0.2, "tell me a joke" wrote the same pattern for 1 024 tokens (~19 s) in one run out of three. `el
serve` gained a DRY repetition penalty; the assistant sends multiplier 0.8, base 1.75, allowed length 2.

| | Result |
|---|---|
| the joke, penalty on | finished normally in 6/6 runs (77–246 tokens) |
| penalty off | `serve_test.py` matches greedy exactly |
| "write it 12 times" on 0.6B | repetitions went from 7 to 1; an explicitly requested repetition is also broken, which is the cost |

## Stage 2: the guard decides

With `guard = "bekci"`, the verdict on every call lives in the bekci process, along with the confirmation ticket, the
session network grant and the taint state. The assistant shows the confirmation in its own UI (`confirm_ui =
"client"`), but permission to execute comes only from the guard's `exec` id, with the guard's normalised arguments.

- **The policy is generated, not hand-maintained.** At every start the assistant writes it from its tool registry and
  config:
  - per-tool danger level, `readonly` (23 tools), `network`, `untrusted`;
  - parameter kinds (path → `path`; note and clipboard text → `multiline`; other text → single line, 2 000
    characters).

  With network off, network tools never enter the policy. Allowed roots are the assistant's; its own data and the
  Startup folder are in `deny`.
- **The old Python validation stays as a second wall.** Arguments the guard allowed must also pass the older rules.
  If they do not, the ticket is closed with "No" or `result ok=false` is reported.
- **The taint rule became stricter than before.** The Python pipeline escalated only the network grant after reading
  untrusted content; the guard escalates every side-effecting tool.
- **Deliberate differences from the Python pipeline:**
  - A single-line text containing a newline or an invisible/bidi character is rejected without asking.
  - A *read-only* call altered by the constraint does not go to confirmation.

**Verification:**
- 13 assistant tests run against the real `bekci.exe`. They cover allow, confirm Yes/No, outside the roots, taint
  escalation, constraint, session grant, emergency stop, dead process, missing exe, the second wall, and policy
  validity over 6 settings combinations.
- The assistant's full suite passes in both modes: Python pipeline 535/535. With the guard enabled, 523 pass and 13
  are skipped (the deliberate differences above, or tests that write internal state directly).

**End to end on the GPU** (Qwen3-4B, engine + bekci + the real orchestrator; every confirmation answered "No"; tools
that touch the screen stubbed): all 8 scenarios got the expected decision, at 3–8 s per turn.
- After listing the desktop, an `open_app` in the same request escalated to confirmation, and the dialog showed why.
- Deleting `C:\Windows\win.ini` was denied without asking.
- Weather, "volume 150%" and a clipboard read went to confirmation.

Two brain issues surfaced and were fixed in the assistant:
- The model did not know real folder paths and invented `C:\Users\User\Desktop`. The context now carries the full
  allowed paths.
- It could report a refused action as done. The tool result now says the action was NOT performed, and a closing note
  is appended if the reply does not say so.
