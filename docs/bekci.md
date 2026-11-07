# bekci: the tool-call guard

*bekci* (Turkish for *guard*) is a separate Rust process that receives the model's text and returns a decision for
every tool call in it:
- **allow**, with an execution id;
- **confirm**, with a single-use ticket;
- **deny**, with a reason.

The executor runs only calls the guard allowed, and only with the arguments the guard normalised. The guard speaks
JSON lines over stdin/stdout. Its only dependencies are serde, serde_json, toml and windows-sys.

Keeping it a separate process in a memory-safe language means the decision does not live in the same address space
as the model, the GPU driver or the host application's Python. It also has a small, reviewable surface.

```powershell
cargo build --release --manifest-path bekci\Cargo.toml
.\bekci\target\release\bekci.exe check-policy bekci\policy.example.toml   # validate the policy, print the tool schema
.\bekci\target\release\bekci.exe serve --policy bekci\policy.example.toml
```

```text
{"op":"parse","text":"<tool_call>{\"name\":\"open_app\",\"arguments\":{\"name\":\"Paint\"}}</tool_call>"}
{"calls":[{"args":{"name":"Paint"},"decision":"allow","exec":1,"tool":"open_app"}],"ok":true}
```

## Rules

| Rule | How it is enforced |
|---|---|
| Only defined tools; no free-form commands | A tool not in the policy does not exist. A `blocked` tool is never shown to the model. A `kind = "shell"` parameter loads only with `shell_mode = true`, and that tool always needs confirmation |
| Risky actions always confirm, default "No" | Applies to `confirm` tools and every network tool (enforced at load time). The guard can open the dialog itself, with "No" as the default button. In client mode the client collects the answer but cannot grant execution. Arguments are bound to the ticket and cannot change after approval. Tickets are 128-bit random and single-use. No answer within the timeout counts as "No" |
| Every call, confirmation and result is logged | `actions.log`: JSONL, UTC timestamps, rotates at 5 MB. Secret-looking fields (`api_key`, `parola`, `sk-…`) are masked. If the log cannot be written, the action is denied |
| Emergency stop | Global hotkey (default Ctrl+Alt+End) or the `stop` op. Pending confirmations drop, running jobs are reported for cancellation, and every call is denied until `resume` |
| Not run as admin; parameters validated | The guard exits if started elevated. Path jail and type checks: see the next table |
| Instructions inside content are not commands | After a tool whose output is third-party content (file, web, clipboard), every side-effecting tool is escalated to confirmation until the user gives a new instruction. Duplicate JSON keys (`{"name":"a","name":"b"}`) are rejected at every level. A half-open or nested `<tool_call>` rejects the whole message. The guard's own policy and log folders are closed to tools |
| Constraint-altered calls confirm | A call in `altered` (see [tool-calls.md](tool-calls.md)) escalates to confirmation unless the tool is read-only |
| No secrets in code or files | A policy file with a key named like api_key, secret, password or token is rejected |

Parameter validation:

| Check | Rule |
|---|---|
| Path jail | Absolute drive paths only. Rejected: UNC and device namespace, `..`, alternate data streams (`:`), device names (CON, NUL…), trailing dots or spaces. Junctions and symlinks are resolved to the real path, so an escape through a junction inside an allowed folder is caught |
| Types | Exact: `true` is not a number, `1.0` is not an integer. Extra or missing fields are rejected. Range and enum checks |
| Text | Invisible and bidi characters are rejected |
| URLs | `https://` only |

## Taint rule and session network grant

If file, web or clipboard content was read in the current user request, **every** side-effecting tool escalates to
confirmation, even adding a note or opening an app. Read-only tools are not asked. A failed untrusted read also
taints, because the error text can carry content.

With `network_session = true`, confirming the first read-only network query can grant network access for the rest of
the session. The grant:
- is suspended while untrusted content is in the conversation (`turn {untrusted_history}` reports the history too);
- is dropped on emergency stop;
- cannot be used by constraint-altered calls.

## Fail closed

No guard, no answer, or a dead process means **no tool runs**. The integration restarts the guard only at the start
of a new user request. A restart mid-turn would not know what had been read and would loosen the taint rule.

## Protocol

| Op | Purpose |
|---|---|
| `hello` | handshake |
| `tools` | the tool schema to put into the model's prompt (also the constrained decoder's input) |
| `parse {text, altered?}` | extract and judge every call in the model's text |
| `check {call, altered?}` | judge a single call |
| `confirm {ticket, answer, session?}` | only with `confirm_ui = "client"`; `session` grants the network session together with the confirmation, only if the ticket carried `"session": "offer"` |
| `result {exec, ok, detail?}` | report the outcome of an allowed call |
| `turn {untrusted_history?}` | a new user instruction; clears taint, keeps the session grant suspended if history is still untrusted |
| `stop`, `resume`, `quit` | emergency stop, resume, exit |

An allow reply may carry `via: "session_grant"`; a confirm reply may carry `session: "offer" | "suspended"`.
Spontaneous events, such as a confirmation decided in the dialog or an emergency stop, arrive as `{"event": ...}` lines.

## Verification

```powershell
cargo test --manifest-path bekci\Cargo.toml        # unit, attack and protocol tests
python tests\e2e_bekci.py --backend cuda            # the engine writes real calls, the guard judges them
```

**Cargo tests:**
- 20 unit tests.
- Attack scenarios: path traversal, junction escape, type confusion, ticket replay, timeout, escalation after
  untrusted data, escalation of constraint-altered calls, emergency stop, log masking, session grant, a ticket without
  an offer, and failed-result tainting.
- Process-level protocol tests.

The dialog-mode test opens a window on screen, so it runs only with `BEKCI_DIALOG_TEST=1`.

**End to end (`e2e_bekci.py`):** the engine produces calls from the tool list the guard supplies. All 6 scenarios
got the expected decision, among them:

| Request | Decision |
|---|---|
| open the calculator | allow |
| delete a file | confirm |
| open a web address | confirm |
| read `C:\Windows\System32\config\SAM` | deny |
| shut down the computer | no call: no shutdown tool is shown to the model |

The guard has also been the decision-maker of a real assistant; see [integration.md](integration.md#stage-2-the-guard-decides).
