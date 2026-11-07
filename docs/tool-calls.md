# Constrained tool-call decoding

A model that can call tools is an attack surface. If its output is parsed leniently, the parser becomes part of the
security boundary. exactinfer removes the parsing problem at the source: **inside a tool call, the engine can only
emit bytes that keep the call valid against the tool schema.** It also reports every place where the constraint
overrode the model's choice, so a call the model did not intend is never executed silently.

```powershell
cmd /c ".\bekci\target\release\bekci.exe check-policy bekci\policy.example.toml > tools.json"
.\build\el.exe generate $m -f prompt.txt -n 120 --top 1 --tools tools.json
```

The tool list is the guard's own `tools` output, so the schema the decoder enforces is the one the guard validates.

## How it works

When the model emits the `<tool_call>` special token, the engine switches to constrained mode. At each step it picks
the highest-logit token **that the matcher accepts**. The matcher is a byte-level state machine (`src/constrain.*`;
the schema reader is `src/json.*`) that accepts only

```
{"name": <tool in the list>, "arguments": {<the tool's parameters, each at most once, required ones included>}}
```

- **Exact types:** string, integer (with range), number (with range, no exponent), `true`/`false`. Enums and tool
  names come from the list.
- **Strings follow the guard's own validators:**
  - no invisible or bidi characters, no `\u` escapes, no raw control characters;
  - `maxLength` counts code points.
  - Each parameter kind has its own rules:

    | Kind | Allowed |
    |---|---|
    | `name` | letters, digits, space, `_ - .`; no leading or trailing space |
    | `path` | an absolute drive path; no `:*?"<>|`, device names or components ending in a dot or space |
    | `uri` | `https://` only; no spaces, no `user@` |
    | `multiline` | the only kind where `\n` and `\t` are allowed |
- **No escape hatch:**
  - Once a call is complete, only `</tool_call>` can follow.
  - End of sequence inside a call is forbidden.
  - In free text, `</tool_call>` and spelling the tags out as plain text are forbidden, so the constraint cannot be
    bypassed by writing a call "by hand".

### The invariant: no dead ends

**Every prefix the matcher accepts can be completed into a valid call.** Without this, the decoder could paint itself
into a corner where no next byte is legal. The cases that need explicit completability checks:
- half-written UTF-8 characters;
- integer prefixes (after `-`; range bounds compared as u64 magnitudes);
- path components (a root such as `con.` can never become valid);
- the last character before a length limit.

`el constrain-test --fuzz` tries all 256 bytes at every step of random walks to check this.

## Constraints guarantee validity, not intent

This was found by measurement. Asked to "set the volume to 150 percent", the model was going to write `150`, but the
schema range is 0..100. The constraint closed the number at `15`, which is schema-valid and not what the user asked
for.

So whenever the constraint rejects the model's most likely token and picks another, the call is listed in
`altered_calls`. The guard **escalates an altered call to confirmation**, even for tools that normally need none and
even under a session-wide network grant. The one exception is read-only tools, where an altered call can only produce
a wrong answer. An altered call never runs silently.

The path jail (allowed root folders) depends on the file system, so it is not part of the schema; the guard enforces
it ([bekci.md](bekci.md)).

## Verification

| Test | What it covers | Result |
|---|---|---|
| `python tests\constrain_bekci.py` (no model) | 82 hand-written attack and boundary cases (see below), each also checked against the guard | all as expected; the guard also rejected the ones that are invalid under the policy |
| `el constrain-test --fuzz` (run by the same script) | random walks trying all 256 bytes per step | **18 000 walks, no dead end** |
| same | 9 000 completed random calls under the policy schema, sent to the guard | **all accepted**: 6 444 allow, 1 113 confirm, 1 443 deny only because the path was outside the allowed folders |
| `python tests\constrain_compare.py` | 17 real prompts, constrained vs unconstrained (table below) | — |

The 82 attack cases include:
- tools outside the allow-list; extra, missing and duplicate fields;
- type confusion; numbers written as `05`, `-0` or `1e1`;
- an RLO character and `‮`;
- path tricks: `..`, UNC, ADS, `NUL`, `con.txt`, `COM1 .log`;
- `http://` and `https://user@…` URLs.

On a model (Qwen3-0.6B F16, CPU, 17 prompts, greedy):

| | Right tool + schema-valid | Generation speed |
|---|---|---|
| Unconstrained | 14/17 | 15.4 tok/s |
| Constrained | 15/17 (1 call altered by the constraint → escalated) | 15.0 tok/s |

At this size the model already writes valid JSON on these prompts, including inner quotes, single-backslash Windows
paths and multiline notes. The constraint changed one token in 17 prompts. What it adds is not a measured accuracy
gain but a **guarantee**: whatever the model emits, the call parses and fits the schema.

The two remaining errors are semantic: "start the file explorer" → `file_search`, and "how is the computer doing?" →
no call. Constraints do not fix those. The overhead is within measurement noise.

## Limits

- Key order is fixed: `name` first, `arguments` second. Qwen3 writes it that way.
- `\u` escapes are forbidden; characters are written directly as UTF-8.
- No nested object or array parameters; the policy has none either.
- Decoding is greedy inside constrained segments.
