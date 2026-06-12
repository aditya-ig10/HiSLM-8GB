# Bug Analysis: Streaming & Response Quality Issues

## Overview

Multi-turn medical QA via `llama-cli` produced two failures:
1. **Blank responses** (`"———"`) on the 2nd+ turn
2. **Irrelevant/generic responses** instead of medical answers

Three independent bugs conspired to cause these failures. Each is documented below.

---

## Bug 1: LoRA Degrades English Medical QA

### Symptom
- `"Helloooo"` → `"Hello, it is nice to meet you."` (generic chat)
- `"i am having high temperature with stomach aching"` → `"Hello, how are you today?"` (ignores query entirely)
- `"what is Hypothermia"` → `"Hypothermia is a state where the core body temperature of the human is below 35.5°C (95.9°F)."` (passable)

### Root Cause
The `medical-lora-qwen2.5-1.5b.gguf` was trained on Chinese medical QA data. When applied to English queries at scale 1.0, it **overrode** the base model's English instruction-following capability. The base Qwen2.5-1.5B-Instruct (without LoRA) gave vastly superior answers:

| Query | With LoRA | Without LoRA |
|---|---|---|
| `"i am having high temperature with stomach aching"` | `"Hello, how are you today?"` | *Lists 5 possible causes (infection, IBD, GERD, food poisoning, etc.) with treatment advice* |
| `"ig i am having food poisioning"` | *(0 chars — see Bug 2)* | *5-step management plan (hydration, bland diet, rest, meds, monitoring)* |

### Fix
Removed `--lora` flag from `llama-cli` invocation. The base model alone provides strong English medical QA (it was pre-trained on clinical/medical text).

---

## Bug 2: Echo Truncation Causes 0-Char Responses

### Symptom
Multi-turn conversations return empty strings on the 2nd+ turn.

### Root Cause
The original `stream_tokens()` used **header counting** to find where generation starts:

```python
n_expected = prompt.count("<|im_start|>assistant\n")
assistant_count = 0
# ... wait until assistant_count >= n_expected, then yield
```

This assumed `llama-cli` faithfully echoes the full prompt. **It doesn't.** With `--single-turn --simple-io`, the echo is truncated:

- `<|im_end|>` is dropped from echoed assistant content
- The last user/assistant turn may be omitted entirely
- Content lines longer than ~390 chars are silently truncated

Example raw output (multi-turn prompt with 4 assistant headers, but only 3 echoed):

```
[28] <|im_start|>assistant\n                ← header 3 (of 4 expected)
[29] Hypothermia is a... (390 chars, no <|im_end|>)  ← content truncated
[30] \n                                       ← blank (separator)
[31] <generation>                             ← model output
```

The counter expects 4 headers, the echo only emits 3, so `assistant_count` never reaches `n_expected`. The state machine never enters response mode → 0 chars yielded.

### Fix
Replaced the line-by-line state machine with a **buffered extraction** approach:

```python
out, _ = proc.communicate()           # read ALL output
response = _extract_response(out)      # find generation in the buffer
for ch in response:
    yield ch
```

The new `_extract_response()` uses a robust heuristic:

1. Split raw output by `\n`
2. Find the **last** line starting with `<|im_start|>assistant`
3. After it, find the **first blank line** (echo-to-generation separator)
4. Everything between the separator and `[ Prompt:` / `Exiting` is the generation
5. Strip and return

This works regardless of echo truncation because it only relies on the LAST assistant header and the blank-line separator — both of which survive truncation.

---

## Bug 3: Per-Character Chunking Overwhelms WS/SSE

### Symptom
WebSocket handler sent 1000+ frames per response (one per character), causing client-side performance issues and potential WS timeouts.

### Root Cause
`stream_tokens()` yielded individual characters:

```python
for ch in response:
    yield ch
```

The WS handler sent each character as a separate JSON chunk:

```python
for token in stream_tokens(prompt):
    ws.send(json.dumps({"type": "chunk", "content": token}))
```

### Fix
Batch into 80-char chunks for both SSE and WS handlers:

```python
full = "".join(stream_tokens(prompt))
for i in range(0, len(full), 80):
    ws.send(json.dumps({"type": "chunk", "content": full[i:i+80]}))
```

---

## Summary of Changes

| File | Change |
|---|---|
| `server_qwen.py:21-26` | Removed `LORA_MODEL` constant |
| `server_qwen.py:56-89` | Rewrote `_extract_response()` to use `split("\n")` + last-header heuristic |
| `server_qwen.py:92-123` | Rewrote `stream_tokens()` to use `proc.communicate()` + `_extract_response()` |
| `server_qwen.py:161-165` | SSE handler: batch 80-char chunks |
| `server_qwen.py:187-196` | WS handler: batch 80-char chunks, remove `stdout=PIPE` args |
| `orin_index.html:953` | Default system prompt strengthened |
| `orin_index.html:1552-1556` | Auto-migration of old "Orin NX" system prompt in localStorage |

## Verification

Multi-turn test results (base model, no LoRA):

```
Turn 1 "hello" → 34c  "Hello! How can I assist you today?"
Turn 2 "high temp + stomach ache" → 1408c  medical advice, NO hypothermia leak
Turn 3 "ig i am having food poisioning" → 1614c  food poisoning advice, NO hypothermia leak
```
