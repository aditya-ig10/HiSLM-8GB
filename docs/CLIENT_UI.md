# HiSLM Client & UI — Working Reference

End-to-end walkthrough of how `client.py` and `static/index.html` fit
together to form the NX → AGX messenger over LAN.

---

## 1. Architecture

```
┌──────────────────────────┐         ┌──────────────────────────┐
│  Jetson Orin NX          │         │  AGX Orin (server.py)    │
│                          │         │                          │
│  client.py               │  LAN    │  FastAPI app             │
│   ├─ REST helpers        │◀──────▶ │   ├─ GET  /health        │
│   ├─ WSCliClient (CLI)   │         │   ├─ GET  /              │
│   └─ open_browser_ui ────┼─opens──▶│   │   (serves static/    │
│                          │         │   │    index.html)        │
│  webbrowser ─────────────┼────────▶│   ├─ WS   /ws            │
│   └─ index.html (UI)     │◀────────│   ├─ POST /send         │
│       └─ WebSocket       │  ws://  │   └─ GET  /messages      │
└──────────────────────────┘         └──────────────────────────┘
```

The AGX hosts everything. The NX is a thin client:

- In **GUI mode**, `client.py` only opens a browser pointing at
  `http://<agx-ip>:<port>/?client_id=<name>`. The page itself runs the
  WebSocket and chat logic.
- In **CLI mode**, `client.py` runs a terminal REPL that talks to the
  server's WebSocket directly. The browser is never used.

---

## 2. `client.py`

File: `client.py:1` (394 lines). Runs on the NX.

### 2.1 Logging — `client.py:40`

```python
LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
```

- Logger name: `NX-CLIENT`
- Level: `DEBUG` (verbose — every WS frame, REST call, etc.)
- Two handlers: stdout and `client.log` (append mode)

### 2.2 REST helpers — `client.py:55`

Synchronous HTTP helpers built on `requests`. Used both for the
startup health check and as a fallback transport if WebSocket fails.

| Function            | Endpoint            | Purpose                                       |
|---------------------|---------------------|-----------------------------------------------|
| `check_health`      | `GET  /health`      | First reachability check at startup           |
| `rest_send`         | `POST /send`        | Send a chat message via HTTP (fallback)       |
| `rest_get_messages` | `GET  /messages?…`  | Pull recent history (used by REST poll loop)  |

### 2.3 `WSCliClient` — `client.py:93`

Threaded WebSocket client used in **CLI mode** (`--cli`). Internally
it wraps `websocket-client`'s `WebSocketApp`.

**State** (`client.py:99`):
- `self._ws` — the `WebSocketApp`
- `self._connected` (`threading.Event`) — set on `on_open`
- `self._stop` (`threading.Event`) — set on close / explicit stop
- `self._recv_thread` — background thread that runs `run_forever()`

**Callbacks** (`client.py:109`):

| Event           | Behaviour                                               |
|-----------------|---------------------------------------------------------|
| `on_open`       | Set `_connected`, log `[WS OPEN]`                       |
| `on_message`    | Parse JSON frame, dispatch by `type` (see protocol)     |
| `on_error`      | Log + print warning                                     |
| `on_close`      | Clear `_connected`, set `_stop`, print disconnect line  |

**Frame handling** (`client.py:121`):

| `type`        | Action                                                    |
|---------------|-----------------------------------------------------------|
| `connected`   | Print `✓ Connected as [<id>] on <node>`                   |
| `history`     | Print last 20 messages from the `payload`                |
| `message`     | Pretty-print a single message                            |
| `ack`         | Debug-log the message id the server confirmed            |
| `pong`        | Debug-log                                                |
| `error`       | Print server-side error message                          |

**Lifecycle** (`client.py:182`):

```
connect() ──► spawns _recv_thread (daemon) ──► waits up to 10s for _connected
                                                       │
                                                       ▼
                                  user types messages in run_cli() REPL
                                                       │
                                                       ▼
                                  pinger thread sends {"type":"ping"} every 15s
                                                       │
                                                       ▼
                                       Ctrl-C / /quit ──► close()
```

**`run_cli()`** (`client.py:222`) is a blocking REPL:

- Reads `input("[<name>] > ")` in a loop
- Sends each line as `{"type":"message","sender":name,"text":line}`
- Special commands: `/quit`, `/exit`, `/q` exit; `/ping` pings manually
- Starts a daemon pinger that sends `{"type":"ping"}` every 15s
- On `KeyboardInterrupt` or `/quit`, calls `close()`

### 2.4 GUI helper — `client.py:264`

```python
def open_browser_ui(base_url: str, client_id: str):
    url = f"{base_url}/?client_id={client_id}"
    webbrowser.open(url)
```

The server already serves `static/index.html` at `/`, so the only
thing the client does in GUI mode is open that URL with the
`client_id` query param so the page knows who is connecting.

### 2.5 `parse_args` — `client.py:280`

| Flag              | Required | Default      | Meaning                                |
|-------------------|----------|--------------|----------------------------------------|
| `--agx-ip`        | yes      | —            | AGX server IP on the LAN               |
| `--port`          | no       | `8000`       | Server port                            |
| `--name`          | no       | `nx-node`    | Identifier sent to the server          |
| `--cli`           | no       | off (GUI)    | Use terminal mode instead of browser   |

### 2.6 `main()` — `client.py:295`

```
1. Build base_url  = http://<agx-ip>:<port>
   Build ws_url    = ws://<agx-ip>:<port>/ws?client_id=<name>
2. Banner log block
3. check_health(base_url) ──► on fail, print hints and sys.exit(1)
4. Branch on --cli:
     CLI  : WSCliClient → connect() → run_cli() (or REST fallback)
     GUI  : open_browser_ui() → sleep loop until Ctrl-C
```

### 2.7 `_rest_cli_loop` — `client.py:340`

Fallback used when the WebSocket cannot connect. Behaviour:

- Polls `GET /messages?limit=50` every 3 seconds
- Maintains a `seen_ids` set so each message is printed exactly once
- Sends via `POST /send` when the user types a line
- Same REPL commands (`/quit`, `/q`, …)

---

## 3. `static/index.html`

File: `static/index.html:1` (1006 lines). Served by the AGX at `/`.
Pure HTML + CSS + vanilla JS — no build step, no framework.

### 3.1 Page structure

```
#connect-modal   — overlay shown until "Connect" is pressed (or
                   auto-bypassed when ?client_id=… is in the URL)
#reconnect-bar   — orange banner that appears when the WS drops
#app
  ├─ <header>        — logo, title, status chips (WS, SERVER), node id
  ├─ #messages-wrap  — chat history scroll area
  └─ #input-bar      — sender name, role select, textarea, SEND button
```

### 3.2 Theme / CSS — `index.html:9`

A "sci-fi terminal" look defined in `:root` custom properties:

- **Palette** — `--bg #080c10`, `--accent #00c8ff` (NX / cyan),
  `--accent2 #00ff9d` (AGX / green), `--warn #ff6b35`
- **Fonts** — `Share Tech Mono` for code/labels, `Rajdhani` for body
- **Effects** — scanline overlay (`body::before`), corner brackets
  on the message wrap, glowing borders on focus, blinking status dots
- **Layout** — CSS grid `#app { grid-template-rows: 56px 1fr auto }`

Message bubbles are colour-coded by role:

| Class        | Alignment | Border accent      | Used for            |
|--------------|-----------|--------------------|---------------------|
| `from-agx`   | left      | left  2px green    | server-side replies |
| `from-nx`    | right     | right 2px cyan     | client-side sends   |
| `from-system`| centred   | none, dim mono     | system info lines   |

### 3.3 JS state — `index.html:652`

```js
let ws, wsUrl, baseUrl;
let clientName = 'unknown';
let clientRole = 'nx';          // 'nx' | 'agx'
let reconnectTimer = null;
let pingTimer      = null;
let seenIds        = new Set(); // dedupe incoming messages
let isConnected    = false;
```

### 3.4 Init from URL — `index.html:687`

```js
(function initFromUrl() {
  const p = new URLSearchParams(window.location.search);
  cfHost.value = window.location.hostname;
  cfPort.value = window.location.port || '8000';
  if (p.get('client_id')) cfName.value = p.get('client_id');
  if (p.get('role'))      cfRole.value  = p.get('role');
  if (p.get('client_id') && window.location.hostname)
    setTimeout(doConnect, 300);   // skip modal when ?client_id=…
})();
```

When `client.py` opens the browser with `?client_id=…`, the page
auto-connects after 300 ms and never shows the modal. Manual users
fill the form and click **Connect →**.

### 3.5 Connect logic — `index.html:714`

```js
function doConnect() {
  const host = cfHost.value.trim() || window.location.hostname;
  const port = cfPort.value.trim() || '8000';
  clientName = cfName.value.trim() || 'unknown';
  clientRole = cfRole.value;
  baseUrl = `http://${host}:${port}`;
  wsUrl   = `ws://${host}:${port}/ws?client_id=${encodeURIComponent(clientName)}`;
  modal.classList.add('hidden');
  openWS();
}
```

### 3.6 WebSocket lifecycle — `index.html:734`

```
openWS()
  ├─ on open   → setWsStatus('online'), enable SEND, startPing()
  ├─ on message→ handleFrame()
  ├─ on close  → setWsStatus('offline'), disable SEND, scheduleReconnect()
  └─ on error  → setWsStatus('offline')
```

**Ping / keepalive** (`index.html:792`):

```js
pingTimer = setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({type:'ping'}));
}, 15000);
```

**Reconnect** (`index.html:783`):

```js
function scheduleReconnect() {
  reconnBar.classList.add('visible');
  reconnectTimer = setTimeout(openWS, 3000);
}
```

3-second backoff, banner visible while disconnected.

### 3.7 Frame handler — `index.html:807`

Mirrors the CLI client's `on_message`, with one extra: `history`.

| `type`        | UI action                                                  |
|---------------|------------------------------------------------------------|
| `history`     | Render every message in `payload` then scroll to bottom   |
| `connected`   | Mark server chip as online, log the assigned `client_id`  |
| `message`     | Dedupe by `msg.id`, call `renderMessage`, scroll           |
| `ack`         | Log the confirmed id                                       |
| `pong`        | No-op                                                      |
| `error`       | Log `frame.detail`                                         |

### 3.8 REST fallback — `index.html:852`

```js
function sendFrame(frame) {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify(frame));
  else
    restSend(frame.sender, frame.text);
}
```

```js
async function restSend(sender, text) {
  await fetch(`${baseUrl}/send`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ sender, text }),
  });
}
```

If the WS is not open, the UI silently falls back to `POST /send`.

### 3.9 Render — `index.html:870`

```js
function renderMessage(msg) {
  // system messages → centered, dim, no meta
  // role === 'server' → from-agx (left, green)
  // role === 'user'   → from-nx  (right, cyan)
  // own message       → adds .own class for extra glow
}
```

The bubble is built as:

```html
<div class="msg-row from-… [own]">
  <div class="msg-meta">
    <span class="msg-sender">▶ AGX-ORIN</span>     <!-- or ◀ <name> -->
    <span>HH:MM:SS</span>
  </div>
  <div class="msg-bubble">…text…</div>
</div>
```

All text is HTML-escaped via `escHtml()` (`index.html:989`).

### 3.10 Sending — `index.html:935`

```js
function sendMessage() {
  const text = msgInput.value.trim();
  // build a temp message, render it optimistically
  // mark its temp id in seenIds so the server echo is deduped
  // sendFrame({ type:'message', sender, text });
  msgInput.value = '';
}
```

`Ctrl+Enter` triggers `sendMessage` from the textarea
(`index.html:919`); the SEND button does the same. The textarea
auto-grows up to 120 px.

### 3.11 Status chips — `index.html:969`

```js
setWsStatus('online')  → chip.online,  label "WS ✓"
setWsStatus('offline') → chip.offline, label "WS ✗"
setWsStatus('connecting')            label "WS …"
```

Same trio for the `SERVER` chip (driven by `connected` frames).

---

## 4. Wire protocol

JSON frames only. All messages travel on the single WebSocket
`/ws?client_id=<name>` endpoint.

### Client → server

```json
{ "type": "ping" }
{ "type": "message", "sender": "nx-node-1", "text": "hello" }
```

### Server → client

```json
{ "type": "connected",  "client_id": "nx-node-1", "node": "agx" }
{ "type": "history",     "payload": [ { /* message */ }, … ] }
{ "type": "message",     "payload": { "id": "…", "sender": "…", "role": "server|user|system", "text": "…", "timestamp": "ISO8601" } }
{ "type": "ack",         "id": "…" }
{ "type": "pong" }
{ "type": "error",       "detail": "…" }
```

### REST surface (used as fallback / health)

| Verb | Path        | Body / Query                                |
|------|-------------|---------------------------------------------|
| GET  | `/health`   | —                                           |
| GET  | `/`         | serves `static/index.html`                  |
| GET  | `/messages` | `?limit=N` → `{ "messages": [ … ] }`        |
| POST | `/send`     | `{ "sender": "…", "text": "…" }`            |

---

## 5. End-to-end flow

### GUI mode (default)

```
$ python client.py --agx-ip 172.16.6.21
  │
  ├─ check_health() ──► GET /health ──► ✓ server online
  ├─ open_browser_ui()
  │     └─ webbrowser.open("http://172.16.6.21:8000/?client_id=nx-node")
  │              │
  │              ▼
  │   index.html loads
  │     ├─ initFromUrl() reads client_id
  │     ├─ doConnect()  → opens ws://…/ws?client_id=nx-node
  │     ├─ server sends {type:'connected'}
  │     ├─ server sends {type:'history', payload:[…]}
  │     └─ user types → sendMessage() → ws.send({type:'message', …})
  └─ main() sleeps in a 60s loop until Ctrl-C
```

### CLI mode (`--cli`)

```
$ python client.py --agx-ip 172.16.6.21 --cli
  │
  ├─ check_health() ✓
  ├─ WSCliClient.connect(timeout=10)
  │     └─ background thread runs run_forever(ping_interval=20, ping_timeout=10)
  ├─ pinger thread (15s) sends {"type":"ping"}
  └─ REPL: input("[nx-node] > ") → send() → ws.send({type:'message',…})
                                            │
        on_message() ←─────────────────────┘
        prints e.g. "  10:21:33  [AGX]  agx: hi back"
```

### Failure paths

| Symptom                              | Where caught                            |
|--------------------------------------|-----------------------------------------|
| AGX unreachable                      | `check_health` → `sys.exit(1)` with hints (`client.py:310`) |
| WS connect fails in CLI mode         | `WSCliClient.connect` returns False → `_rest_cli_loop` (`client.py:325`) |
| WS drops mid-session in browser      | `onclose` → `scheduleReconnect` (3 s backoff, banner shown) |
| WS down when sending from browser    | `sendFrame` falls through to `restSend` (`index.html:773`) |

---

## 6. File map

```
client.py                NX-side launcher / CLI client
static/index.html        Browser chat UI (served by AGX)
client.log               Append-only log written by client.py
```

Logs are written by `client.py` only — `index.html` logs to
`console.log` in the browser DevTools (the helper at
`index.html:984` prefixes `[HH:MM:SS.mmm] [TAG]`).
