# HiSLM Remote Desktop — GUI Access from MacBook

Full remote desktop control of Jetson Orin NX with Gnome GUI over the internet.

## What you get

| File | On | Purpose |
|------|----|---------|
| `hislm-remote.pem` | Your MacBook | SSH private key (PEM format) for password-less auth |
| `remote-desktop.sh` | Your MacBook | One-click launcher for RDP GUI access |
| `setup-remote-desktop.sh` | NX Jetson | Install xRDP server (run once with sudo) |

---

## Two Ways to Connect

### Way 1: Tailscale (recommended — fastest, most secure)

Requires Tailscale on your MacBook. Gives a direct encrypted connection.

```bash
# On MacBook — one-time install
./remote-desktop.sh install
./remote-desktop.sh copy-key

# On NX Jetson — one-time setup (requires sudo)
chmod +x setup-remote-desktop.sh
./setup-remote-desktop.sh

# On MacBook — connect anytime
./remote-desktop.sh
```

This opens an SSH tunnel (localhost:33389 → NX:3389) plus Microsoft Remote Desktop.

### Way 2: ngrok (no Tailscale needed — public URL)

Gives you a public `0.tcp.ngrok.io:xxxxx` URL accessible from anywhere.

```bash
# Step 1: Get an ngrok auth token
# Sign up at https://dashboard.ngrok.com → get your token

# Step 2: Set it on the NX Jetson
ssh nvidia@100.85.30.17 "ngrok config add-authtoken YOUR_TOKEN"

# Step 3: Launch from MacBook
./remote-desktop.sh ngrok
```

---

## PEM Key Details

- **Algorithm:** RSA 4096-bit (PEM format)
- **Location on NX:** `~/llama/HiSLM-8G/hislm-remote.pem`
- **Location on MacBook:** Download to same dir as `remote-desktop.sh`
- **Public key** is already in `~/.ssh/authorized_keys` on the NX

To manually download the PEM key to your MacBook:
```bash
scp nvidia@100.85.30.17:~/llama/HiSLM-8G/hislm-remote.pem .
chmod 600 hislm-remote.pem
```

---

## If Microsoft Remote Desktop is not installed

```bash
brew install --cask microsoft-remote-desktop
```

Or use any RDP client — connect to `localhost:33389`, user `nvidia`, with the NX password.

---

## Manual Tunnel (if you prefer Terminal only)

```bash
# Create the tunnel
ssh -i hislm-remote.pem \
  -L 33389:localhost:3389 \
  -o StrictHostKeyChecking=accept-new \
  nvidia@100.85.30.17

# Then in Microsoft Remote Desktop → Add PC → localhost:33389
```

---

## Network Details

| Endpoint | Type | Notes |
|----------|------|-------|
| `100.85.30.17` | Tailscale IP | Requires Tailscale on MacBook |
| `103.15.228.94` | Public IP | Router-level, may need port forwarding |
| `0.tcp.ngrok.io:xxxxx` | ngrok TCP | Public URL, no Tailscale needed |
| `172.16.6.28` | LAN IP | Local network only |
