#!/bin/bash
# HiSLM Remote Desktop Setup — Run this ON THE NX TERMINAL
# Sets up xRDP + ngrok TCP tunnel for external GUI access.
#
# Usage:
#   ./setup-remote-desktop.sh
#   (will prompt for sudo password once)

set -e

echo "╔══════════════════════════════════════════╗"
echo "║   HiSLM — Remote Desktop Setup           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Install xRDP ──────────────────────────────
echo "▶ Installing xRDP (RDP server)..."
sudo apt-get update -qq
sudo apt-get install -y xrdp
sudo systemctl enable xrdp
sudo systemctl start xrdp
echo "  ✓ xRDP installed and running on port 3389"

# ── 2. Fix color depth for better performance ─────
echo "▶ Configuring xRDP..."
sudo sed -i 's/max_bpp=32/max_bpp=24/' /etc/xrdp/xrdp.ini 2>/dev/null || true
sudo sed -i 's/use_compression=yes/use_compression=yes/' /etc/xrdp/xrdp.ini 2>/dev/null || true

# ── 3. Configure xRDP to use the existing desktop ─
echo "▶ Configuring xRDP session..."
cat << 'XSESSION' | sudo tee /etc/xrdp/startwm.sh > /dev/null
#!/bin/sh
# xRDP start script — use existing Gnome/Ubuntu desktop
if [ -r /etc/default/locale ]; then
  . /etc/default/locale
  export LANG LANGUAGE
fi
export GNOME_SHELL_SESSION_MODE=ubuntu
export XDG_CURRENT_DESKTOP=ubuntu:GNOME
export XDG_CONFIG_DIRS=/etc/xdg/xdg-ubuntu:/etc/xdg
. /usr/share/gnome/gnome-session-shutdown
for f in /etc/profile.d/*.sh; do [ -r "$f" ] && . "$f"; done
unset DBUS_SESSION_BUS_ADDRESS
mate-session 2>/dev/null || xfce4-session 2>/dev/null || gnome-session
XSESSION
sudo chmod +x /etc/xrdp/startwm.sh

# ── 4. Add user to ssl-cert group ──────────────────
echo "▶ Adding user to ssl-cert group..."
sudo adduser "$USER" ssl-cert 2>/dev/null || true

# ── 5. Restart xRDP ────────────────────────────────
sudo systemctl restart xrdp
echo "  ✓ xRDP restarted"

# ── 6. Firewall ────────────────────────────────────
echo "▶ Checking firewall..."
if sudo ufw status | grep -q active; then
  sudo ufw allow 3389/tcp
  echo "  ✓ Port 3389 opened in firewall"
else
  echo "  - Firewall inactive (no change needed)"
fi

# ── 7. Test ─────────────────────────────────────────
echo "▶ Testing xRDP..."
if ss -tlnp | grep -q :3389; then
  echo "  ✓ xRDP listening on port 3389"
else
  echo "  ✗ xRDP failed to start. Check: sudo journalctl -u xrdp"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Setup complete on NX side!              ║"
echo "╠══════════════════════════════════════════╣"
echo "║                                          ║"
echo "║  Now on your MacBook:                    ║"
echo "║                                          ║"
echo "║  1. Download the PEM key:                ║"
echo "║     scp nvidia@100.85.30.17:~/llama/    ║"
echo "║       HiSLM-8G/hislm-remote.pem .       ║"
echo "║     chmod 600 hislm-remote.pem           ║"
echo "║                                          ║"
echo "║  2. Download the launcher:               ║"
echo "║     scp nvidia@100.85.30.17:~/llama/    ║"
echo "║       HiSLM-8G/scripts/remote-desktop.sh .║"
echo "║     chmod +x remote-desktop.sh           ║"
echo "║                                          ║"
echo "║  3. Run the launcher:                    ║"
echo "║     ./remote-desktop.sh                  ║"
echo "║                                          ║"
echo "╚══════════════════════════════════════════╝"
