#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-funasr-api-cpu}"
CERT_DIR="${CERT_DIR:-$(pwd)/temp/https-hotfix}"
NGINX_CERT_DIR="/etc/nginx/certs"
NGINX_CONF_PATH="/tmp/funasr-internal-nginx.conf"

detect_lan_ip() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    local default_if
    default_if="$(route -n get default 2>/dev/null | awk '/interface:/{print $2; exit}')"
    if [[ -n "$default_if" ]]; then
      ipconfig getifaddr "$default_if" 2>/dev/null || true
      return 0
    fi
  fi

  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | awk '{print $1}' || true
    return 0
  fi

  return 1
}

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found"
  exit 1
fi

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl not found"
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "container not running: $CONTAINER_NAME"
  exit 1
fi

LAN_IP="${LAN_IP:-$(detect_lan_ip)}"
if [[ -z "$LAN_IP" ]]; then
  echo "failed to detect LAN IP, please run with: LAN_IP=<your-ip> $0 $CONTAINER_NAME"
  exit 1
fi

HOST_PORT="${HOST_PORT:-$(docker port "$CONTAINER_NAME" 8000/tcp | awk -F: 'NR==1 {print $NF}') }"
HOST_PORT="${HOST_PORT//[[:space:]]/}"
if [[ -z "$HOST_PORT" ]]; then
  echo "failed to detect published host port for container port 8000"
  exit 1
fi

mkdir -p "$CERT_DIR"

CERT_FILE="$CERT_DIR/server.crt"
KEY_FILE="$CERT_DIR/server.key"
LOCAL_CONF="$CERT_DIR/funasr-internal-nginx.conf"

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout "$KEY_FILE" \
  -out "$CERT_FILE" \
  -subj "/CN=${LAN_IP}" \
  -addext "subjectAltName=IP:${LAN_IP}" \
  >/dev/null 2>&1

docker exec "$CONTAINER_NAME" mkdir -p "$NGINX_CERT_DIR"
docker cp "$CERT_FILE" "$CONTAINER_NAME:$NGINX_CERT_DIR/server.crt"
docker cp "$KEY_FILE" "$CONTAINER_NAME:$NGINX_CERT_DIR/server.key"
docker cp "$CONTAINER_NAME:$NGINX_CONF_PATH" "$LOCAL_CONF"

awk '
  BEGIN {
    in_server = 0
    inserted_ssl = 0
  }
  /^[[:space:]]*server[[:space:]]*\{/ && in_server == 0 {
    in_server = 1
  }
  in_server == 1 && /^[[:space:]]*ssl_certificate[[:space:]]+/ { next }
  in_server == 1 && /^[[:space:]]*ssl_certificate_key[[:space:]]+/ { next }
  in_server == 1 && /^[[:space:]]*ssl_protocols[[:space:]]+/ { next }
  in_server == 1 && /^[[:space:]]*listen[[:space:]]+[0-9]+;[[:space:]]*$/ {
    sub(/;[[:space:]]*$/, " ssl;")
    print
    next
  }
  in_server == 1 && /^[[:space:]]*listen[[:space:]]+[0-9]+[[:space:]]+ssl;[[:space:]]*$/ {
    print
    next
  }
  in_server == 1 && /^[[:space:]]*server_name[[:space:]]+_[[:space:]]*;[[:space:]]*$/ {
    print
    print "        ssl_certificate /etc/nginx/certs/server.crt;"
    print "        ssl_certificate_key /etc/nginx/certs/server.key;"
    print "        ssl_protocols TLSv1.2 TLSv1.3;"
    inserted_ssl = 1
    next
  }
  in_server == 1 && /proxy_set_header X-Forwarded-Proto \$scheme;/ {
    sub(/\$scheme/, "https")
    print
    next
  }
  in_server == 1 && /^[[:space:]]*}\s*$/ {
    in_server = 0
  }
  {
    print
  }
' "$LOCAL_CONF" > "$LOCAL_CONF.tmp"

mv "$LOCAL_CONF.tmp" "$LOCAL_CONF"

docker cp "$LOCAL_CONF" "$CONTAINER_NAME:$NGINX_CONF_PATH"
docker exec "$CONTAINER_NAME" nginx -t -c "$NGINX_CONF_PATH"
docker exec "$CONTAINER_NAME" sh -c 'kill -HUP "$(cat /run/nginx.pid)"'

echo
echo "HTTPS enabled for current container runtime."
echo "Open: https://${LAN_IP}:${HOST_PORT}/ws/v1/asr/test"
echo "Certificate: $CERT_FILE"
echo
echo "Note: trust the generated certificate on the client device, or the browser may still block microphone access."
