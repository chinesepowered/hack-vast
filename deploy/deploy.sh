#!/usr/bin/env bash
# Deploy Edge-Case Miner into the team's Kubernetes namespace. Run this ON the workshop VM.
#
# No image build or registry: a public python:3.12-slim image runs the code from a ConfigMap,
# credentials come from a Secret, and an Ingress serves it at http://<team-host>/app
# (the Ingress strips the /app prefix, so the app's own routes stay at /).
#
#   bash deploy/deploy.sh
#   APP_NAME=my-miner bash deploy/deploy.sh          # different resource names
#   KUBECTL=/path/to/kubectl bash deploy/deploy.sh   # kubectl not on PATH
#
# Secret values are never printed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$REPO_ROOT/app"
APP_NAME="${APP_NAME:-edge-case-miner}"
APP_PORT=8080
KUBECTL="${KUBECTL:-kubectl}"

die() { echo "ERROR: $*" >&2; exit 1; }

command -v "$KUBECTL" >/dev/null 2>&1 || die "kubectl not found (install it or set KUBECTL=/path/to/kubectl)"
[[ -f "$APP_DIR/main.py" ]] || die "app code not found at $APP_DIR"

# --- kubeconfig: $KUBECONFIG if it names a file, else /config/kubeconfig, else the single /config/*-k8s.yaml
if [[ -z "${KUBECONFIG:-}" || ! -f "${KUBECONFIG}" ]]; then
  if [[ -f /config/kubeconfig ]]; then
    export KUBECONFIG=/config/kubeconfig
  else
    mapfile -t KUBECONFIGS < <(find /config -maxdepth 1 \( -type f -o -type l \) -name '*-k8s.yaml' 2>/dev/null | sort)
    (( ${#KUBECONFIGS[@]} == 1 )) || die "expected /config/kubeconfig or exactly one /config/*-k8s.yaml"
    export KUBECONFIG="${KUBECONFIGS[0]}"
  fi
fi

# --- team config: credentials, INGRESS_URL, USERNAME (sourced, never echoed)
mapfile -t TEAM_CONFIGS < <(find /config -maxdepth 1 \( -type f -o -type l \) -name '*.config' 2>/dev/null | sort)
if (( ${#TEAM_CONFIGS[@]} == 1 )); then
  set +u
  set -a
  # shellcheck disable=SC1090
  source "${TEAM_CONFIGS[0]}"
  set +a
  set -u
elif (( ${#TEAM_CONFIGS[@]} > 1 )); then
  die "expected exactly one /config/*.config, found ${#TEAM_CONFIGS[@]}"
else
  echo "WARN: no /config/*.config found; using the current environment" >&2
fi

[[ -n "${USERNAME:-}" ]] || die "USERNAME (the team namespace) is not set"
[[ -n "${INGRESS_URL:-}" ]] || die "INGRESS_URL is not set"
NS="$USERNAME"
APP_HOST="${INGRESS_URL#http://}"
APP_HOST="${APP_HOST#https://}"
APP_HOST="${APP_HOST%%/*}"
[[ -n "$APP_HOST" ]] || die "could not derive the team host from INGRESS_URL"

# --- path: /app, unless another app in this (possibly shared) namespace already serves /app on this host
if [[ -z "${APP_PATH:-}" ]]; then
  APP_PATH=/app
  taken="$("$KUBECTL" -n "$NS" get ingress -o json 2>/dev/null | python3 -c '
import json, sys
host, mine = sys.argv[1], sys.argv[2]
for item in json.load(sys.stdin).get("items", []):
    name = item["metadata"]["name"]
    for rule in item.get("spec", {}).get("rules", []):
        for p in rule.get("http", {}).get("paths", []):
            if name != mine and rule.get("host") == host and (p.get("path") or "").startswith("/app"):
                print(name)
' "$APP_HOST" "$APP_NAME" | sort -u | tr '\n' ' ' || true)"
  if [[ -n "$taken" ]]; then
    APP_PATH=/edge-case-miner
    echo "WARN: /app on $APP_HOST is already taken by: ${taken}— using $APP_PATH instead" >&2
  fi
fi

# --- the ConfigMap holds app/ (flat: --from-file does not recurse) and must stay under ~1 MiB
APP_BYTES="$(find "$APP_DIR" -maxdepth 1 -type f -printf '%s\n' | awk '{s += $1} END {print s + 0}')"
(( APP_BYTES < 950000 )) || die "app/ is ${APP_BYTES} bytes; a ConfigMap must stay under 1 MiB"

echo "== deploying $APP_NAME to namespace $NS (host $APP_HOST, code ${APP_BYTES} bytes)"

# 1. Code -> ConfigMap. Server-side apply avoids the 256 KiB last-applied annotation limit.
"$KUBECTL" -n "$NS" create configmap "${APP_NAME}-code" --from-file="$APP_DIR" --dry-run=client -o yaml \
  | "$KUBECTL" -n "$NS" apply --server-side --force-conflicts -f -

# 2. Credentials -> Secret. Unset values become empty strings; the app degrades gracefully without them.
empty=()
for var in INGRESS_URL USERNAME PASSWORD GPU_BEARER_TOKEN COSMOS3_REASON_URL COSMOS3_REASON_MODEL \
           WANDB_API_KEY WANDB_TEAM WANDB_PROJECT; do
  [[ -n "${!var:-}" ]] || empty+=("$var")
done
if (( ${#empty[@]} )); then
  echo "WARN: empty in the app Secret: ${empty[*]}" >&2
fi
# Values go through a private temp file, not --from-literal: on a shared VM, command lines are
# visible to every user via ps.
ENV_FILE="$(mktemp)"
chmod 600 "$ENV_FILE"
trap 'rm -f "$ENV_FILE"' EXIT
{
  printf 'VSS_URL=%s\n' "${INGRESS_URL:-}"
  printf 'VSS_USERNAME=%s\n' "${USERNAME:-}"
  printf 'VSS_PASSWORD=%s\n' "${PASSWORD:-}"
  printf 'GPU_BEARER_TOKEN=%s\n' "${GPU_BEARER_TOKEN:-}"
  printf 'COSMOS3_REASON_URL=%s\n' "${COSMOS3_REASON_URL:-}"
  printf 'COSMOS3_REASON_MODEL=%s\n' "${COSMOS3_REASON_MODEL:-}"
  printf 'WANDB_API_KEY=%s\n' "${WANDB_API_KEY:-}"
  printf 'WANDB_TEAM=%s\n' "${WANDB_TEAM:-}"
  printf 'WANDB_PROJECT=%s\n' "${WANDB_PROJECT:-}"
} > "$ENV_FILE"
"$KUBECTL" -n "$NS" create secret generic "${APP_NAME}-env" --from-env-file="$ENV_FILE" \
  --dry-run=client -o yaml \
  | "$KUBECTL" -n "$NS" apply --server-side --force-conflicts -f -

# 3. Deployment + Service + Ingress (path /app on the team's existing host).
"$KUBECTL" -n "$NS" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP_NAME}
  labels:
    app: ${APP_NAME}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${APP_NAME}
  template:
    metadata:
      labels:
        app: ${APP_NAME}
    spec:
      containers:
      - name: app
        image: python:3.12-slim
        imagePullPolicy: IfNotPresent
        workingDir: /code
        command: ["bash", "-c"]
        args:
        - |
          pip install --no-cache-dir -q -r requirements.txt \\
            && (pip install --no-cache-dir -q -r requirements-optional.txt || echo "optional deps skipped") \\
            && exec python main.py
        ports:
        - containerPort: ${APP_PORT}
        env:
        - name: PORT
          value: "${APP_PORT}"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: PYTHONDONTWRITEBYTECODE
          value: "1"
        - name: PIP_DISABLE_PIP_VERSION_CHECK
          value: "1"
        - name: PIP_ROOT_USER_ACTION
          value: "ignore"
        envFrom:
        - secretRef:
            name: ${APP_NAME}-env
        volumeMounts:
        - name: code
          mountPath: /code
        readinessProbe:
          httpGet:
            path: /health
            port: ${APP_PORT}
          initialDelaySeconds: 20
          periodSeconds: 10
          failureThreshold: 30
      volumes:
      - name: code
        configMap:
          name: ${APP_NAME}-code
---
apiVersion: v1
kind: Service
metadata:
  name: ${APP_NAME}
  labels:
    app: ${APP_NAME}
spec:
  type: ClusterIP
  selector:
    app: ${APP_NAME}
  ports:
  - name: http
    port: 80
    targetPort: ${APP_PORT}
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${APP_NAME}
  labels:
    app: ${APP_NAME}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-body-size: "8m"
spec:
  ingressClassName: nginx
  rules:
  - host: ${APP_HOST}
    http:
      paths:
      - path: ${APP_PATH}(/|\$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: ${APP_NAME}
            port:
              number: 80
EOF

# 4. Restart so ConfigMap / Secret changes are picked up, then wait for the new pod.
"$KUBECTL" -n "$NS" rollout restart deploy/"$APP_NAME"
"$KUBECTL" -n "$NS" rollout status deploy/"$APP_NAME" --timeout=600s

APP_URL="http://${APP_HOST}${APP_PATH}"
echo "$APP_URL" > "$REPO_ROOT/.app-url" 2>/dev/null || true

# 5. Public HTTPS link via a Cloudflare quick tunnel (no account needed). It points at this app's
#    Service only, never at the VSS backend. Re-applying an unchanged spec doesn't restart the
#    tunnel pod, so the random *.trycloudflare.com URL survives app redeploys. TUNNEL=0 skips it.
TUNNEL_URL=""
if [[ "${TUNNEL:-1}" == "1" ]]; then
  echo "== public tunnel"
  "$KUBECTL" -n "$NS" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP_NAME}-tunnel
  labels:
    app: ${APP_NAME}-tunnel
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${APP_NAME}-tunnel
  template:
    metadata:
      labels:
        app: ${APP_NAME}-tunnel
    spec:
      containers:
      - name: cloudflared
        image: cloudflare/cloudflared:latest
        args: ["tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://${APP_NAME}:80"]
EOF
  "$KUBECTL" -n "$NS" rollout status deploy/"${APP_NAME}-tunnel" --timeout=300s || true
  for _ in $(seq 1 30); do
    TUNNEL_URL="$("$KUBECTL" -n "$NS" logs deploy/"${APP_NAME}-tunnel" 2>/dev/null \
      | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1 || true)"
    [[ -n "$TUNNEL_URL" ]] && break
    sleep 4
  done
  if [[ -n "$TUNNEL_URL" ]]; then
    echo "$TUNNEL_URL" > "$REPO_ROOT/.tunnel-url" 2>/dev/null || true
  else
    rm -f "$REPO_ROOT/.tunnel-url"
    echo "WARN: no tunnel URL yet; check: $KUBECTL -n $NS logs deploy/${APP_NAME}-tunnel" >&2
  fi
fi

echo
echo "== Edge-Case Miner is live: $APP_URL"
[[ -n "$TUNNEL_URL" ]] && echo "   public:  $TUNNEL_URL   (share this one: works outside the event network)"
echo "   health:  $APP_URL/health"
echo "   logs:    $KUBECTL -n $NS logs deploy/$APP_NAME"
echo "   warm-up: $KUBECTL -n $NS exec deploy/$APP_NAME -- python main.py --warm   (or 'Warm cache' in the Coverage tab)"
