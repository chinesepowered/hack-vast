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
"$KUBECTL" -n "$NS" create secret generic "${APP_NAME}-env" \
  --from-literal=VSS_URL="${INGRESS_URL:-}" \
  --from-literal=VSS_USERNAME="${USERNAME:-}" \
  --from-literal=VSS_PASSWORD="${PASSWORD:-}" \
  --from-literal=GPU_BEARER_TOKEN="${GPU_BEARER_TOKEN:-}" \
  --from-literal=COSMOS3_REASON_URL="${COSMOS3_REASON_URL:-}" \
  --from-literal=COSMOS3_REASON_MODEL="${COSMOS3_REASON_MODEL:-}" \
  --from-literal=WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --from-literal=WANDB_TEAM="${WANDB_TEAM:-}" \
  --from-literal=WANDB_PROJECT="${WANDB_PROJECT:-}" \
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
      - path: /app(/|\$)(.*)
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

APP_URL="http://${APP_HOST}/app"
echo "$APP_URL" > "$REPO_ROOT/.app-url" 2>/dev/null || true
echo
echo "== Edge-Case Miner is live: $APP_URL"
echo "   health:  $APP_URL/health"
echo "   logs:    $KUBECTL -n $NS logs deploy/$APP_NAME"
echo "   warm-up: $KUBECTL -n $NS exec deploy/$APP_NAME -- python main.py --warm   (or 'Warm cache' in the Coverage tab)"
