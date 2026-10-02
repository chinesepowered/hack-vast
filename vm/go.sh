#!/usr/bin/env bash
# One command on the workshop VM: update the code, check every service, deploy, print the URL.
#
#   bash vm/go.sh          # preflight + deploy to Kubernetes (the real demo URL)
#   bash vm/go.sh check    # preflight only
#   bash vm/go.sh local    # fallback: run the app on this VM and open it in the VM's browser
#   bash vm/go.sh warm     # precompute the demo (coverage grid + verdicts) inside the pod
#   bash vm/go.sh url      # print the in-event and public (Cloudflare tunnel) links
#   bash vm/go.sh ui       # push web-page-only changes (index.html) without restarting: keeps the warm cache
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${GO_UPDATED:-}" ]]; then
  echo "== updating code"
  git pull --ff-only -q 2>/dev/null || echo "   (git pull skipped)"
  # bash keeps reading the old copy of a script that changed under it: restart on the new one.
  GO_UPDATED=1 exec bash "$0" "$@"
fi
source vm/env.sh

mode="${1:-deploy}"
case "$mode" in
  warm|url|ui) ;;
  *) python3 vm/preflight.py ;;
esac

show_status() {  # show_status <url>: what the running app can reach (VSS, Cosmos, W&B)
  sleep 5  # the app probes its integrations right after start
  curl -s -m 30 "$1/api/config" | python3 -c '
import json, sys
d = json.load(sys.stdin)
mark = lambda ok: "✅" if ok else ("…" if ok is None else "❌")
v, c, l = d.get("vss") or {}, d.get("cosmos") or {}, d.get("llm") or {}
print("   VSS backend:    %s %s" % (mark(v.get("ok")), v.get("error") or ""))
print("   Cosmos3-Reason: %s %s" % (mark(bool(c.get("model"))), c.get("model") or c.get("error") or ""))
print("   W&B models:     %s %s / %s" % (mark(bool(l.get("model"))), l.get("model"), l.get("judge_model")))
print("   Weave tracing:  %s   W&B Artifacts: %s" % (mark(d.get("weave")), mark(d.get("wandb"))))
' 2>/dev/null || echo "   (could not read $1/api/config)"
}

wait_healthy() {  # wait_healthy <label> <url>: poll <url>/health for up to ~2 minutes
  local code=""
  echo "== waiting for $2/health"
  for _ in $(seq 1 24); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "$2/health" || true)"
    [[ "$code" == "200" ]] && { echo "✅ $1: $2"; return 0; }
    sleep 5
  done
  echo "⚠️  $2 not healthy yet (last HTTP $code). Logs: $KUBECTL -n $USERNAME logs deploy/$APP_NAME"
  return 1
}

case "$mode" in
  check)
    ;;
  local)
    # The pool VMs ship python3 without pip: use a user-local uv to build a venv.
    if [[ ! -x .venv/bin/python ]]; then
      if [[ ! -x .bin/uv ]]; then
        echo "== installing uv into .bin (one time)"
        mkdir -p .bin
        case "$(uname -m)" in aarch64|arm64) uv_arch=aarch64 ;; *) uv_arch=x86_64 ;; esac
        curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-${uv_arch}-unknown-linux-gnu.tar.gz" \
          | tar xz -C .bin --strip-components=1 || { echo "❌ uv download failed"; exit 1; }
      fi
      .bin/uv venv -q --python python3 .venv
    fi
    .bin/uv pip install -q --python .venv/bin/python -r app/requirements.txt || exit 1
    .bin/uv pip install -q --python .venv/bin/python -r app/requirements-optional.txt \
      || echo "   (optional deps skipped)"
    # Shared VM: take any free port instead of assuming 8080 is ours.
    export PORT="${PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"
    echo
    echo "== open http://localhost:$PORT in the VM's browser (Ctrl+C here stops the app)"
    cd app && exec ../.venv/bin/python main.py
    ;;
  warm)
    echo "== precomputing the demo inside the pod (takes a few minutes)"
    "$KUBECTL" -n "$USERNAME" exec deploy/"$APP_NAME" -- python main.py --warm
    ;;
  ui)
    # index.html is read on every request, so updating the code ConfigMap is enough; kubelet syncs the
    # mounted file within about a minute. No restart, so the app's warm cache survives.
    "$KUBECTL" -n "$USERNAME" create configmap "${APP_NAME}-code" --from-file=app --dry-run=client -o yaml \
      | "$KUBECTL" -n "$USERNAME" apply --server-side --force-conflicts -f -
    echo "== page updated in the cluster; it goes live within ~1 minute (no restart, cache kept)"
    ;;
  url)
    turl="$("$KUBECTL" -n "$USERNAME" logs deploy/"$APP_NAME-tunnel" 2>/dev/null \
      | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1)"
    echo "in-event: $(cat .app-url 2>/dev/null || echo unknown)"
    echo "public:   ${turl:-none (run: bash vm/go.sh)}"
    ;;
  *)
    # vm/READY is committed only once the app has been tested, so a half-built snapshot never deploys.
    if [[ ! -f vm/READY || ! -f deploy/deploy.sh ]]; then
      echo
      echo ">>> The app isn't ready yet. Send the report above to Claude, then re-run this command later."
      exit 0
    fi
    if [[ -z "$KUBECTL" || -z "$KUBECONFIG" ]]; then
      echo
      echo ">>> No Kubernetes access from this VM. Run the app here instead: bash vm/go.sh local"
      exit 1
    fi
    bash deploy/deploy.sh || {
      echo
      echo "❌ Kubernetes deploy failed. Fallback for the demo: bash vm/go.sh local"
      exit 1
    }
    [[ -f .app-url ]] && wait_healthy "LIVE (event network)" "$(cat .app-url)" && show_status "$(cat .app-url)"
    [[ -f .tunnel-url ]] && wait_healthy "PUBLIC (share this)" "$(cat .tunnel-url)"
    ;;
esac
