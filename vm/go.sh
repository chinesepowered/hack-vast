#!/usr/bin/env bash
# One command on the workshop VM: update the code, check every service, deploy, print the URL.
#
#   bash ~/hack-vast/vm/go.sh          # preflight + deploy to Kubernetes (the real demo URL)
#   bash ~/hack-vast/vm/go.sh check    # preflight only
#   bash ~/hack-vast/vm/go.sh local    # fallback: run the app on the VM itself at http://localhost:8080
#   bash ~/hack-vast/vm/go.sh warm     # precompute the demo (coverage grid + verdicts) inside the pod
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== updating code"
git pull --ff-only -q 2>/dev/null || echo "   (git pull skipped)"

mode="${1:-deploy}"
if [[ "$mode" != "warm" ]]; then
  python3 vm/preflight.py
fi

case "$mode" in
  check)
    ;;
  local)
    mapfile -t TEAM_CONFIGS < <(find /config -maxdepth 1 -type f -name '*.config' | sort)
    (( ${#TEAM_CONFIGS[@]} == 1 )) || { echo "expected exactly one /config/*.config"; exit 1; }
    set -a && source "${TEAM_CONFIGS[0]}" && set +a
    cd app
    python3 -m pip install -q --user -r requirements.txt 2>/dev/null \
      || python3 -m pip install -q --user --break-system-packages -r requirements.txt
    echo "== open http://localhost:8080 in the VM's browser (Ctrl+C to stop)"
    exec python3 main.py
    ;;
  warm)
    export KUBECONFIG="${KUBECONFIG:-/config/kubeconfig}"
    [[ -f "$KUBECONFIG" ]] || KUBECONFIG="$(find /config -maxdepth 1 -name '*-k8s.yaml' | head -1)"
    mapfile -t TEAM_CONFIGS < <(find /config -maxdepth 1 -type f -name '*.config' | sort)
    set -a && source "${TEAM_CONFIGS[0]}" && set +a
    echo "== precomputing the demo inside the pod (takes a few minutes)"
    kubectl -n "$USERNAME" exec deploy/"${APP_NAME:-edge-case-miner}" -- python main.py --warm
    ;;
  *)
    # vm/READY is committed only once the app has been tested, so a half-built snapshot never deploys.
    if [[ ! -f vm/READY || ! -f deploy/deploy.sh ]]; then
      echo
      echo ">>> The app isn't ready yet. Send the report above to Claude, then re-run this command later."
      exit 0
    fi
    bash deploy/deploy.sh || {
      echo
      echo "❌ Kubernetes deploy failed. Fallback for the demo: bash ~/hack-vast/vm/go.sh local"
      exit 1
    }
    if [[ -f .app-url ]]; then
      url="$(cat .app-url)"
      echo "== waiting for $url/health"
      for _ in $(seq 1 40); do
        code="$(curl -s -o /dev/null -w '%{http_code}' "$url/health" || true)"
        [[ "$code" == "200" ]] && { echo "✅ LIVE: $url"; exit 0; }
        sleep 5
      done
      echo "⚠️  $url not healthy yet (last HTTP $code). Check: kubectl -n \$USERNAME logs deploy/${APP_NAME:-edge-case-miner}"
    fi
    ;;
esac
