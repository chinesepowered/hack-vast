#!/usr/bin/env bash
# One command on the workshop VM: update the code, check every service, deploy, print the URL.
#
#   bash vm/go.sh          # preflight + deploy to Kubernetes (the real demo URL)
#   bash vm/go.sh check    # preflight only
#   bash vm/go.sh local    # fallback: run the app on this VM and open it in the VM's browser
#   bash vm/go.sh warm     # precompute the demo (coverage grid + verdicts) inside the pod
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
if [[ "$mode" != "warm" ]]; then
  python3 vm/preflight.py
fi

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
    if [[ -f .app-url ]]; then
      url="$(cat .app-url)"
      echo "== waiting for $url/health"
      for _ in $(seq 1 40); do
        code="$(curl -s -o /dev/null -w '%{http_code}' "$url/health" || true)"
        [[ "$code" == "200" ]] && { echo "✅ LIVE: $url"; exit 0; }
        sleep 5
      done
      echo "⚠️  $url not healthy yet (last HTTP $code). Logs: $KUBECTL -n $USERNAME logs deploy/$APP_NAME"
    fi
    ;;
esac
