# Sourced by vm/go.sh. Loads the team config and fills in what it leaves out, so
# preflight, deploy and the local fallback all see the same environment.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mapfile -t TEAM_CONFIGS < <(find /config -maxdepth 1 -type f -name '*.config' 2>/dev/null | sort)
if (( ${#TEAM_CONFIGS[@]} == 1 )); then
  set -a && source "${TEAM_CONFIGS[0]}" && set +a
fi

# The team config carries GPU_BEARER_TOKEN but not the model hosts. The organizers
# document them in their gpu skill (.cursor/skills/gpu/README.md): prefer the copy on
# this VM, fall back to the value published there.
if [[ -z "${COSMOS3_REASON_URL:-}" ]]; then
  gpu_skill="$HOME/vast-builders-challenge/.cursor/skills/gpu/README.md"
  gpu_host="$(grep -oE '^GPU_HOST=[0-9.]+' "$gpu_skill" 2>/dev/null | head -1 | cut -d= -f2)"
  export COSMOS3_REASON_URL="http://${gpu_host:-166.19.38.112}:8001"
fi

export KUBECONFIG="${KUBECONFIG:-/config/kubeconfig}"
if [[ ! -f "$KUBECONFIG" ]]; then
  KUBECONFIG="$(find /config -maxdepth 1 -name '*-k8s.yaml' 2>/dev/null | head -1)"
fi

# kubectl isn't preinstalled on the pool VMs: fetch a user-local copy once.
if ! command -v kubectl >/dev/null 2>&1; then
  if [[ ! -x "$REPO_ROOT/.bin/kubectl" ]]; then
    echo "== installing kubectl into $REPO_ROOT/.bin (one time)"
    mkdir -p "$REPO_ROOT/.bin"
    case "$(uname -m)" in aarch64|arm64) k_arch=arm64 ;; *) k_arch=amd64 ;; esac
    k_ver="$(curl -fsSL https://dl.k8s.io/release/stable.txt 2>/dev/null)"
    if [[ -n "$k_ver" ]] && curl -fsSL -o "$REPO_ROOT/.bin/kubectl" \
        "https://dl.k8s.io/release/${k_ver}/bin/linux/${k_arch}/kubectl"; then
      chmod +x "$REPO_ROOT/.bin/kubectl"
    else
      rm -f "$REPO_ROOT/.bin/kubectl"
      echo "   (kubectl download failed)"
    fi
  fi
  export PATH="$REPO_ROOT/.bin:$PATH"
fi
export KUBECTL="$(command -v kubectl || true)"
export APP_NAME="${APP_NAME:-edge-case-miner}"
