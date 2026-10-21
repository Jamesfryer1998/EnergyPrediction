#!/usr/bin/env bash
# Deploy a checked-out commit to this server and restart the services.
# Called by the CI deploy job running on the server:  src/deploy/deploy.sh <checkout-dir>
#
# Layout on the server:
#   /opt/energy-prediction/app        code (rsynced from the checkout)
#   /opt/energy-prediction/env        symlink to the EnergyPrediction conda env
#   /var/lib/energy-prediction/data   data store, models, runs (never touched by deploys)
set -euo pipefail

SRC="${1:?usage: deploy.sh <checkout-dir>}"
APP_ROOT="${APP_ROOT:-/opt/energy-prediction}"
APP_DIR="$APP_ROOT/app"
ENV_NAME="${ENV_NAME:-EnergyPrediction}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"

echo "==> Deploying $(git -C "$SRC" rev-parse --short HEAD) to $APP_DIR"
mkdir -p "$APP_DIR"
rsync -a --delete \
  --exclude '.git' --exclude 'data/' --exclude 'logs/' --exclude '.env' \
  --exclude 'src/config/overrides.yaml' --exclude '__pycache__' \
  "$SRC"/ "$APP_DIR"/

# ── conda env: create on first deploy, update only when dependencies change ──
# The server has no GPU, so torch is installed from PyTorch's CPU-only index first
# (~200 MB). From PyPI it would pull ~3 GB of CUDA libraries.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
DEPS_HASH=$(cat "$APP_DIR/environment.yml" "$APP_DIR/requirements.txt" | sha256sum | cut -d' ' -f1)
HASH_FILE="$APP_ROOT/.deps_hash"

install_pip_deps() {
  conda activate "$ENV_NAME"
  pip install --no-cache-dir "$(grep -E '^torch' "$APP_DIR/requirements.txt" | awk '{print $1}')" \
    --index-url "$TORCH_INDEX"
  pip install --no-cache-dir -r "$APP_DIR/requirements.txt"
  conda deactivate
}

env_exists() { conda env list | grep -qE "^$ENV_NAME\s"; }

if ! env_exists || [[ ! -f "$HASH_FILE" ]]; then
  # No hash file means the env was never finished (e.g. an earlier deploy failed part-way): rebuild it.
  echo "==> Creating conda env $ENV_NAME"
  env_exists && conda env remove -y -n "$ENV_NAME"
  # conda packages only; the pip section is installed by install_pip_deps
  sed '/^  - pip:/,$d' "$APP_DIR/environment.yml" > "$APP_ROOT/environment.conda.yml"
  conda env create -n "$ENV_NAME" -f "$APP_ROOT/environment.conda.yml"
  install_pip_deps
  echo "$DEPS_HASH" > "$HASH_FILE"
elif [[ "$(cat "$HASH_FILE")" != "$DEPS_HASH" ]]; then
  echo "==> Dependencies changed; updating env"
  install_pip_deps
  echo "$DEPS_HASH" > "$HASH_FILE"
fi
ln -sfn "$(conda env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')" "$APP_ROOT/env"

# ── smoke test before touching the running service ──
conda activate "$ENV_NAME"
(cd "$APP_DIR" && python src/main.py --log-level WARNING sources > /dev/null)
conda deactivate

# ── systemd units (reload only if they changed) ──
UNITS_CHANGED=0
for unit in "$APP_DIR"/src/deploy/systemd/*; do
  dest="/etc/systemd/system/$(basename "$unit")"
  if ! cmp -s "$unit" "$dest"; then
    sudo install -m 0644 "$unit" "$dest"
    UNITS_CHANGED=1
  fi
done
if [[ $UNITS_CHANGED -eq 1 ]]; then
  echo "==> systemd units changed; reloading"
  sudo systemctl daemon-reload
  sudo systemctl enable energy-prediction-web.service
fi

# ── don't cut a running pipeline job or trading cycle short ──
PORT="${WEB_PORT:-5050}"
WAIT_MINUTES="${DEPLOY_WAIT_MINUTES:-60}"
for ((i = 0; i < WAIT_MINUTES * 2; i++)); do
  hb=$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/api/heartbeat" 2>/dev/null || true)
  if [[ -z "$hb" ]] || ! grep -Eq '"status": ?"running"|"busy": ?true' <<<"$hb"; then
    break
  fi
  [[ $i -eq 0 ]] && echo "==> A job is running on the dashboard; waiting for it to finish (up to ${WAIT_MINUTES} min)"
  sleep 30
done

echo "==> Restarting services"
sudo systemctl restart energy-prediction-web.service
sudo systemctl --no-pager --lines=0 status energy-prediction-web.service
echo "==> Deploy complete"
