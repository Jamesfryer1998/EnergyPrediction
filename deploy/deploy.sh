#!/usr/bin/env bash
# Deploy a checked-out commit to this server and restart the services.
# Called by the CI deploy job running on the server:  deploy/deploy.sh <checkout-dir>
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
  --exclude 'config/overrides.yaml' --exclude '__pycache__' \
  "$SRC"/ "$APP_DIR"/

# ── conda env: create on first deploy, update only when dependencies change ──
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
DEPS_HASH=$(cat "$APP_DIR/environment.yml" "$APP_DIR/requirements.txt" | sha256sum | cut -d' ' -f1)
HASH_FILE="$APP_ROOT/.deps_hash"
if ! conda env list | grep -qE "^$ENV_NAME\s"; then
  echo "==> Creating conda env $ENV_NAME"
  (cd "$APP_DIR" && conda env create -n "$ENV_NAME" -f environment.yml)
  echo "$DEPS_HASH" > "$HASH_FILE"
elif [[ "$(cat "$HASH_FILE" 2>/dev/null)" != "$DEPS_HASH" ]]; then
  echo "==> Dependencies changed; updating env"
  conda activate "$ENV_NAME"
  pip install -r "$APP_DIR/requirements.txt"
  conda deactivate
  echo "$DEPS_HASH" > "$HASH_FILE"
fi
ln -sfn "$(conda env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')" "$APP_ROOT/env"

# ── smoke test before touching the running service ──
conda activate "$ENV_NAME"
(cd "$APP_DIR" && python main.py --log-level WARNING sources > /dev/null)
conda deactivate

# ── systemd units (reload only if they changed) ──
UNITS_CHANGED=0
for unit in "$APP_DIR"/deploy/systemd/*; do
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
