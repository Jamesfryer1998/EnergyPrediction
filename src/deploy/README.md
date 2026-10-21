# Server setup (one-off)

Releases use **GitHub Actions**: `.github/workflows/ci.yml` lints and tests every push, and
`.github/workflows/deploy.yml` deploys `main` once CI passes. The deploy job runs **on the server**
through a self-hosted GitHub Actions runner. The runner
checks out the commit and calls `src/deploy/deploy.sh`, which syncs the code, updates the conda
env if dependencies changed, installs the systemd units and restarts the web service.

Tested layout: an Ubuntu 24.04 VM or LXC (for example on the Proxmox box). Give it about 4
cores, 8 GB RAM and 30 GB disk. The runner only makes outbound connections to GitHub, so
nothing needs opening on your router.

## 1. User, folders, conda

```bash
apt update && apt install -y sudo curl git rsync ca-certificates   # as root in a fresh container
sudo adduser --system --group --shell /bin/bash --home /home/energy energy
sudo mkdir -p /opt/energy-prediction /var/lib/energy-prediction/data
sudo chown -R energy:energy /opt/energy-prediction /var/lib/energy-prediction

sudo -iu energy
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
```

## 2. Let the deploy restart services without a password

`sudo visudo -f /etc/sudoers.d/energy-prediction`:

```
energy ALL=(root) NOPASSWD: /usr/bin/systemctl daemon-reload, \
  /usr/bin/systemctl enable energy-prediction-web.service, \
  /usr/bin/systemctl restart energy-prediction-web.service, \
  /usr/bin/systemctl --no-pager --lines=0 status energy-prediction-web.service, \
  /usr/bin/install -m 0644 /opt/energy-prediction/app/src/deploy/systemd/* /etc/systemd/system/*, \
  /usr/bin/journalctl -u energy-prediction-web -n 50 --no-pager
```

## 3. Self-hosted GitHub Actions runner

Repo → Settings → Actions → Runners → *New self-hosted runner* (Linux). Run the install
commands it shows as the `energy` user, and when asked for labels add `energy-prediction`.
Then install it as a service: `sudo ./svc.sh install energy && sudo ./svc.sh start`.

The deploy job uses a GitHub *environment* called `production` (created automatically on the
first run). Repo → Settings → Environments → production lets you require a manual approval
before each deploy, if you want one. You can also deploy any time from the Actions tab: Deploy
→ *Run workflow*.

If a pipeline job or paper-trading cycle is running when a deploy arrives, `deploy.sh` waits
for it to finish (up to `DEPLOY_WAIT_MINUTES`, default 60) before restarting the service.

## 4. Optional environment file

`/etc/energy-prediction.env` (read by both units). This is where server secrets go: the code
is rsynced on every deploy, so don't keep a `.env` in the app folder.

```
FLASK_SECRET_KEY=change-me
# IG demo login for paper trading
IG_API_KEY=...
IG_USERNAME=...
IG_PASSWORD=...
```

`sudo chown root:energy /etc/energy-prediction.env && sudo chmod 640 /etc/energy-prediction.env`
(only root and the `energy` user can read it).

To use the CLI on the server with the same settings, as `energy`:

```bash
cd /opt/energy-prediction/app && conda activate EnergyPrediction
set -a; . /etc/energy-prediction.env; set +a
export ENERGY_DATA_DIR=/var/lib/energy-prediction/data
python src/main.py ig-check
```

The dashboard has no login. Keep port 5050 on your LAN or VPN, or put it behind a reverse proxy
with authentication. Don't expose it to the internet as it is.

## 5. First deploy and history backfill

Push to `main` (or run the Deploy workflow by hand). Then backfill once:

```bash
sudo -iu energy
conda activate EnergyPrediction
cd /opt/energy-prediction/app
ENERGY_DATA_DIR=/var/lib/energy-prediction/data python src/main.py ingest --full
ENERGY_DATA_DIR=/var/lib/energy-prediction/data python src/main.py nightly --skip-ingest --force-tune
```

## Daily schedule

The daily pipeline is scheduled from the dashboard's **SCHEDULE** tab and runs inside the
web service, so nothing else is needed. `src/deploy/systemd/energy-prediction-nightly.timer` is
kept as an alternative for running without the dashboard. Enable it by hand only if the
dashboard schedule is stopped, or the pipeline will run twice:
`sudo systemctl enable --now energy-prediction-nightly.timer`.

## Useful commands

```bash
systemctl status energy-prediction-web
journalctl -u energy-prediction-nightly -f          # watch a nightly run
sudo systemctl start energy-prediction-nightly      # run the nightly job now
```
