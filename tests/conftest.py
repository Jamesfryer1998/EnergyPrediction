import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.pop("ENERGY_DATA_DIR", None)
os.environ.pop("ENERGY_CONFIG", None)
os.environ["ENERGY_NO_SCHEDULER"] = "1"  # tests drive Scheduler.tick() directly

# Never use a real IG login from .env in tests: settings.py loads .env on import, so import it
# first and then drop the IG variables.
import settings  # noqa: E402,F401

for _k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_ACCOUNT_ID"):
    os.environ.pop(_k, None)
