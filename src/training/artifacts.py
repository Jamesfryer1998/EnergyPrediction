"""Where models, tuned params and run records live on disk."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from settings import Settings


class Artifacts:
    def __init__(self, settings: Settings, region: str = "GB"):
        self.root = settings.data_dir
        self.region = region

    def model_dir(self, target: str, model: str) -> Path:
        return self.root / "models" / self.region / target / model

    def model_path(self, target: str, model: str) -> Path:
        return self.model_dir(target, model) / "model.pkl"

    # tuned hyper-parameters ------------------------------------------------
    def params_path(self, target: str, model: str) -> Path:
        return self.model_dir(target, model) / "best_params.json"

    def load_params(self, target: str, model: str) -> dict | None:
        p = self.params_path(target, model)
        return json.loads(p.read_text()) if p.exists() else None

    def save_params(self, target: str, model: str, params: dict, score: float, n_trials: int) -> None:
        p = self.params_path(target, model)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"params": params, "val_mae": score, "n_trials": n_trials,
                                 "tuned_at": _now()}, indent=2, default=str))

    # champion per target ---------------------------------------------------
    def champion_path(self, target: str) -> Path:
        return self.root / "models" / self.region / target / "champion.json"

    def load_champion(self, target: str) -> dict | None:
        p = self.champion_path(target)
        return json.loads(p.read_text()) if p.exists() else None

    def save_champion(self, target: str, model: str, metrics: dict, run_id: str) -> None:
        p = self.champion_path(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"model": model, "metrics": metrics, "run_id": run_id, "chosen_at": _now()},
                                indent=2, default=str))

    # ensemble blend per target ---------------------------------------------
    def ensemble_path(self, target: str) -> Path:
        return self.root / "models" / self.region / target / "ensemble.json"

    def load_ensemble(self, target: str) -> dict | None:
        p = self.ensemble_path(target)
        return json.loads(p.read_text()) if p.exists() else None

    def save_ensemble(self, target: str, spec: dict) -> None:
        p = self.ensemble_path(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**spec, "saved_at": _now()}, indent=2, default=str))

    # run records -----------------------------------------------------------
    @property
    def runs_dir(self) -> Path:
        d = self.root / "runs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_run(self, record: dict) -> Path:
        p = self.runs_dir / f"{record['run_id']}.json"
        p.write_text(json.dumps(record, indent=2, default=str))
        return p

    def list_runs(self, limit: int = 30) -> list[dict]:
        files = sorted(self.runs_dir.glob("*.json"), reverse=True)[:limit]
        out = []
        for f in files:
            try:
                out.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                continue
        return out

    def study_generation(self, target: str, model: str, reset_every_days: int) -> tuple[str, bool, int]:
        """Optuna study name for a target/model, starting a fresh study ("generation") every
        `reset_every_days` (0 = never). Returns (study name, is_new_generation, age in days)."""
        d = self.root / "optuna"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "generations.json"
        try:
            gens = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            gens = {}
        key = f"{self.region}:{target}:{model}"
        now = datetime.now(timezone.utc)
        g = gens.get(key)
        age = (now - datetime.fromisoformat(g["started"])).days if g else 0
        new = g is None or (reset_every_days > 0 and age >= reset_every_days)
        if new:
            g = {"gen": (g["gen"] + 1) if g else 1, "started": now.isoformat(timespec="seconds")}
            gens[key] = g
            p.write_text(json.dumps(gens, indent=2))
            age = 0
        # generation 1 keeps the original name, so the studies from before this existed carry on
        name = key if g["gen"] == 1 else f"{key}:g{g['gen']}"
        return name, new, age

    @property
    def optuna_storage(self) -> str:
        d = self.root / "optuna"
        d.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(d / 'studies.db').as_posix()}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(kind: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{kind}"
