"""Ensemble champion: a weighted average of the best few models, weighted from their backtest.

Nothing new is trained for it. The walk-forward backtest already stores every model's
out-of-sample predictions; the blend weights are fitted on the first half of that window and
scored on the second half (alongside the single models, on the same days), so the comparison
that decides the champion is fair. If the blend wins, production uses the weights refitted on the
whole window and trains each member model as usual.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from storage import TS

log = logging.getLogger(__name__)

NAME = "ensemble"


def fit_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise the mean absolute error of P @ w against y.
    Solved exactly as a linear programme; falls back to inverse-MAE weights if that fails."""
    n, k = P.shape
    if k == 1:
        return np.ones(1)
    try:
        from scipy.optimize import linprog

        # variables: w (k), e (n) with e_i >= |P_i w - y_i|; minimise sum(e)
        c = np.r_[np.zeros(k), np.ones(n) / n]
        eye = np.eye(n)
        A_ub = np.block([[P, -eye], [-P, -eye]])
        b_ub = np.r_[y, -y]
        A_eq = np.r_[np.ones(k), np.zeros(n)][None, :]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0, 1)] * k + [(0, None)] * n, method="highs")
        if res.success:
            w = np.clip(res.x[:k], 0, None)
            return w / w.sum()
    except Exception as exc:  # scipy missing / solver trouble
        log.debug("ensemble LP failed (%s); using inverse-MAE weights", exc)
    mae = np.abs(P - y[:, None]).mean(axis=0)
    w = 1 / np.maximum(mae, 1e-9)
    return w / w.sum()


def prune(weights: dict[str, float], min_weight: float) -> dict[str, float]:
    """Drop members below min_weight and renormalise (keeps at least the largest one)."""
    kept = {m: w for m, w in weights.items() if w >= min_weight} or {max(weights, key=weights.get): 1.0}
    total = sum(kept.values())
    return {m: w / total for m, w in sorted(kept.items(), key=lambda kv: -kv[1])}


def describe(weights: dict[str, float]) -> str:
    return " · ".join(f"{m} {w:.2f}" for m, w in weights.items())


def build(preds: pd.DataFrame, board: pd.DataFrame, target: str, max_members: int = 3,
          min_weight: float = 0.05) -> dict | None:
    """From one target's backtest predictions (TS, model, y_true, y_pred) and leaderboard rows,
    fit the blend. Returns None when fewer than two usable models ran."""
    ok = board[(board["target"] == target)]
    if "mae" not in ok:
        return None
    ok = ok.dropna(subset=["mae"]).sort_values("mae")
    members = list(ok["model"])[:max_members]
    if len(members) < 2:
        return None
    p = preds[preds["model"].isin(members)]
    wide = p.pivot_table(index=TS, columns="model", values="y_pred", aggfunc="last")
    y = p.groupby(TS)["y_true"].last()
    wide = wide.join(y.rename("__y")).dropna()
    if len(wide) < 96:  # need at least a couple of days to split
        return None
    days = pd.Index(pd.to_datetime(wide.index).normalize().unique())
    cut = days[len(days) // 2]
    fit_part, hold = wide[wide.index < cut], wide[wide.index >= cut]
    if len(fit_part) < 48 or len(hold) < 48:
        return None
    w_fit = fit_weights(fit_part[members].to_numpy(), fit_part["__y"].to_numpy())
    w_fit = prune(dict(zip(members, w_fit)), min_weight)
    hold_pred = hold[list(w_fit)].to_numpy() @ np.array(list(w_fit.values()))
    hold_mae = {m: float(np.abs(hold[m] - hold["__y"]).mean()) for m in members}
    ens_mae = float(np.abs(hold_pred - hold["__y"]).mean())
    # production weights: refit on the whole window
    w_all = fit_weights(wide[members].to_numpy(), wide["__y"].to_numpy())
    w_all = prune(dict(zip(members, w_all)), min_weight)
    if len(w_all) < 2:  # one model takes (nearly) all the weight: nothing to blend
        return None
    best_single = min(hold_mae, key=hold_mae.get)
    return {
        "target": target, "members": list(w_all), "weights": w_all, "holdout_weights": w_fit,
        "holdout_start": cut.isoformat(), "holdout_mae": ens_mae, "member_holdout_mae": hold_mae,
        "best_single": best_single, "best_single_holdout_mae": hold_mae[best_single],
        "beats_best_single": ens_mae < hold_mae[best_single],
        "holdout_preds": pd.DataFrame({TS: hold.index, "y_true": hold["__y"].to_numpy(), "y_pred": hold_pred}),
    }
