"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    m = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[m], yp[m]
    if not len(yt):
        return {"n": 0}
    err = yp - yt
    denom = (np.abs(yt) + np.abs(yp)) / 2
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) or np.nan
    return {
        "n": int(len(yt)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "smape": float(np.mean(np.where(denom > 0, np.abs(err) / denom, 0)) * 100),
        "bias": float(np.mean(err)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot == ss_tot else float("nan"),
    }


def daily_shape_metrics(df: pd.DataFrame, top_n: int = 6, tz: str = "Europe/London") -> dict[str, float]:
    """How well the model orders the half-hours within each day (what matters for
    'when to buy / sell'). df: index=timestamp, columns y_true, y_pred."""
    if df.empty:
        return {}
    day = df.index.tz_convert(tz).date
    spearman, hit_low, hit_high, direction = [], [], [], []
    prev_true = None
    for _, g in df.groupby(day):
        g = g.dropna()
        if len(g) < top_n * 2:
            continue
        spearman.append(g["y_true"].rank().corr(g["y_pred"].rank()))
        lo_t = set(g["y_true"].nsmallest(top_n).index)
        lo_p = set(g["y_pred"].nsmallest(top_n).index)
        hi_t = set(g["y_true"].nlargest(top_n).index)
        hi_p = set(g["y_pred"].nlargest(top_n).index)
        hit_low.append(len(lo_t & lo_p) / top_n)
        hit_high.append(len(hi_t & hi_p) / top_n)
        mt, mp = g["y_true"].mean(), g["y_pred"].mean()
        if prev_true is not None:
            direction.append(np.sign(mt - prev_true) == np.sign(mp - prev_true))
        prev_true = mt
    return {
        "intraday_spearman": float(np.nanmean(spearman)) if spearman else float("nan"),
        "cheapest_hit_rate": float(np.mean(hit_low)) if hit_low else float("nan"),
        "dearest_hit_rate": float(np.mean(hit_high)) if hit_high else float("nan"),
        "daily_direction_acc": float(np.mean(direction)) if direction else float("nan"),
    }
