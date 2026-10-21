"""Site-specific predictor-corrector: Derek's live data assimilation step.

The predictor is the v1 blend (surge.py), which already starts from the water the gauge has
now. The corrector learns how the predictor has been over- or under-predicting at this gauge
from its own past forecasts (recomputed from the archived ensemble runs, so no column of past
StormSense output is needed), checked against the observations that came in since, and corrects
the new forecast. For each lead band (0-12, 12-36, 36-72 h) it fits

    observed - predicted = a + b * predicted surge

by recency-weighted least squares over the runs of the last 7 days (weights exp(-age / 3 days)).
a is a standing offset at the site (mean water level drifting away from the tide's mean, a
datum slip); b scales the predicted surge (b = -0.3: the surge has been 30% too big, as with
Tidewatch's fixed 0.7 factor, but learned per site and lead). The correction for the new run
is a + b * its own predicted surge, eased across band edges and faded in over the first hours.

Guards against over- and under-correcting:
- Each coefficient is shrunk by James-Stein (to nothing when it is no bigger than its own
  standard error) and by n / (n + K) with K = 8 runs. Errors of one storm are correlated, so
  the standard errors count runs, not six-minute points.
- |a| <= 0.25 ft, |b| <= 0.3, total correction <= 0.3 ft; nothing before 4 verified runs.
- Faded in over the first hours (1 - exp(-h / 3 h)): the forecast still starts at the gauge.
- Recency weighting lets the correction follow seasonal mean-level changes and forget a
  storm within days.
"""
import numpy as np

EFOLD_DAYS = 3.0
WINDOW_DAYS = 7.0
MIN_RUNS = 4
K_RUNS = 8.0
CAP_A, CAP_B, CAP_FT = 0.25, 0.3, 0.3
TAPER_H = 3.0
EASE_H = 3.0
BANDS = [("0-12", 0, 12), ("12-36", 12, 36), ("36-72", 36, 72)]
LEADS = np.arange(1, 73)  # hourly lead bins (h-1, h]


def lead_errors(grid, t0, forecast, truth, now, tide):
    """Per lead hour bin: mean (observed - forecast), NaN where not yet verified, and the mean
    predicted surge (forecast - tide)."""
    lead = np.array([(t - t0).total_seconds() / 3600.0 for t in grid])
    obs = np.array([truth.get(t, np.nan) if t <= now else np.nan for t in grid])
    e, s = obs - forecast, forecast - tide
    err, surge = np.full(len(LEADS), np.nan), np.full(len(LEADS), np.nan)
    for i, h in enumerate(LEADS):
        sel = (lead > h - 1) & (lead <= h) & ~np.isnan(e) & ~np.isnan(s)
        if sel.sum() >= 5:  # most of the hour's 10 six-minute values
            err[i], surge[i] = e[sel].mean(), s[sel].mean()
    return err, surge


def _shrink(coef, se, n):
    js = max(0.0, 1.0 - (se / coef) ** 2) if coef else 0.0
    return coef * js * n / (n + K_RUNS)


def fit_band(rows):
    """(a, b, info) for one band from [(weight, run id, error, surge)]."""
    w, run, e, s = (np.array(x) for x in zip(*rows))
    runs = np.unique(run)
    wr = np.array([w[run == r][0] for r in runs])
    n_eff = wr.sum() ** 2 / (wr ** 2).sum()
    if len(runs) < MIN_RUNS:
        return 0.0, 0.0, None
    X = np.column_stack([np.ones(len(s)), s])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], e * sw, rcond=None)
    res = e - X @ coef
    sigma2 = (w * res ** 2).sum() / w.sum()
    xtwx = (X * w[:, None]).T @ X / w.sum()
    try:
        cov = sigma2 * np.linalg.inv(xtwx) / n_eff  # run-level sample size, not points
    except np.linalg.LinAlgError:
        return 0.0, 0.0, None
    se_a, se_b = np.sqrt(np.clip(np.diag(cov), 0, None))
    if np.std(s) < 0.05:  # surge hardly varied: the slope is undetermined, fit the offset alone
        a_raw, b_raw = float(np.average(e, weights=w)), 0.0
        se_a = np.sqrt(np.average((e - a_raw) ** 2, weights=w) / n_eff)
        a, b = _shrink(a_raw, se_a, n_eff), 0.0
    else:
        a_raw, b_raw = float(coef[0]), float(coef[1])
        a, b = _shrink(a_raw, se_a, n_eff), _shrink(b_raw, se_b, n_eff)
    a, b = float(np.clip(a, -CAP_A, CAP_A)), float(np.clip(b, -CAP_B, CAP_B))
    return a, b, {"runs": int(len(runs)), "fit_a_ft": round(a_raw, 3), "fit_b": round(b_raw, 3),
                  "a_ft": round(a, 3), "b": round(b, 3)}


def fit(errors, now):
    """Per-band (a, b) from [(t0, errors, surges)] of past runs. Returns (coefs, info)."""
    errors = [x for x in errors if (now - x[0]).total_seconds() <= WINDOW_DAYS * 86400]
    coefs, info = {}, {"runs": len(errors), "bands": {}}
    for name, lo, hi in BANDS:
        rows = []
        for i, (t0, err, surge) in enumerate(errors):
            w = np.exp(-(now - t0).total_seconds() / 86400.0 / EFOLD_DAYS)
            sel = (LEADS > lo) & (LEADS <= hi) & ~np.isnan(err)
            rows += [(w, i, e, s) for e, s in zip(err[sel], surge[sel])]
        a, b, band = fit_band(rows) if rows else (0.0, 0.0, None)
        coefs[name] = (a, b)
        if band:
            info["bands"][name] = band
    info["applied"] = any(a or b for a, b in coefs.values())
    return coefs, info


def correction(grid, t0, surge, coefs):
    """Correction (ft) on grid for a forecast whose predicted surge is `surge`."""
    lead = np.array([(t - t0).total_seconds() / 3600.0 for t in grid])
    (a0, b0), (a1, b1), (a2, b2) = (coefs.get(n, (0.0, 0.0)) for n, _, _ in BANDS)
    knots = [BANDS[0][2] - EASE_H, BANDS[0][2] + EASE_H, BANDS[1][2] - EASE_H, BANDS[1][2] + EASE_H]
    a = np.interp(lead, knots, [a0, a1, a1, a2])
    b = np.interp(lead, knots, [b0, b1, b1, b2])
    c = np.clip(a + b * np.nan_to_num(surge), -CAP_FT, CAP_FT) * (1 - np.exp(-np.clip(lead, 0, None) / TAPER_H))
    c[lead <= 0] = 0.0
    return c


def apply(grid, t0, out, tide, coefs):
    """Correct the forecast and its band (dict of arrays on grid, key "stormsense" first) in place,
    all by the correction for the forecast's own predicted surge."""
    c = correction(grid, t0, out["stormsense"] - tide, coefs)
    for k in out:
        out[k] = out[k] + c
    return out


def correction_for(st, past_runs, truth, now, predict):
    """Fit the corrector for station st. past_runs: {cycle: {station: (grid, cols)}} of earlier
    ensemble runs; truth: {station: {time: obs}}; predict(grid, t0, cols) -> (forecast, tide)
    for that run, or None."""
    errs = []
    for cyc, by_st in past_runs.items():
        if st not in by_st:
            continue
        grid, cols = by_st[st]
        seen = np.where(~np.isnan(cols.get("observed", np.full(len(grid), np.nan))))[0]
        if not len(seen) or grid[seen[-1]] >= now:
            continue
        t0 = grid[seen[-1]]
        try:
            p = predict(grid, t0, cols)
        except Exception:
            continue
        if p is None:
            continue
        err, surge = lead_errors(grid, t0, p[0], truth.get(st, {}), now, p[1])
        if not np.isnan(err).all():
            errs.append((t0, err, surge))
    return fit(errs, now)
