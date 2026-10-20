"""StormSense surge forecast, v1: the gauge's current residual handed over to a skill-weighted
blend of the ensemble members' surges.

All series are on one 6-minute grid in ft above the station's MLLW. For each member
(already lined up with the gauge by the ensemble):

    surge_m(t)    = member(t) - tide(t)
    anchored_m(t) = surge_m(t) + (r0 - surge_m(t0)) * exp(-(t - t0) / TAU_H)

r0 is the observed residual (observed - tide) over the last hour, so each member starts from
the water the gauge has now and relaxes to its own forecast surge, as Tidewatch offsets its
surge to the latest residual. The blend is the weighted mean of the anchored surges, with
weights 1/RMSE^2 per lead band (0-12, 12-36, 36-72 h) from the skill scoring's recent
summary, eased across the band edges. The forecast is tide + blend.

Since tide(t0) cancels in r0 - surge_m(t0), the blend depends on our tide only through the
tide's shape after t0; without any member the forecast falls back to tide plus decaying
persistence. The 10-90% band combines the anchored members' spread with an error that grows
with lead time (Z90 sigma), and is never narrower than P-ETSS's own 10-90% band.
"""
import warnings

import numpy as np

TAU_H = 12.0          # e-folding time (h) of the hand-over from the observed residual to each member
PERSIST_TAU_H = 24.0  # without members: persistence fading over a day, as the ensemble baseline
BANDS = [("0-12", 0, 12), ("12-36", 12, 36), ("36-72", 36, 72)]
EASE_H = 3.0          # weights move from one band's to the next over +/- this many hours
SIGMA0, SIGMA_SLOPE = 0.12, 0.004  # ft, ft per hour: error growth before the hindcast refines it
Z90 = 1.2816
# members that forecast total water at the gauge (the baseline is our own persistence)
MEMBERS = ["petss", "stofs2d", "stofs3d", "cbofs", "nwm_short", "nwm_medium", "etss", "psurge", "nws", "tidewatch"]
DEFAULT_RMSE = 0.35   # ft: a member with no recent score yet counts as roughly this good
MIN_RMSE = 0.05


def lead_hours(grid, t0):
    return np.array([(t - t0).total_seconds() / 3600.0 for t in grid])


def residual_now(grid, t0, observed, tide, hours=1.0):
    """Median observed residual over the last `hours` before t0 (NaN if no observations)."""
    h = lead_hours(grid, t0)
    sel = (h > -hours) & (h <= 0) & ~np.isnan(observed) & ~np.isnan(tide)
    return float(np.median(observed[sel] - tide[sel])) if sel.any() else np.nan


def band_weights(lead, rmse):
    """Weight per grid point for one member: 1/RMSE^2 of each band, eased over the edges.
    rmse: {band name: ft or None}."""
    w0, w1, w2 = [1.0 / max(rmse.get(name) or DEFAULT_RMSE, MIN_RMSE) ** 2 for name, _, _ in BANDS]
    e1, e2 = BANDS[0][2], BANDS[1][2]  # 12 and 36 h
    return np.interp(lead, [e1 - EASE_H, e1 + EASE_H, e2 - EASE_H, e2 + EASE_H], [w0, w1, w1, w2])


def anchor(lead, surge, r0, tau=TAU_H):
    """A member's surge shifted to start from r0 at t0, the shift decaying with lead."""
    ok = ~np.isnan(surge) & (lead >= 0)
    if not ok.any() or np.isnan(r0):
        return surge
    s0 = surge[ok][0] if lead[ok][0] <= 3 else np.nan  # its value at t0 (or within 3 h)
    if np.isnan(s0):
        return surge
    return surge + (r0 - s0) * np.exp(-np.clip(lead, 0, None) / tau)


def forecast(grid, t0, cols, tide, skill=None, tau=TAU_H):
    """StormSense forecast for one station.

    grid: 6-minute times; cols: {"observed": ..., member key: ..., "petss_p10"/"petss_p90"} on the
    grid, ft MLLW, members lined up with the gauge; tide: our tide on the grid; skill: {member:
    {band: rmse ft}} from the skill scoring (None = equal weights).
    Returns ({"stormsense", "stormsense_p10", "stormsense_p90"} arrays, notes)."""
    lead = lead_hours(grid, t0)
    fut = lead > 0
    r0 = residual_now(grid, t0, cols["observed"], tide)
    used, num, den, anchored = [], np.zeros(len(grid)), np.zeros(len(grid)), []
    for key in MEMBERS:
        if key not in cols:
            continue
        a = anchor(lead, cols[key] - tide, r0, tau)
        ok = fut & ~np.isnan(a)
        if not ok.any():
            continue
        w = band_weights(lead, (skill or {}).get(key, {}))
        num[ok] += w[ok] * a[ok]
        den[ok] += w[ok]
        anchored.append(np.where(ok, a, np.nan))
        used.append(key)
    persist = (r0 if not np.isnan(r0) else 0.0) * np.exp(-np.clip(lead, 0, None) / PERSIST_TAU_H)
    blend = np.where(den > 0, num / np.where(den > 0, den, 1), persist)
    mean = np.where(fut, tide + blend, np.nan)
    if len(anchored) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spread = np.nanstd(np.array(anchored), axis=0)
        spread = np.nan_to_num(spread)
    else:
        spread = np.zeros(len(grid))
    sigma = np.sqrt(spread ** 2 + (SIGMA0 + SIGMA_SLOPE * np.clip(lead, 0, None)) ** 2)
    lo, hi = mean - Z90 * sigma, mean + Z90 * sigma
    if "petss_p10" in cols and "petss_p90" in cols and "petss" in cols:
        # never narrower than P-ETSS's own ensemble band, centred on our forecast
        below = np.nan_to_num(cols["petss"] - cols["petss_p10"])
        above = np.nan_to_num(cols["petss_p90"] - cols["petss"])
        lo, hi = np.minimum(lo, mean - below), np.maximum(hi, mean + above)
    notes = {"version": "v1-blend", "residual_now_ft": None if np.isnan(r0) else round(r0, 3), "members": used,
             "tau_h": tau, "weights": "1/RMSE^2 per lead band" if skill else "equal (no skill scores yet)"}
    return {"stormsense": mean, "stormsense_p10": np.where(fut, lo, np.nan),
            "stormsense_p90": np.where(fut, hi, np.nan)}, notes
