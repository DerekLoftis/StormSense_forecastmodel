"""StormSense's own astronomical tide for each gauge: a least-squares harmonic analysis
(UTide, with nodal corrections) of about a year of hourly heights, as noaa_override.py
already does for CBBT, MNPT, DAHL, WADC and BRDH.

- 27 constituents (no Sa or Ssa: the seasonal cycle and other slow changes in mean level are
  left in the residual, which the surge model forecasts from the recent record).
- Under about 6 months of record (CNRC, USLN-X): only the 13 constituents a month separates.
- The fitted mean is kept, so the tide is in feet above the gauge's MLLW, like the observations.

Fits are stored as JSON in harmonics/<ST>.json and refit about monthly.

    python -m ssmodel.tide fit [--stations SWPT,USJT] [--days 369]   fit and check against NOAA
"""
import csv
import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np

from . import obs

HERE = os.path.dirname(os.path.abspath(__file__))
HARMONICS = os.path.join(os.path.dirname(HERE), "harmonics")
CONSTITUENTS = ["M2", "S2", "N2", "K1", "O1", "K2", "P1", "Q1", "NU2", "MU2", "2N2", "L2", "T2", "LDA2", "J1",
                "OO1", "NO1", "RHO1", "2Q1", "MK3", "MO3", "M3", "M4", "MS4", "MN4", "S4", "M6"]
SHORT_RECORD = ["M2", "S2", "N2", "K1", "O1", "Q1", "2N2", "L2", "MK3", "M4", "MS4", "MN4", "M6"]
_cache = {}


def fit(times, heights, lat):
    """UTide coefficients for hourly (or finer) heights; times are UTC datetimes."""
    import utide
    t = np.array([np.datetime64(x.replace(tzinfo=None)) for x in times])
    constit = CONSTITUENTS if (times[-1] - times[0]) > timedelta(days=180) else SHORT_RECORD
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return utide.solve(t, np.asarray(heights, float), lat=lat, constit=constit, method="ols",
                           conf_int="none", trend=False, nodal=True, verbose=False)


def to_json(coef, **meta):
    """A UTide coefficient Bunch as plain JSON (the per-sample weights are left out)."""
    def enc(x):
        if hasattr(x, "items"):
            return {k: enc(v) for k, v in x.items() if k != "weights"}
        if isinstance(x, np.ndarray):
            return {"__array__": x.tolist(), "dtype": str(x.dtype)}
        if isinstance(x, np.generic):
            return x.item()
        return x
    return {"meta": meta, "coef": enc(coef)}


def from_json(doc):
    from utide.utilities import Bunch

    def dec(x):
        if isinstance(x, dict):
            if "__array__" in x:
                return np.array(x["__array__"], dtype=x["dtype"])
            return Bunch({k: dec(v) for k, v in x.items()})
        return x
    return dec(doc["coef"])


def load(st, folder=HARMONICS):
    if (st, folder) not in _cache:
        _cache[st, folder] = from_json(json.load(open(os.path.join(folder, f"{st}.json"))))
    return _cache[st, folder]


def predict(coef, times):
    """Tide in ft above MLLW at the given UTC datetimes (the fitted mean included)."""
    import utide
    t = np.array([np.datetime64(x.replace(tzinfo=None)) for x in times])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return utide.reconstruct(t, coef, verbose=False).h


def hourly(t, v):
    """Top-of-the-hour values of a 6-minute series."""
    on = np.array([x.minute == 0 for x in t], bool)
    return t[on], v[on]


def record(st, info, a, b):
    """Hourly heights ft MLLW for the fit: the top of each hour of the gauge's own quality-
    controlled record (at NOAA gauges the 6-minute water levels, which, unlike CO-OPS's
    verified hourly heights, run up to the present)."""
    return hourly(*obs.fetch(st, a, b, info))


def check(st, info, coef, days=30):
    """RMS difference (ft) between our tide and NOAA's published predictions over the last
    `days`, and the difference of their means, at CO-OPS gauges; None elsewhere."""
    if info["source"] != "coops":
        return None, None
    b = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t, p = obs.coops_predictions(info["source_id"], b - timedelta(days=days), b)
    if not len(t):
        return None, None
    d = predict(coef, t) - p
    return float(np.sqrt(np.mean((d - d.mean()) ** 2))), float(d.mean())


def main(argv):
    if not argv or argv[0] != "fit":
        sys.exit(__doc__)
    table = obs.stations()
    names = argv[argv.index("--stations") + 1].split(",") if "--stations" in argv else sorted(table)
    days = float(argv[argv.index("--days") + 1]) if "--days" in argv else 369.0
    out = argv[argv.index("--out") + 1] if "--out" in argv else HARMONICS
    os.makedirs(out, exist_ok=True)
    b = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    a = b - timedelta(days=days)
    rows = []
    for st in names:
        info = table[st]
        try:
            t, v = record(st, info, a, b)
            if len(t) < 24 * 14:
                raise ValueError(f"only {len(t)} hourly heights")
            coef = fit(t, v, info["lat_f"])
            resid = v - predict(coef, t)
            doc = to_json(coef, station=st, source=info["source"], source_id=info["source_id"],
                          start=t[0].strftime("%Y-%m-%dT%HZ"), end=t[-1].strftime("%Y-%m-%dT%HZ"), n_hours=len(t),
                          fitted=b.strftime("%Y-%m-%dT%HZ"), residual_sd_ft=round(float(np.std(resid)), 3))
            json.dump(doc, open(os.path.join(out, f"{st}.json"), "w"), separators=(",", ":"))
            rms, bias = check(st, info, coef)
            rows.append([st, len(t), doc["meta"]["start"], doc["meta"]["end"], len(coef.name),
                         round(float(coef.mean), 3), doc["meta"]["residual_sd_ft"],
                         "" if rms is None else round(rms, 3), "" if bias is None else round(bias, 3), ""])
        except Exception as e:
            rows.append([st, 0, "", "", "", "", "", "", "", str(e)[:160]])
        print(",".join(str(x) for x in rows[-1]), flush=True)
    with open(os.path.join(out, "fit_report.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "n_hours", "start", "end", "n_constituents", "mean_ft_mllw", "residual_sd_ft",
                    "rms_vs_noaa_ft", "mean_vs_noaa_ft", "error"])
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
