"""Replay the StormSense forecast (v1 blend plus the predictor-corrector) over archived StormSense Ensemble runs and score it, and every
member, against the observations that came in afterwards, with Derek's skill formula.

    python tools/hindcast_ensemble.py RUNS_DIR OUT_DIR [--tau 12]

RUNS_DIR holds ensemble_<cycle>.tar.gz files (or folders of the same name, extracted) from the
StormSense-ensemble releases. Truth is the observed_ft column stitched across the runs, newest
run winning, as the skill scoring does. The tide is each run's astro_tide_ft (the P-ETSS tide
lined up with the gauge); the v1 blend barely depends on it. Skill weights for each run come only
and the corrector's offsets come only from earlier runs verified before that run's start (no
look-ahead). "stormsense" is the corrected forecast, "stormsense_v1" the blend alone.

Writes scores.csv (run x station x forecast x band) and summary.csv (paired comparison of the
StormSense blend with each member over the same runs, stations and bands).
"""
import csv
import os
import sys
from collections import defaultdict
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssmodel import archive, corrector, surge  # noqa: E402

BANDS = surge.BANDS + [("0-72", 0, 72)]
SKIP = {"DAHL", "WADC", "USSC", "USAR", "USCB", "USNR", "USSB", "GS01", "USBB"}  # aliased obs in Tidewatch's record
WINDOW = timedelta(days=5)


def stats(f, o):
    """R2, bias, MAE, RMSE, SD of obs and Derek's skill for paired arrays."""
    ok = ~np.isnan(f) & ~np.isnan(o)
    f, o = f[ok], o[ok]
    if len(o) < 30:
        return None
    sd = float(np.std(o))
    e = f - o
    bias, mae, rmse = float(e.mean()), float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))
    r2 = float(np.corrcoef(f, o)[0, 1] ** 2) if np.std(f) > 0 and sd > 0 else 0.0
    skill = 0.25 * (np.clip(r2, 0, 1) + np.exp(-abs(bias) / sd) + np.exp(-mae / sd) + np.exp(-rmse / sd)) if sd > 0 else np.nan
    return {"N": len(o), "R2": r2, "Bias": bias, "MAE": mae, "RMSE": rmse, "SDobs": sd, "Skill": float(skill)}


def predict(grid, t0, cols):
    """The uncorrected blend, as the corrector re-runs it on each earlier run."""
    if "astro_tide" not in cols:
        return None
    return surge.forecast(grid, t0, cols, cols["astro_tide"])[0]["stormsense"], cols["astro_tide"]


def main(argv):
    runs_dir, out = argv[0], argv[1]
    tau = float(argv[argv.index("--tau") + 1]) if "--tau" in argv else surge.TAU_H
    os.makedirs(out, exist_ok=True)
    runs = archive.read_folder(runs_dir)
    truth = archive.truth_series(runs)
    last_obs = max(max(d) for d in truth.values())
    scores = []      # rows for scores.csv
    paired = []      # (forecast, band, station, its stats, StormSense's stats on the same hours)
    errors = []      # (verified_at, station, member, band, rmse) for no-look-ahead weights
    for cyc in sorted(runs):
        for st, (grid, cols) in sorted(runs[cyc].items()):
            if st in SKIP or "observed" not in cols or "astro_tide" not in cols:
                continue
            seen = np.where(~np.isnan(cols["observed"]))[0]
            if not len(seen):
                continue
            t0 = grid[seen[-1]]
            tide = cols["astro_tide"]
            # skill weights from forecasts already verified by t0
            recent = defaultdict(lambda: defaultdict(list))
            for when, s, m, b, r in errors:
                if s == st and t0 - WINDOW <= when <= t0:
                    recent[m][b].append(r)
            skill = {m: {b: float(np.sqrt(np.mean(np.square(v)))) for b, v in bb.items()} for m, bb in recent.items()}
            fc = {}
            v1, _ = surge.forecast(grid, t0, cols, tide, skill or None, tau)
            fc["stormsense_v1"] = v1["stormsense"]
            fc["stormsense_eq"] = surge.forecast(grid, t0, cols, tide, None, tau)[0]["stormsense"]
            # predictor-corrector: the blend's own errors in earlier runs, verified by t0
            past = {c: r for c, r in runs.items() if c < cyc}
            corr, _ = corrector.correction_for(st, past, truth, t0, predict)
            fc["stormsense"] = corrector.apply(grid, t0, {"stormsense": v1["stormsense"].copy()}, tide, corr)["stormsense"]
            for k in surge.MEMBERS + ["baseline"]:
                if k in cols:
                    fc[k] = cols[k]
            lead = surge.lead_hours(grid, t0)
            obs_after = np.array([truth[st].get(t, np.nan) for t in grid])
            obs_after[lead <= 0] = np.nan
            for band, a, b in BANDS:
                sel = (lead > a) & (lead <= b)
                end = t0 + timedelta(hours=b)
                if end > last_obs:
                    continue  # not yet fully verified
                for k, f in fc.items():
                    s = stats(f[sel], obs_after[sel])
                    if s is None:
                        continue
                    scores.append({"cycle": cyc, "station": st, "forecast": k, "band": band, **s})
                    if band != "0-72" and k in surge.MEMBERS:
                        errors.append((end, st, k, band, s["RMSE"]))
                    if k != "stormsense":  # StormSense over exactly the hours this forecast covers
                        same = np.where(np.isnan(f), np.nan, fc["stormsense"])
                        p = stats(same[sel], obs_after[sel])
                        if p is not None:
                            paired.append((k, band, st, s, p))
    head = ["cycle", "station", "forecast", "band", "N", "R2", "Bias", "MAE", "RMSE", "SDobs", "Skill"]
    with open(os.path.join(out, "scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, head)
        w.writeheader()
        for r in scores:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    # paired comparison: StormSense vs each forecast on the same cycle, station and band
    rows = []
    for k in ["stormsense_v1", "stormsense_eq"] + surge.MEMBERS + ["baseline"]:
        for band, _, _ in BANDS:
            pairs = [(p, s) for f, b, st, s, p in paired if f == k and b == band]
            if not pairs:
                continue
            ss = np.array([[p[0]["RMSE"], p[0]["Skill"], p[1]["RMSE"], p[1]["Skill"]] for p in pairs])
            wins = int(np.sum(ss[:, 0] < ss[:, 2]))
            rows.append([k, band, len(pairs), len({st for f, b, st, s, p in paired if f == k and b == band}), round(ss[:, 2].mean(), 3),
                         round(ss[:, 0].mean(), 3), round(ss[:, 3].mean(), 3), round(ss[:, 1].mean(), 3),
                         f"{wins}/{len(pairs)}"])
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["forecast", "band", "pairs", "stations", "rmse_ft", "stormsense_rmse_ft", "skill",
                    "stormsense_skill", "stormsense_better"])
        w.writerows(rows)
    for r in rows:
        print(",".join(str(x) for x in r))


if __name__ == "__main__":
    main(sys.argv[1:])
