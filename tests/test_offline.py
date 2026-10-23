"""Offline tests (no network): QC, the 6-minute clock, source parsers, and the tide fit's
JSON round trip.    python -m pytest tests  (or python tests/test_offline.py)"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssmodel import corrector, obs, surge, tide  # noqa: E402

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def series(hours, step_min=6, f=lambda h: 1.3 + np.cos(2 * np.pi * h / 12.42)):
    t = np.array([T0 + timedelta(minutes=step_min * i) for i in range(int(hours * 60 / step_min))])
    h = np.array([(x - T0).total_seconds() / 3600 for x in t])
    return t, f(h)


def test_qc_drops_spikes_range_and_stuck_runs():
    t, v = series(24)
    v = v.copy()
    v[50] += 3.0          # spike
    v[60] = 99.0          # out of range
    v[100:115] = v[100]   # stuck for 84 minutes
    tq, vq = obs.qc(t, v)
    assert 99.0 not in vq
    assert t[50] not in set(tq)
    assert not any(t[100] <= x <= t[114] for x in tq)
    assert len(tq) > len(t) - 25


def test_qc_flags_keep_every_reading_and_say_why():
    t, v = series(24)
    v = v.copy()
    v[50] += 3.0
    v[60] = 99.0
    v[100:115] = v[100]
    tf, vf, f = obs.qc_flags(t[::-1], v[::-1])  # sorted on the way
    assert len(tf) == len(t) and tf[0] == t[0]
    assert f[60] == 1 and f[50] == 2 and (f[100:115] == 3).all()
    tq, _ = obs.qc(t, v)
    assert list(tq) == list(tf[f == 0])


def test_flatline_keeps_short_holds():
    t, v = series(6)
    v = v.copy()
    v[10:13] = v[10]  # 18-minute hold, like USBB's steps: kept
    assert len(obs.flatline(t, v)[0]) == len(t)


def test_six_minute_clock_from_15_minutes_and_gaps():
    t, v = series(12, step_min=15)
    keep = np.ones(len(t), bool)
    keep[20:26] = False  # a 105-minute gap is not filled
    t6, v6 = obs.on_six_minutes(t[keep], v[keep])
    assert all(x.minute % 6 == 0 and x.second == 0 for x in t6)
    assert not any(t[20] < x < t[25] for x in t6)
    exact = 1.3 + np.cos(2 * np.pi * np.array([(x - T0).total_seconds() / 3600 for x in t6]) / 12.42)
    assert np.max(np.abs(v6 - exact)) < 0.05


def test_parse_nwis_and_ogc():
    nwis = {"value": {"timeSeries": [{"variable": {"variableCode": [{"value": "62620"}], "noDataValue": -999999.0},
                                      "values": [{"value": [{"value": "1.25", "dateTime": "2026-10-20T08:00:00.000-04:00"},
                                                            {"value": "-999999", "dateTime": "2026-10-20T08:15:00.000-04:00"}]}]}]}}
    t, v = obs.parse_nwis(json.dumps(nwis))
    assert list(v) == [1.25] and t[0] == datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
    ogc = {"features": [{"properties": {"time": "2026-10-20T12:00:00+00:00", "value": "0.5"}}],
           "links": [{"rel": "next", "href": "https://x/next"}]}
    t, v, nxt = obs.parse_ogc(json.dumps(ogc))
    assert list(v) == [0.5] and nxt == "https://x/next"


def test_tide_fit_json_round_trip():
    t, v = series(24 * 60, step_min=60, f=lambda h: 1.3 + np.cos(2 * np.pi * h / 12.4206) + 0.3 * np.cos(2 * np.pi * h / 23.9345))
    coef = tide.fit(t, v, 37.0)
    doc = json.loads(json.dumps(tide.to_json(coef, station="TEST")))
    back = tide.from_json(doc)
    ahead = [T0 + timedelta(days=70, hours=k) for k in range(48)]
    assert np.allclose(tide.predict(coef, ahead), tide.predict(back, ahead), atol=1e-9)
    h = np.array([(x - T0).total_seconds() / 3600 for x in ahead])
    truth = 1.3 + np.cos(2 * np.pi * h / 12.4206) + 0.3 * np.cos(2 * np.pi * h / 23.9345)
    assert np.sqrt(np.mean((tide.predict(back, ahead) - truth) ** 2)) < 0.05


def test_station_table():
    table = obs.stations()
    assert {"SWPT", "BRDH", "USLN-X", "CNRC"} <= set(table)
    assert not any(st.startswith(("NN", "VB")) or st in ("USNW", "USPF", "USSV") for st in table)
    assert all(r["navd88"] is not None for r in table.values())
    assert table["USNR"]["alias_of"] == "USSF"


def test_surge_blend_starts_at_gauge_and_relaxes_to_members():
    t0 = T0 + timedelta(hours=24)
    grid = np.array([T0 + timedelta(minutes=6 * i) for i in range(961)])  # t0-24 h .. t0+72 h
    lead = surge.lead_hours(grid, t0)
    tide_ = 1.3 + np.cos(2 * np.pi * lead / 12.42)
    observed = np.where(lead <= 0, tide_ + 0.8, np.nan)          # 0.8 ft residual now
    cols = {"observed": observed, "petss": tide_ + 0.2, "stofs2d": tide_ + 0.4}
    out, notes = surge.forecast(grid, t0, cols, tide_)
    f = out["stormsense"] - tide_
    near, far = (lead > 0) & (lead < 0.3), lead > 70
    assert abs(np.nanmean(f[near]) - 0.8) < 0.02                 # starts from the gauge
    assert abs(np.nanmean(f[far]) - 0.3) < 0.02                  # equal weights: mean of 0.2 and 0.4
    assert np.all(out["stormsense_p10"][lead > 0] < out["stormsense"][lead > 0])
    skill = {"petss": {"0-12": 0.1, "12-36": 0.1, "36-72": 0.1}, "stofs2d": {"0-12": 0.3, "12-36": 0.3, "36-72": 0.3}}
    f2 = surge.forecast(grid, t0, cols, tide_, skill)[0]["stormsense"] - tide_
    assert abs(np.nanmean(f2[far]) - 0.22) < 0.01                # 1/RMSE^2 weights favour P-ETSS 9:1
    assert notes["members"] == ["petss", "stofs2d"]


def test_surge_without_members_is_persistence():
    t0 = T0 + timedelta(hours=24)
    grid = np.array([T0 + timedelta(minutes=6 * i) for i in range(961)])
    lead = surge.lead_hours(grid, t0)
    tide_ = np.zeros(len(grid))
    out, _ = surge.forecast(grid, t0, {"observed": np.where(lead <= 0, 0.5, np.nan)}, tide_)
    k = np.argmin(np.abs(lead - 24))
    assert abs(out["stormsense"][k] - 0.5 / np.e) < 0.01


def _past_runs(scale, noise, n=12, seed=1):
    """Synthetic earlier runs at one gauge: the predicted surge is `scale` times what happened."""
    rng = np.random.default_rng(seed)
    runs, truth = {}, {}
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for i in range(n):
        t0 = start + timedelta(hours=6 * i)
        grid = np.array([t0 - timedelta(hours=24) + timedelta(minutes=6 * k) for k in range(961)])
        lead = surge.lead_hours(grid, t0)
        hrs = np.array([(t - start).total_seconds() / 3600 for t in grid])
        tide_ = np.cos(2 * np.pi * hrs / 12.42)
        real = 0.8 * np.sin(2 * np.pi * hrs / 96.0)  # a slow surge, different in each run's window
        for t, v in zip(grid, tide_ + real + noise * rng.standard_normal(len(grid))):
            truth[t] = v
        pred = np.where(lead > 0, tide_ + scale * real, np.nan)
        runs[f"r{i:02d}"] = {"TEST": (grid, {"observed": np.where(lead <= 0, tide_ + real, np.nan), "pred": pred, "tide": tide_})}
    now = start + timedelta(hours=6 * n + 72)
    return runs, {"TEST": truth}, now


def _predict(grid, t0, cols):
    return cols["pred"], cols["tide"]


def test_corrector_learns_a_surge_scaling():
    runs, truth, now = _past_runs(scale=1.5, noise=0.02)
    coefs, info = corrector.correction_for("TEST", runs, truth, now, _predict)
    assert info["applied"]
    for band in ("12-36", "36-72"):
        a, b = coefs[band]
        assert abs(info["bands"][band]["fit_b"] + 1 / 3) < 0.02  # truth = pred / 1.5: b = -1/3 ...
        assert -0.3 < b < -0.15 and abs(a) < 0.05                # ... applied shrunk by n / (n + K)


def test_corrector_leaves_noise_alone():
    runs, truth, now = _past_runs(scale=1.0, noise=0.3)
    coefs, info = corrector.correction_for("TEST", runs, truth, now, _predict)
    assert all(abs(a) < 0.03 and abs(b) < 0.05 for a, b in coefs.values())


def test_corrector_needs_enough_runs_and_starts_at_the_gauge():
    runs, truth, now = _past_runs(scale=1.5, noise=0.02, n=3)
    coefs, info = corrector.correction_for("TEST", runs, truth, now, _predict)
    assert not info["applied"]
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    grid = np.array([t0 + timedelta(minutes=6 * k) for k in range(721)])
    c = corrector.correction(grid, t0, np.full(len(grid), 2.0), {"0-12": (0.25, 0.5), "12-36": (0.25, 0.5), "36-72": (0.25, 0.5)})
    assert c[0] == 0 and abs(c[-1] - corrector.CAP_FT) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
