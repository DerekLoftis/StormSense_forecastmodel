"""Offline tests (no network): QC, the 6-minute clock, source parsers, and the tide fit's
JSON round trip.    python -m pytest tests  (or python tests/test_offline.py)"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssmodel import obs, tide  # noqa: E402

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
