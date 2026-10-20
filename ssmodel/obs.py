"""Gauge observations for the StormSense model, straight from each gauge's own source
(no Tidewatch/web4), in feet above the station's MLLW, quality-controlled.

    NOAA CO-OPS      6-minute water level (MLLW)            api.tidesandcurrents.noaa.gov
    USGS NWIS        instantaneous 62620, ft NAVD88         waterservices.usgs.gov (new OGC API as fallback)
    VIMS THREDDS     BRDH radar gauge, 1-minute             tds.vims.edu (MLLW = 9.71 - value)
    archives         USLN-X (FloodVISION-AI), CNRC (GreenStream), ft NAVD88, kept in tidewatch_latest

NAVD88 heights go to MLLW with stations.csv's navd88_above_mllw_ft (NOAA published datums, or
VDatum at USGS and VIMS gauges, as the StormSense styler uses).

    python -m ssmodel.obs OUT_DIR [--days 3] [--stations SWPT,USJT]   one CSV per station
"""
import csv
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "StormSense forecast model (VIMS)"}
COOPS = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?application=VIMS_StormSense&units=english"
         "&time_zone=gmt&format=json&datum=MLLW&station={id}&product={product}&begin_date={a}&end_date={b}")
NWIS = ("https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site}&parameterCd={pc}"
        "&startDT={a}&endDT={b}&siteStatus=all")
OGC = ("https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items?f=json&limit=10000"
       "&monitoring_location_id=USGS-{site}&parameter_code={pc}&datetime={a}/{b}")
RAW = "https://raw.githubusercontent.com/DerekLoftis/tidewatch_latest/main/"
ARCHIVE_COLUMN = {"floodvision": ("ml_ft_navd88", "v2"), "greenstream": ("level_ft_navd88", None)}
GAUGE_MLLW = {"BRDH": 9.71}  # VIMS radar gauges: MLLW lies this far below the sensor (ft)
NAVD_PARAM = "62620"  # estuary or ocean water surface elevation above NAVD88, ft
DEFAULT_LAT = 37.0  # for the few USGS sites stations.csv has no position for (nodal factors only)


def get(url, tries=3, timeout=90):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(4 * (i + 1))


def stations():
    """{station: row} from stations.csv, with navd88 as a float (None when blank)."""
    out = {}
    for r in csv.DictReader(open(os.path.join(HERE, "stations.csv"))):
        r["navd88"] = float(r["navd88_above_mllw_ft"]) if r["navd88_above_mllw_ft"] else None
        r["lat_f"] = float(r["lat"]) if r["lat"] else DEFAULT_LAT
        out[r["station"]] = r
    return out


def utc(s):
    """Datetime from an ISO-ish string with or without an offset, as UTC."""
    s = s.replace("Z", "+00:00")
    t = datetime.fromisoformat(s)
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _chunks(a, b, days):
    while a < b:
        c = min(b, a + timedelta(days=days))
        yield a, c
        a = c


# --- sources ---------------------------------------------------------------------
def coops(noaa_id, a, b, product="water_level"):
    """CO-OPS heights ft above MLLW between a and b (6-minute water_level, or hourly_height)."""
    days = 30 if product == "water_level" else 360
    t_all, v_all = [], []
    for x, y in _chunks(a, b, days):
        url = COOPS.format(id=noaa_id, product=product, a=x.strftime("%Y%m%d %H:%M"), b=y.strftime("%Y%m%d %H:%M"))
        js = json.loads(get(url.replace(" ", "%20")))
        if "error" in js:
            if "No data" in str(js["error"]):
                continue
            raise RuntimeError(js["error"].get("message", js["error"]))
        for r in js.get("data", []):
            if r.get("v") not in ("", None):
                t_all.append(datetime.strptime(r["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc))
                v_all.append(float(r["v"]))
    return np.array(t_all), np.array(v_all)


def coops_predictions(noaa_id, a, b):
    """NOAA's published 6-minute tide predictions, ft MLLW (to check our own harmonic fits)."""
    t_all, v_all = [], []
    for x, y in _chunks(a, b, 30):
        url = COOPS.format(id=noaa_id, product="predictions", a=x.strftime("%Y%m%d %H:%M"), b=y.strftime("%Y%m%d %H:%M"))
        for r in json.loads(get(url.replace(" ", "%20"))).get("predictions", []):
            t_all.append(datetime.strptime(r["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc))
            v_all.append(float(r["v"]))
    return np.array(t_all), np.array(v_all)


def parse_nwis(raw, pc=NAVD_PARAM):
    """(times, values) of one parameter from an NWIS IV JSON reply (first series that has data)."""
    for ts in json.loads(raw).get("value", {}).get("timeSeries", []):
        if ts["variable"]["variableCode"][0]["value"] != pc:
            continue
        nodata = ts["variable"].get("noDataValue")
        for block in ts.get("values", []):
            pts = [(utc(p["dateTime"]), float(p["value"])) for p in block.get("value", [])
                   if p.get("value") not in (None, "") and float(p["value"]) != nodata]
            if pts:
                t, v = zip(*pts)
                return np.array(t), np.array(v)
    return np.array([]), np.array([])


def parse_ogc(raw):
    """(times, values, next link) from one page of the USGS OGC API continuous collection."""
    js = json.loads(raw)
    pts = [(utc(f["properties"]["time"]), float(f["properties"]["value"])) for f in js.get("features", [])
           if f["properties"].get("value") not in (None, "")]
    nxt = next((ln["href"] for ln in js.get("links", []) if ln.get("rel") == "next"), None)
    t, v = (zip(*pts) if pts else ((), ()))
    return np.array(t), np.array(v), nxt


def nwis(site, a, b, pc=NAVD_PARAM):
    """USGS instantaneous values (ft NAVD88 for 62620) between a and b; the legacy NWIS service
    first, the OGC API if that fails (NWIS is being retired)."""
    fmt = "%Y-%m-%dT%H:%MZ"
    try:
        t_all, v_all = [], []
        for x, y in _chunks(a, b, 120):
            t, v = parse_nwis(get(NWIS.format(site=site, pc=pc, a=x.strftime(fmt), b=y.strftime(fmt))), pc)
            t_all += list(t)
            v_all += list(v)
        return np.array(t_all), np.array(v_all)
    except Exception as e:
        legacy = e
    t_all, v_all = [], []
    url = OGC.format(site=site, pc=pc, a=a.strftime("%Y-%m-%dT%H:%M:%SZ"), b=b.strftime("%Y-%m-%dT%H:%M:%SZ"))
    try:
        while url:
            t, v, url = parse_ogc(get(url))
            t_all += list(t)
            v_all += list(v)
    except Exception as e:
        raise RuntimeError(f"NWIS failed ({legacy}); OGC API failed ({e})")
    o = np.argsort(np.array([x.timestamp() for x in t_all]))
    return np.array(t_all)[o], np.array(v_all)[o]


def thredds(url, gauge_mllw, a, b):
    """A VIMS radar gauge's 1-minute record over OPeNDAP between a and b, ft MLLW."""
    dds = get(url + ".dds").decode()
    n = int(re.search(r"time\[time = (\d+)\]", dds).group(1))
    last = float(get(url + f".ascii?time%5B{n - 1}:1:{n - 1}%5D").decode().split()[-1].strip(","))
    end = datetime.fromtimestamp(last, timezone.utc)
    i1 = max(0, n - 1 - int((end - b).total_seconds() // 60 * 0.95)) if b < end else n - 1
    i0 = max(0, i1 - int((b - a).total_seconds() // 60 * 1.2))
    rng = f"%5B{i0}:1:{i1}%5D"  # Tomcat refuses raw brackets in a query
    txt = get(url + f".ascii?time{rng},values{rng}", timeout=300).decode()
    cols = []
    for name in ("time", "values"):
        m = re.search(r"^" + name + r"\[\d+\]\s*\n(.*?)(?:\n\s*\n|\Z)", txt, re.S | re.M)
        cols.append(np.array([float(x) for x in m.group(1).replace("\n", ",").split(",") if x.strip()]))
    secs, dist = cols
    ok = (secs > 0) & (dist > -900)
    t = np.array([datetime.fromtimestamp(x, timezone.utc).replace(second=0, microsecond=0) for x in secs[ok]])
    v = gauge_mllw - dist[ok]
    sel = (t >= a) & (t <= b)
    return t[sel], v[sel]


def archive(source_id, a, b):
    """USLN-X / CNRC monthly archive CSVs in tidewatch_latest, ft NAVD88 between a and b."""
    folder, st = source_id.split("/")
    col, model = ARCHIVE_COLUMN[folder]
    rows = {}
    y, m = a.year, a.month
    while (y, m) <= (b.year, b.month):
        try:
            text = get(f"{RAW}{folder}/{st}_{y:04d}-{m:02d}.csv", tries=1).decode()
        except Exception:
            text = ""
        for r in csv.DictReader(io.StringIO(text)):
            if r.get(col) and (model is None or r.get("model", model) == model):
                rows[r["time_utc"]] = float(r[col])
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    t = np.array([utc(k) for k in sorted(rows)])
    v = np.array([rows[k] for k in sorted(rows)])
    if not len(t):
        return t, v
    sel = (t >= a) & (t <= b)
    return t[sel], v[sel]


# --- quality control ---------------------------------------------------------------
def despike(t, v, window=15, tol=0.5):
    """Drop values more than tol ft off the running median of `window` points."""
    if len(v) <= window:
        return t, v
    import pandas as pd
    med = pd.Series(v).rolling(window, center=True, min_periods=max(3, window // 3)).median().to_numpy()
    keep = np.abs(v - med) <= tol
    return t[keep], v[keep]


def flatline(t, v, max_same=timedelta(minutes=48)):
    """Drop runs where the value doesn't change for longer than max_same: a tidal gauge always
    moves, so a held value is a stuck or repeated reading (USBB repeats values in ~18-minute steps)."""
    if len(v) < 3:
        return t, v
    keep = np.ones(len(v), bool)
    start = 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[start]:
            if t[i - 1] - t[start] > max_same:
                keep[start:i] = False
            start = i
    return t[keep], v[keep]


def qc(t, v, lo=-4.0, hi=14.0):
    """Sorted, de-duplicated, in a plausible range (ft MLLW), despiked, no stuck runs."""
    if not len(t):
        return t, v
    o = np.argsort(np.array([x.timestamp() for x in t]), kind="stable")
    t, v = t[o], v[o]
    _, first = np.unique(np.array([x.timestamp() for x in t]), return_index=True)
    t, v = t[first], v[first]
    ok = np.isfinite(v) & (v > lo) & (v < hi)
    t, v = flatline(*despike(t[ok], v[ok]))
    return t, v


def on_six_minutes(t, v, max_gap=timedelta(minutes=30)):
    """Series on the 6-minute clock: values on it kept, others interpolated across gaps up to
    max_gap (USGS gauges report every 15 minutes, VIMS gauges every minute)."""
    if len(t) < 2:
        return t, v
    s = np.array([x.timestamp() for x in t])
    g0 = np.ceil(s[0] / 360) * 360
    grid = np.arange(g0, s[-1] + 1, 360)
    out = np.interp(grid, s, v)
    j = np.searchsorted(s, grid)
    j = np.clip(j, 1, len(s) - 1)
    gap = s[j] - s[j - 1]
    exact = np.isin(grid, s)
    ok = exact | (gap <= max_gap.total_seconds())
    tt = np.array([datetime.fromtimestamp(x, timezone.utc) for x in grid[ok]])
    return tt, out[ok]


# --- one station -------------------------------------------------------------------
def fetch(st, a, b, info=None):
    """(times, ft MLLW) for station st between a and b, quality-controlled, on the 6-minute clock.
    Raises if the station has no source of its own."""
    info = info or stations()[st]
    src, sid = info["source"], info["source_id"]
    if info["alias_of"]:
        raise ValueError(f"{st} is the same gauge as {info['alias_of']}")
    if src == "coops":
        t, v = coops(sid, a, b)
    elif src == "nwis":
        t, v = nwis(sid, a, b)
        v = v + info["navd88"]
    elif src == "thredds":
        t, v = thredds(sid, GAUGE_MLLW[st], a, b)
    elif src == "archive":
        t, v = archive(sid, a, b)
        v = v + info["navd88"]
    else:
        raise ValueError(f"{st}: no observation source ({info['note'] or 'none listed'})")
    return on_six_minutes(*qc(t, v))


def write_csv(path, t, v):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date_time_z", "water_level_ft_mllw"])
        w.writerows([x.strftime("%Y-%m-%dT%H:%MZ"), f"{y:.3f}"] for x, y in zip(t, v))


def read_csv(path):
    rows = list(csv.reader(open(path)))[1:]
    return (np.array([datetime.strptime(r[0], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc) for r in rows]),
            np.array([float(r[1]) for r in rows]))


def main(argv):
    out = argv[0]
    days = float(argv[argv.index("--days") + 1]) if "--days" in argv else 3.0
    table = stations()
    names = argv[argv.index("--stations") + 1].split(",") if "--stations" in argv else sorted(table)
    os.makedirs(out, exist_ok=True)
    b = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    a = b - timedelta(days=days)
    report = []
    for st in names:
        try:
            t, v = fetch(st, a, b, table[st])
        except Exception as e:
            report.append((st, table[st]["source"], 0, None, str(e)[:160]))
            continue
        if len(t):
            write_csv(os.path.join(out, f"{st}_obs.csv"), t, v)
        report.append((st, table[st]["source"], len(t), t[-1].strftime("%Y-%m-%dT%H:%MZ") if len(t) else None, ""))
    with open(os.path.join(out, "report.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "source", "n", "last", "error"])
        w.writerows(report)
    for r in report:
        print(",".join("" if x is None else str(x) for x in r))
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
