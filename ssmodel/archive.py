"""Archived StormSense Ensemble runs: download them from the StormSense-ensemble releases, read
their MLLW ens72 CSVs, and stitch the observations into a truth record. Used by the
predictor-corrector (live, inside the ensemble run) and by the hindcasts."""
import csv
import io
import json
import os
import re
import tarfile
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

REPO = "DerekLoftis/StormSense-ensemble"
UA = {"User-Agent": "StormSense forecast model (VIMS)"}
NAME = re.compile(r"ensemble_(\d{4}-\d\d-\d\dT\d\dZ)(\.tar\.gz)?$")


def parse_t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def cycle(name):
    return datetime.strptime(NAME.search(os.path.basename(name)).group(1), "%Y-%m-%dT%HZ").replace(tzinfo=timezone.utc)


def _get(url):
    head = dict(UA)
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok and url.startswith("https://api.github.com/"):
        head["Authorization"] = f"Bearer {tok}"
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=head), timeout=120) as r:
                return r.read()
        except Exception:
            if i == 2:
                raise
            time.sleep(5 * (i + 1))


def download_recent(folder, now, days=8, repo=REPO):
    """Fetch the run tarballs of the last `days` from the ensemble-YYYY-MM releases into folder
    (kept if already there). Returns their paths, oldest first."""
    os.makedirs(folder, exist_ok=True)
    since = now - timedelta(days=days)
    paths = []
    for rel in json.loads(_get(f"https://api.github.com/repos/{repo}/releases?per_page=6")):
        if not rel["tag_name"].startswith("ensemble-"):
            continue
        for a in rel["assets"]:
            if NAME.match(a["name"]) and since <= cycle(a["name"]) < now:
                p = os.path.join(folder, a["name"])
                if not os.path.exists(p):
                    open(p, "wb").write(_get(a["browser_download_url"]))
                paths.append(p)
    return sorted(set(paths), key=cycle)


def read_run(path):
    """{station: (grid, cols)} from one run's MLLW ens72 CSVs (a tarball or an extracted folder);
    cols maps the column name without _ft to its values, NaN for blanks."""
    files = {}
    if path.endswith(".tar.gz"):
        with tarfile.open(path) as t:
            for m in t.getmembers():
                if m.name.endswith("_ens72_MLLW.csv"):
                    files[os.path.basename(m.name)] = t.extractfile(m).read().decode()
    else:
        d = os.path.join(path, "ens72")
        for n in os.listdir(d):
            if n.endswith("_ens72_MLLW.csv"):
                files[n] = open(os.path.join(d, n)).read()
    by_st = {}
    for n, text in files.items():
        rows = list(csv.reader(io.StringIO(text)))
        head, rows = rows[0], rows[1:]
        grid = np.array([parse_t(r[0]) for r in rows])
        cols = {}
        for j, h in enumerate(head[1:], 1):
            v = np.array([float(r[j]) if r[j] else np.nan for r in rows])
            if not np.isnan(v).all():
                cols[h[:-3]] = v
        by_st[n.split("_ens72_")[0]] = (grid, cols)
    return by_st


def read_runs(paths):
    """{cycle string: {station: (grid, cols)}} for tarballs or folders named ensemble_<cycle>."""
    return {NAME.search(os.path.basename(p.rstrip("/"))).group(1): read_run(p) for p in paths}


def read_folder(folder):
    return read_runs([os.path.join(folder, n) for n in sorted(os.listdir(folder)) if NAME.match(n)])


def truth_series(runs, extra=None):
    """{station: {time: observed ft}} stitched across runs, the newest run winning; extra
    ({station: (grid, observed)}, e.g. the run being built) wins over all of them."""
    out = defaultdict(dict)
    for cyc in sorted(runs):
        for st, (grid, cols) in runs[cyc].items():
            if "observed" in cols:
                for t, v in zip(grid, cols["observed"]):
                    if not np.isnan(v):
                        out[st][t] = v
    for st, (grid, ob) in (extra or {}).items():
        for t, v in zip(grid, ob):
            if not np.isnan(v):
                out[st][t] = v
    return out


def t0_of(grid, cols):
    seen = np.where(~np.isnan(cols.get("observed", np.full(len(grid), np.nan))))[0]
    return grid[seen[-1]] if len(seen) else None
