# StormSense forecast model

StormSense's own 72-hour storm tide forecast for the Virginia and Chesapeake Bay gauges, built
from its own tide, its own observations and a surge model trained on the StormSense, ensemble
and Tidewatch archives. It runs as the `stormsense` member of
[StormSense Ensemble](https://github.com/DerekLoftis/StormSense-ensemble) and becomes the primary
StormSense forecast only once it scores best.

| Part | Module | Status |
|---|---|---|
| Observations straight from each gauge (CO-OPS, USGS, VIMS THREDDS, FloodVISION, GreenStream), quality-controlled, ft MLLW | `ssmodel/obs.py` | built |
| Harmonic tide per gauge (UTide, 27 constituents), `harmonics/<ST>.json` | `ssmodel/tide.py` | built |
| Surge v1: recent residual handed over to a skill-weighted blend of the member surges | `ssmodel/surge.py` | built; hindcast in `hindcast/` |
| Surge v2: learned correction trained on 2023-2026 backfill | `ssmodel/train.py` | later |

`ssmodel/stations.csv` lists the gauges, their sources and NAVD88 above MLLW. Retired stations
(NN01-NN10, VB01-VB10, USNW, USPF, USSV) are not in it. USNR is the same USGS site as USSF; GS01
and GS02 need a GreenStream key and are not fetched yet.

    pip install -r requirements.txt
    python -m pytest tests                          # offline tests
    python -m ssmodel.obs work/obs --days 3         # observations, one CSV per gauge
    python -m ssmodel.tide fit --stations SWPT      # refit a gauge's tide

Workflows: `ci.yml` (tests, plus a live check of every observation source, as annotations) and
`harmonics.yml` (monthly refit, committed to the repo).

The plan: https://claude.ai/code/artifact/10e684a7-605c-4560-8356-41d90ad31543
