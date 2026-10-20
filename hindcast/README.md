# Hindcasts of the StormSense forecast

`v1_2026-10-16_to_20_*`: the v1 blend (tau 12 h) replayed over the 17 StormSense Ensemble runs
archived 16-20 Oct 2026 (`tools/hindcast_ensemble.py`), scored with Derek's skill formula against
the observations stitched from later runs. `summary.csv` compares StormSense with each forecast
over exactly the same runs, stations, bands and hours; `stormsense_better` counts the pairs where
StormSense's RMSE is lower. Only four days and one high-water period: an early look, not a verdict.
Aliased gauges (DAHL, WADC, USSC, USAR, USCB, USNR, USSB, GS01, USBB) are left out.
