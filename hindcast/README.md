# Hindcasts of the StormSense forecast

`v1_2026-10-16_to_20_*`: the v1 blend (tau 12 h) replayed over the 17 StormSense Ensemble runs
archived 16-20 Oct 2026 (`tools/hindcast_ensemble.py`), scored with Derek's skill formula against
the observations stitched from later runs. `summary.csv` compares StormSense with each forecast
over exactly the same runs, stations, bands and hours; `stormsense_better` counts the pairs where
StormSense's RMSE is lower. Only four days and one high-water period: an early look, not a verdict.
Aliased gauges (DAHL, WADC, USSC, USAR, USCB, USNR, USSB, GS01, USBB) are left out.

`v1pc_2026-10-16_to_21_*`: the v1 blend plus the predictor-corrector (`ssmodel/corrector.py`)
over the 20 runs archived 16-21 Oct. Here `stormsense` is the corrected forecast and
`stormsense_v1` the blend alone; the corrector for each run is fitted only from earlier runs
verified by its start. Against the blend alone, on the same hours (mean RMSE, ft):

| band | blend | corrected |
|---|---|---|
| 0-12 h | 0.195 | 0.197 |
| 12-36 h | 0.313 | 0.304 |
| 36-72 h | 0.772 | 0.750 |

`offset_only_2026-10-16_to_21_summary.csv`: the same corrector restricted to a constant offset
per lead band (no surge scaling). It over-corrects after the 19-20 Oct high water (12-36 h
0.313 to 0.362 ft), which is why the corrector also fits the surge scaling and shrinks both.
