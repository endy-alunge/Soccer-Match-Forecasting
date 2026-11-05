# MatchLab — Soccer Forecasting Prototype

A complete local Streamlit application for **Evaluating the Predictive Reliability, Probability Calibration, and Explainability of Machine Learning Frameworks in Soccer Match Forecasting**.

Supports **1X2, O/U 1.5 and BTTS** with XGBoost, Dixon–Coles, temperature scaling, chronological market selection, TreeSHAP, and immutable forecast tracking.

## Install in VS Code

Recommended: **Python 3.11 or 3.12, 64-bit**. Extract the ZIP and open the inner soccer_forecast folder (containing app.py) in VS Code.

Windows PowerShell:

    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m streamlit run app.py

macOS / Linux:

    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m streamlit run app.py

Open the URL printed in the terminal, normally **http://localhost:8501**.
Select the .venv interpreter in VS Code. The included “Debug MatchLab” launch configuration supports breakpoints (install Microsoft's Python and Python Debugger extensions). run_windows.bat and run.sh launch the app after installation.

No API keys, accounts, internet data feeds or external database server are needed at runtime.

## Try the whole workflow with synthetic data

1. Open **Overview → Load synthetic demo data**.
2. Open **Train & freeze**. Keep the five default dates and select the Demo leagues.
3. Click **Train and save frozen version**, then wait for completion.
4. Open **Forecasts**, choose the new version and **Live**, and generate the future demo fixtures.
5. Open **Explanations** for probability charts, TreeSHAP, batch feature importance and Dixon–Coles score grids.
6. To test evaluation immediately, return to Forecasts and choose **Historical replay**. Use 2025-08-01 as cutoff and 2025-08-14 as final date. Save this batch.
7. Open **Performance**, select that replay batch, and inspect scores against existing demo results.

All demo teams and scores are fictional. Demo scores are not research findings. History is generated up to yesterday and upcoming fixtures are seven days ahead. Demo leagues are separate from your real leagues.

## Load your research data

### Completed matches

Use **Data & readiness → Import matches → Completed matches / results**.

**Football-Data files** require Date, HomeTeam, AwayTeam, FTHG and FTAG. Time is optional; FTR is checked against goals if present. Select fallback league and season, for example Premier League / 2023/24. Import different seasons separately unless each CSV contains its own metadata. One fallback league/season applies to the entire upload batch.

Set the local timezone correctly: Europe/London for Premier League, Europe/Madrid for LaLiga. Missing Time means midnight local time.

**Canonical CSVs** require league, season, kickoff, home_team, away_team, home_goals, away_goals. Recommended: fixture_id and available_at. Kickoff is ISO 8601 UTC (e.g. 2026-11-14T15:00:00Z); timestamps without offsets are interpreted as UTC.

Auto-generated IDs use season|home_team|away_team. Supply unique fixture IDs for repeat pairings in a season. Match names must be identical across history, xG and upcoming fixtures; no fuzzy team matching is performed. Scores must be regulation full-time results, excluding extra time and shoot-outs.

Imports persist across browser sessions and restarts in SQLite.

### xG

Use **Attach xG** after importing completed matches. Required columns: fixture_id, home_xg, away_xg, provider. Include league when IDs are ambiguous. Optional: source_url, xg_available_at.

If your match_xg.csv lacks a provider column, add the accurate label from your documented source. Player-summed xG should be labelled as such. Do not invent source URLs or provider provenance.

Use one consistent provider definition per xG version. Upcoming matches do not need their own xG: inputs are the last three completed matches. Missing or incompatible lagged xG causes abstention, not silent substitution of a form-only model. Independently published team-total validation remains pending.

### Upcoming fixtures and results

Use **Upcoming fixtures** with league, season, fixture_id, kickoff, home_team and away_team. Scores must be absent or blank. The included templates contain illustrative dates and teams; replace them with your fixtures.

After the match, import a completed-match CSV with the same identity. Explicit correction checkboxes allow intentional changes to existing scores or kickoffs; normal uploads reject conflicts. A changed kickoff excludes the original snapshot from evaluation.

### Optional bookmaker odds

| Market | Outcome order | Columns |
| --- | --- | --- |
| 1X2 | H, D, A | AvgCH, AvgCD, AvgCA |
| O/U 1.5 | Under, Over | odds_under_1_5, odds_over_1_5 |
| BTTS | No, Yes | odds_btts_no, odds_btts_yes |

Import these with completed results. Complete finite decimal odds greater than 1 are required for all outcomes. Probabilities are normalized reciprocals. O/U 2.5 odds never substitute for O/U 1.5. Closing odds are a retrospective benchmark, not a forecast input.

## Training and freezing

The application follows the notebook recipe with explicit persistence and application checks:

- Six form features: home/away last-three sums of points, goals scored and conceded.
- Optional four xG features: home/away last-three sums of xG for and against.
- XGBoost: 100 trees, depth 2, learning rate 0.05, seed 42, one CPU thread.
- Dixon–Coles: centered attack/defence effects, home advantage, low-score correction and no time decay; fitted to earlier scores. Unseen teams use neutral effects and are flagged.
- Temperature scaling: 0.25–4, fitted only on calibration data.
- Candidates: raw/scaled XGBoost and raw/scaled Dixon–Coles.
- Choose each market's candidate by mean log-loss skill against its training-frequency baseline over two ranking windows.
- League eligibility requires positive skill in both windows.
- Common eligibility requires positive average league skill in both windows; underlying per-league forecasts stay the same.
- Threshold grid: 0.500–0.900 by 0.025. Choose maximum supported coverage meeting the target, otherwise supported fallback; no supported threshold means no selections.
- At most one eligible market per fixture: highest confidence with alphabetical market tie-breaking.
- Thresholds are pooled across included leagues per policy, as in the notebook. League market eligibility is separate.
- No refit after validation. Changing cutoffs creates a new version.

Default exclusive boundaries at 00:00 UTC:
training ends 2024-08-01; calibration ends 2024-12-01; ranking 1 ends 2025-01-15; ranking 2 ends 2025-02-15; threshold validation ends 2025-04-01.

Each partition needs the configured minimum ready-fixture count. Increase history rather than weakening support just to force a selection.

Saved versions contain XGBoost UBJ files plus JSON parameters, calibration, baselines, rankings, thresholds, feature order, dependency versions and a data hash. No pickle is used.

**No fitted real-data models ship with the app.** Train a version after importing your data. This is a new fit, not a claim that notebook results have been reproduced. The optional calibrated xG version is a new configuration distinct from the report's raw xG ablation.

## Live versus replay

**Live** uses current UTC time and unplayed future fixtures. Every target in a batch shares the same cutoff; it does not gain results from intervening games. Generate a new batch nearer kickoff when needed.

**Historical replay** uses a past cutoff and remains labelled retrospective. Targets may have existing scores, but those scores are excluded from their inputs. Forecasting before the saved policy's freeze date is blocked.

Both modes:
- exclude matches on the cutoff's UTC calendar day;
- require earlier result availability (default assumption: kickoff + 3 hours);
- require earlier xG availability for xG inputs;
- require three prior completed fixtures per team;
- preserve input features, probabilities, baselines, selection, model version, cutoff and explanations;
- record actual generation time and a source-data hash.

Three-hour availability is an assumption, not a verified publication timestamp. Supply actual availability timestamps where known. Current CSVs cannot reconstruct retrospectively revised provider data. Replay is not prospective validation.

## Explanations

Native XGBoost exact TreeSHAP is used through pred_contribs=True. The separate shap package is not required.

- Contributions plus bias are checked against raw margins.
- Binary explanations concern the positive outcome (Over or Yes) in log-odds.
- 1X2 class margins are jointly transformed by softmax.
- Raw and temperature-scaled XGBoost probabilities are shown separately.
- XGBoost SHAP never claims to explain a Dixon–Coles selection.
- Dixon–Coles displays its own expected goal rates and score distribution.
- Aggregate importance uses one saved batch, league and model version.
- Attribution describes model behaviour, not causality.

The app provides local and batch visuals; notebooks remain the source of cross-season explanation-stability research.

## Performance tracking

Choose a saved batch after importing its results. Outputs include:
- readiness, settled fixtures, selections and coverage on settled ready fixtures;
- selection hit rate, matched-frequency hit rate and paired difference;
- descriptive four-week block-bootstrap intervals when at least two blocks exist;
- per-league/market log loss and summed multiclass Brier score;
- exact-market odds-matched comparisons;
- reliability plots with bin counts;
- CSV exports of per-fixture scores and selections.

Unsettled matches are excluded, not counted as failures. Compare one batch at a time to avoid double-counting repeated predictions. Do not tune a version on these outcomes and describe the resulting scores as untouched test evidence. The prototype makes no guaranteed accuracy or profitability claims.

## Project and storage

| File | Purpose |
| --- | --- |
| app.py | Streamlit screens |
| soccer/data.py | CSV validation and lagged features |
| soccer/models.py | Models, calibration, TreeSHAP |
| soccer/training.py | Chronological fitting and frozen policies |
| soccer/forecast.py | Immutable forecast generation |
| soccer/storage.py | SQLite persistence |
| soccer/evaluation.py | Metrics, reliability and uncertainty |
| soccer/demo.py | Fictional data |
| tests/ | Regression and UI tests |
| examples/ | CSV templates |
| workspace/ | Created at runtime: database and model bundles |

Set SOCCER_DATA_DIR to change storage location. Back up the workspace directory with the application stopped. This is a local, single-user prototype, with no authentication or production hosting setup.

Export a model ZIP from Train & freeze. To move it, extract into workspace/models/<bundle_id>/ with manifest.json and the UBJ files together. Only load bundles you trust. Copy the database too if you want history and forecast records.

## Tests and debugging

Windows:

    .\.venv\Scripts\python.exe -m pytest -q

macOS/Linux:

    .venv/bin/python -m pytest -q

Tests use synthetic data and do not execute research notebooks. They fit a small synthetic version to verify chronology, serialization, SHAP and result matching.

UI errors include expandable tracebacks. Common causes are wrong fallback seasons, mismatched team names, unlabelled xG, insufficient validation fixtures, missing recent xG, a pre-freeze cutoff or already-started live targets.

Use a supported 64-bit interpreter if XGBoost installation fails. On macOS, XGBoost may need the OpenMP runtime. See TESTING.md for tested versions.

## Scope

Implemented: uploads, persistent data, optional xG, chronological training, calibration, market eligibility, thresholds, live/replay inference, explanations, result matching, metrics and exports.

Not implemented: automatic feeds, bookmaker accounts or betting, live odds, authentication, scheduled retraining, production deployment, independent external xG validation.

Official API references:
- https://docs.streamlit.io/develop/api-reference
- https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest
- https://xgboost.readthedocs.io/en/stable/prediction.html
- https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
