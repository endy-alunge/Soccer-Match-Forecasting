# Soccer Forecast — Simple two-phase application

## What you do

**Phase One:** upload a historical CSV → click **Train models & show results** → inspect charts.

**Phase Two:** upload an upcoming-fixture CSV → click **Predict upcoming matches** → inspect probabilities and download predictions.

There are no database imports, league-eligibility settings, manually entered cutoffs, or xG requirements. This version is separate from the original research prototype.

## Start in VS Code (Windows)

Extract this ZIP into a NEW folder and open soccer_simple in VS Code.

    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m streamlit run app.py

Python 3.12 is also suitable. Substitute py -3.12 if that is your installed interpreter.
If your .venv already exists, skip the first command.
The installation includes scikit-learn, required by XGBClassifier.

macOS / Linux:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m streamlit run app.py

Open the localhost address printed in the terminal, normally http://localhost:8501.
VS Code includes a launch configuration for Python debugging.

## Phase One input

Use your existing Football-Data-style CSV:
Date, HomeTeam, AwayTeam, FTHG, FTAG.

Date supports day-first dates such as 11/08/2023 or ISO 2023-08-11.
Div identifies the league if present. FTR, if present, must agree with goals.
Time is accepted but not used: this application works at calendar-day resolution.
All the other columns can remain in the file.

Upload multiple seasons together if you wish. Duplicated fixtures are rejected rather than counted twice. If multiple league codes are present, choose one league to train. Keep team names and Div codes consistent across files.

The included examples/historical_example.csv is the user's supplied 380-match Premier League 2023/24 CSV, not synthetic data.

## Automatic training

The application creates six features from each team's last three earlier matches:
home/away points, goals scored and goals conceded. It never uses the current fixture's scores, shots or corners as inputs. Same-day matches are excluded.

It splits feature-ready fixtures into approximately:
- 60% training;
- 20% calibration and model choice;
- 20% test.

All fixtures on the same calendar date stay in the same partition, so exact percentages vary.
At least 70 feature-ready fixtures are required. The date-preserving split must provide at least 30 training, 10 calibration and 10 test fixtures. The nearest feasible 60/20/20 split is chosen. A warning marks small calibration/test blocks as preliminary. A full season is optional.
Every market outcome must occur in the training block.

Models:
- XGBoost: 100 trees, depth 2, learning rate 0.05, seed 42.
- Dixon–Coles: team attack, defence, home advantage and low-score correction.
- Temperature scaling fitted on the calibration block.
- Historical-frequency baseline from the same feature-ready training rows.

Market labels:
- 1X2: H home win, D draw, A away win.
- O/U 1.5: Under means 0 or 1 total goals; Over means 2 or more.
- BTTS: Yes means both teams score.

A model/variant is chosen separately for each market using calibration log loss. Test results do not select the deployed variant. Calibration is also fitted on this development block, so the separate test block is essential for assessing generalization.

The models remain fitted on the training block. They are not secretly refitted on the test data, which would invalidate the saved calibration. Upcoming predictions use the uploaded completed history to construct current form inputs.

Upload updated history and train again to refresh the saved model. This is a new exploratory evaluation; repeatedly inspecting and changing settings does not create fresh confirmatory evidence.

## Phase One visuals

- Observed home/draw/away counts.
- Total goals distribution.
- Team goals and summary statistics.
- Log loss and accuracy comparisons for raw/calibrated models and baseline.
- Brier scores (sum over classes, including both binary outcomes).
- Chosen-model confusion matrix and reliability diagram with bin counts.
- Exact native TreeSHAP global importance and individual waterfall explanations.
- Optional 1X2 closing-odds comparison if AvgCH, AvgCD and AvgCA are available.

Descriptive charts use the entire uploaded dataset; model scores use only the test block.
The test simulation uses earlier completed test dates as history for later test fixtures. This is a rolling feature evaluation with fixed fitted models.

TreeSHAP explains XGBoost raw margins, even when the selected model is Dixon–Coles. It does not explain Dixon–Coles or directly add up to a calibrated probability. Contribution sums are checked against model margins. Binary explanations concern Over/Yes; 1X2 explanations concern class margins.

## Phase Two input

Required columns: Date, HomeTeam, AwayTeam.
Optional columns: Div, Time.
FTHG and FTAG must be absent or blank.

All fixture dates must be AFTER the last date in uploaded history. This avoids accidentally presenting old training/evaluation matches as future forecasts.
Team names must match exactly. A team without three earlier matches is shown as unavailable with a reason.
If Div is absent, the saved model's league is assumed.

Download a template directly from Phase Two; it uses known team spelling.
The example/template pairings and dates are illustrative, NOT official fixtures.
With the included 2023/24 history, the May 2024 upcoming example is a retrospective demonstration. For current forecasts, upload current completed-match history and genuinely upcoming fixtures.

Every target uses only the uploaded completed history. Results predicted for earlier upcoming fixtures are never treated as actual results for later ones.

The output shows all market probabilities and the most likely outcome per market.
“Most likely” is not a guaranteed result and is not the original research policy's validation-qualified betting selection. There are no automated bets or profitability claims.

Dixon–Coles uses neutral team effects for teams absent from its fitted training period; this is flagged if such teams have enough recent form to forecast.

## Persistence and files

Training automatically saves native XGBoost UBJ models, JSON metadata, and history under saved_models. The most recent successful model loads when the app restarts. No pickle is used. Do not load model files from untrusted sources.

Uploading a new historical file does not change the saved model until Train is clicked. The app labels the active model's history range. Forecast exports are CSV downloads; this simple version does not keep a SQLite forecast ledger.

Files:
- app.py: two-tab interface and charts;
- engine.py: CSV checks, feature engineering, training, prediction and persistence;
- models.py: XGBoost, Dixon–Coles, temperature scaling and TreeSHAP;
- examples/: your historical example and illustrative upcoming input;
- tests/: integration and regression tests;
- requirements.txt: dependencies.

This is a local single-user app. Run it in its own folder; it does not migrate or change the old prototype's workspace.
The SOCCER_SIMPLE_DATA environment variable can point saved models elsewhere.

## Tests

    .\.venv\Scripts\python.exe -m pytest -q

Tests fit the application models using the supplied historical example; they do not run the research notebooks. See TESTING.md for actual verification results.

Official API references:
https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
https://docs.streamlit.io/develop/api-reference/layout/st.tabs
https://xgboost.readthedocs.io/en/stable/prediction.html
