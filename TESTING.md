# Verification record

Verified on 5 November 2026, Linux, Python 3.12.14.

23 tests passed. The suite covers:
- canonical and Football-Data input validation and timezone conversion;
- exclusion of same-day and unavailable results;
- invariance to changing future scores;
- missing xG and mismatched providers;
- temperature scaling and probability normalization;
- Dixon–Coles score-grid mass;
- model save/load and live forecasting;
- raw-margin TreeSHAP additivity;
- rejection of pre-freeze forecasts and already-started live targets;
- one-pick selection, deterministic ties and abstention;
- immutable SQLite snapshots, duplicate-run rejection and result updates;
- exact-market bookmaker matching and changed-kickoff exclusion;
- insufficient training support and failed-bundle cleanup;
- xG joining and form+xG training;
- paired block interval behaviour;
- all six empty Streamlit screens;
- populated data, model, forecast, explanation and performance screens;
- generating and persisting a live batch through the Streamlit test API.

Tests trained synthetic models only. The research notebooks were not executed.
One non-project warning came from the installed anyio pytest plugin's deprecated pytest alias.

Tested direct dependencies:
- streamlit 1.66.1
- pandas 2.2.3
- numpy 2.3.5
- scipy 1.17.0
- xgboost 3.4.1
- plotly 7.1.0
- pytest 9.2.0

The server starts successfully on localhost. Streamlit AppTest exercised the UI.
A full browser screenshot check could not run because the available Chromium
process was blocked by the host's socket permissions. Windows/macOS installation
has not been tested here. Test on your VS Code environment before research use.

requirements.txt gives compatible major-version ranges. requirements-tested.txt
pins these direct dependencies for the tested environment; transitive packages
are resolved by pip and are not a complete reproducible lock file.
