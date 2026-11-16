# Verification

Tested 16 November 2026 with the supplied 380-match Premier League 2023/24 CSV.

Result: **13 tests passed**.

Covered:
- direct reading of the attached CSV format;
- chronological partitions with no date overlap;
- invariance of features to changed future match results;
- invalid scores, missing columns, duplicates and scored upcoming rows;
- probability normalization for all three markets;
- TreeSHAP additivity against raw margins;
- native model saving/loading and identical restored predictions;
- unknown-team abstention, old dates and wrong leagues;
- the empty two-tab application;
- all markets in the populated results screen;
- the complete UI workflow: upload historical CSV, train, upload fixture CSV,
  predict and switch all market visuals.

Tests exercise the actual application, not research notebooks.
The upcoming example is illustrative and retrospective, not a real fixture list.
The UI was tested with Streamlit AppTest; a full browser visual inspection and
Windows installation have not been performed here.

Environment: Linux, Python 3.12.14.
Direct dependency versions are in requirements-tested.txt.
One unrelated pytest plugin warning concerns anyio's deprecated pytest alias.

Small-data update: training now accepts exactly 70 feature-ready matches. Tests cover a complete fit at that boundary and rejection below it. Date groups stay intact with minimum block sizes 30/10/10. Loading existing models uses their saved test interval, not the revised splitting rule.
