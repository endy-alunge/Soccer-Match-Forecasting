"""CSV contracts and pre-cutoff features. Never uses target-match statistics."""
from __future__ import annotations
from io import BytesIO
import hashlib
import numpy as np
import pandas as pd

FORM = [f"{side}_{stat}_last_3" for side in ("home", "away")
        for stat in ("points", "goals_scored", "goals_conceded")]
XG = [f"{side}_{stat}_last_3" for side in ("home", "away")
      for stat in ("xg_for", "xg_against")]
MARKETS = {"1X2": ["H", "D", "A"], "O/U 1.5": ["Under", "Over"], "BTTS": ["No", "Yes"]}
KEYS = ["league", "fixture_id"]
TIMEZONES = {"Premier League": "Europe/London", "LaLiga": "Europe/Madrid"}

def utc(value):
    return pd.to_datetime(value, utc=True, errors="raise")

def read_csv(payload):
    return pd.read_csv(BytesIO(payload), encoding="utf-8-sig")

def normalize(frame, league="", season="", timezone="UTC", completed=True):
    """Accept canonical CSV or Football-Data headers; fail closed on ambiguous inputs."""
    f = frame.copy()
    f.columns = f.columns.str.strip()
    f = f.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team",
                          "FTHG": "home_goals", "FTAG": "away_goals"})
    if "kickoff" not in f:
        if "Date" not in f:
            raise ValueError("Need kickoff (ISO 8601), or Football-Data Date and Time.")
        dates = pd.to_datetime(f["Date"].astype(str) + " " +
                              f.get("Time", pd.Series("00:00", index=f.index)).fillna("00:00").astype(str),
                              dayfirst=True, errors="raise")
        f["kickoff"] = dates.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    else:
        # Canonical timestamp contract: UTC when no explicit offset is given.
        f["kickoff"] = pd.to_datetime(f.kickoff, utc=True, errors="raise")
    for col, default in [("league", league), ("season", season)]:
        if col not in f:
            f[col] = default
    for col in ["league", "season", "home_team", "away_team"]:
        if col not in f or f[col].isna().any():
            raise ValueError(f"Missing {col}.")
        f[col] = f[col].astype(str).str.strip()
        if f[col].eq("").any() or f[col].str.contains("|", regex=False).any():
            raise ValueError(f"{col}: blank values and pipe characters are not allowed.")
    if f.home_team.eq(f.away_team).any():
        raise ValueError("A team cannot play itself.")
    if "fixture_id" not in f:
        f["fixture_id"] = f.season + "|" + f.home_team + "|" + f.away_team
    if f.fixture_id.isna().any() or f.fixture_id.astype(str).str.strip().eq("").any():
        raise ValueError("Blank fixture ID.")
    f["fixture_id"] = f.fixture_id.astype(str)
    if f.duplicated(KEYS).any():
        raise ValueError("Duplicate league + fixture_id. Provide unique IDs for repeat fixtures.")
    if f.kickoff.isna().any():
        raise ValueError("Missing kickoff.")
    for col in ["home_goals", "away_goals"]:
        if completed:
            if col not in f:
                raise ValueError(f"Missing {col} for completed matches.")
            f[col] = pd.to_numeric(f[col], errors="raise")
            if f[col].isna().any() or (~np.isfinite(f[col])).any() or (f[col] < 0).any() or (f[col] % 1 != 0).any():
                raise ValueError(f"{col} must contain nonnegative integers.")
            f[col] = f[col].astype(int)
        else:
            if col in f and f[col].notna().any():
                raise ValueError("Upcoming fixtures must not contain scores. Use completed-match import.")
            f[col] = np.nan
    if completed and "FTR" in f:
        expected = np.where(f.home_goals > f.away_goals, "H", np.where(f.home_goals < f.away_goals, "A", "D"))
        if not np.all(f.FTR == expected):
            raise ValueError("FTR disagrees with goals.")
    f["available_at"] = pd.to_datetime(f["available_at"], utc=True) if "available_at" in f else f.kickoff + pd.Timedelta(hours=3)
    if f.available_at.isna().any() or (f.available_at < f.kickoff).any():
        raise ValueError("available_at must be at or after kickoff.")
    for col in ["home_xg", "away_xg"]:
        f[col] = pd.to_numeric(f[col], errors="raise") if col in f else np.nan
        v = f[col].dropna()
        if (~np.isfinite(v)).any() or (v < 0).any():
            raise ValueError("xG must be nonnegative and finite, or blank.")
    if f.home_xg.isna().ne(f.away_xg.isna()).any():
        raise ValueError("Supply both team xG values, or neither.")
    if f.home_xg.notna().any() and ("provider" not in f or f.loc[f.home_xg.notna(), "provider"].fillna("").eq("").any()):
        raise ValueError("xG rows require a provider label.")
    f["xg_available_at"] = pd.to_datetime(f["xg_available_at"], utc=True) if "xg_available_at" in f else f.available_at
    if (f.xg_available_at < f.kickoff).any():
        raise ValueError("xg_available_at cannot precede kickoff.")
    for col in ["provider", "source_url"]:
        f[col] = f[col].fillna("").astype(str) if col in f else ""
    # Preserve only documented exact-market decimal odds.
    odds = ["AvgCH", "AvgCD", "AvgCA", "odds_under_1_5", "odds_over_1_5", "odds_btts_no", "odds_btts_yes"]
    for col in odds:
        f[col] = pd.to_numeric(f[col], errors="coerce") if col in f else np.nan
        if ((f[col].notna()) & ((f[col] <= 1) | ~np.isfinite(f[col]))).any():
            raise ValueError(f"{col} needs decimal odds greater than 1 or blank.")
    keep = KEYS + ["season", "kickoff", "home_team", "away_team", "home_goals", "away_goals",
                   "available_at", "home_xg", "away_xg", "xg_available_at", "provider", "source_url"] + odds
    return f[keep].sort_values(["kickoff", *KEYS]).reset_index(drop=True)

def features(history, targets, cutoff=None, use_xg=False):
    """Last three completed fixtures, across seasons. Same UTC day is excluded.
    For historical training each target's kickoff is its own forecast cutoff.
    For a live batch the common as-of cutoff also limits all inputs.
    """
    rows = []
    for _, t in targets.iterrows():
        boundary = min(utc(t.kickoff), utc(cutoff)) if cutoff is not None else utc(t.kickoff)
        row = t.to_dict()
        reasons = []
        for side in ("home", "away"):
            h = history.loc[
                history.league.eq(t.league) & history.home_goals.notna()
                & (history.kickoff.dt.normalize() < boundary.normalize())
                & (history.available_at < boundary)
                & (history.home_team.eq(t[f"{side}_team"]) | history.away_team.eq(t[f"{side}_team"]))
            ].sort_values(["kickoff", "fixture_id"]).tail(3)
            row[f"{side}_prior_count"] = len(h)
            names = ("points", "goals_scored", "goals_conceded", "xg_for", "xg_against")
            values = {n: np.nan for n in names}
            if len(h) == 3:
                home = h.home_team.eq(t[f"{side}_team"])
                scored = np.where(home, h.home_goals, h.away_goals)
                conceded = np.where(home, h.away_goals, h.home_goals)
                values.update(points=float(np.where(scored > conceded, 3, np.where(scored == conceded, 1, 0)).sum()),
                              goals_scored=float(scored.sum()), goals_conceded=float(conceded.sum()))
                if h[["home_xg", "away_xg"]].notna().all().all() and (h.xg_available_at < boundary).all():
                    values.update(xg_for=float(np.where(home, h.home_xg, h.away_xg).sum()),
                                  xg_against=float(np.where(home, h.away_xg, h.home_xg).sum()))
                elif use_xg:
                    reasons.append(f"{side}: missing/unavailable xG in last three fixtures")
            else:
                reasons.append(f"{side}: needs three earlier completed fixtures")
            for n, v in values.items():
                row[f"{side}_{n}_last_3"] = v
        row["ready"] = not reasons
        row["readiness_reason"] = "; ".join(reasons) or "Ready"
        rows.append(row)
    return pd.DataFrame(rows)

def labels(frame, market):
    h, a = frame.home_goals.to_numpy(), frame.away_goals.to_numpy()
    if market == "1X2":
        return np.where(h > a, 0, np.where(h == a, 1, 2))
    if market == "O/U 1.5":
        return (h + a >= 2).astype(int)
    return ((h > 0) & (a > 0)).astype(int)

def fingerprint(frame):
    return hashlib.sha256(frame.sort_values(KEYS).to_json(orient="records", date_format="iso").encode()).hexdigest()
