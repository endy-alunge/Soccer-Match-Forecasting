"""Local SQLite store. Forecast payloads are immutable snapshots; results are separate."""
import json
import sqlite3
from pathlib import Path
import pandas as pd

def records(frame):
    return json.loads(frame.to_json(orient="records",date_format="iso"))

class Store:
    def __init__(self,path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS matches(
              league TEXT NOT NULL, fixture_id TEXT NOT NULL, payload TEXT NOT NULL,
              PRIMARY KEY(league,fixture_id));
            CREATE TABLE IF NOT EXISTS runs(
              run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, bundle_id TEXT NOT NULL,
              mode TEXT NOT NULL, payload TEXT NOT NULL);
            """)
    def connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def import_matches(self,frame,allow_corrections=False):
        new = records(frame)
        # Entire import commits or rolls back; no partly imported files.
        with self.connect() as con:
            for row in new:
                old = con.execute("SELECT payload FROM matches WHERE league=? AND fixture_id=?",
                                  (row["league"],row["fixture_id"])).fetchone()
                if old:
                    old = json.loads(old[0])
                    for key in ["home_team","away_team","season"]:
                        if old[key] != row[key]:
                            raise ValueError(f"Fixture identity conflict for {row['fixture_id']}: {key}.")
                    if old["home_goals"] is not None and row["home_goals"] is None:
                        raise ValueError("Cannot replace a completed result with an upcoming fixture.")
                    conflict = any(old[k] != row[k] for k in ["kickoff","home_goals","away_goals"]
                                   if old[k] is not None)
                    if conflict and not allow_corrections:
                        raise ValueError("Existing kickoff/result differs. Enable explicit corrections if intended.")
                    # A result-only update must not silently erase previously supplied xG/odds.
                    for key,value in old.items():
                        if (row.get(key) is None or row.get(key) == "") and value is not None:
                            row[key] = value
                con.execute("INSERT OR REPLACE INTO matches VALUES(?,?,?)",
                            (row["league"],row["fixture_id"],json.dumps(row,allow_nan=False)))
        return len(new)

    def matches(self):
        with self.connect() as con:
            rows = [json.loads(r[0]) for r in con.execute("SELECT payload FROM matches")]
        f = pd.DataFrame(rows)
        if len(f):
            for c in ["kickoff","available_at","xg_available_at"]:
                f[c] = pd.to_datetime(f[c],utc=True)
            f = f.sort_values(["kickoff","league","fixture_id"]).reset_index(drop=True)
        return f

    def attach_xg(self,frame,allow_corrections=False):
        required = ["fixture_id","home_xg","away_xg","provider"]
        if not set(required).issubset(frame):
            raise ValueError("xG CSV needs fixture_id, home_xg, away_xg, provider; league is recommended.")
        data = self.matches()
        if data.empty:
            raise ValueError("Import match history first.")
        frame = frame.copy()
        if "league" not in frame:
            mapping = data.groupby("fixture_id").league.agg(list)
            if not frame.fixture_id.map(lambda x: x in mapping and len(mapping[x])==1).all():
                raise ValueError("Ambiguous/unknown fixture IDs: add league to xG CSV.")
            frame["league"] = frame.fixture_id.map(lambda x:mapping[x][0])
        if frame.duplicated(["league","fixture_id"]).any():
            raise ValueError("Duplicate xG fixture IDs.")
        from .data import normalize
        changes = []
        for _,r in frame.iterrows():
            hit = data.loc[data.league.eq(r.league)&data.fixture_id.eq(r.fixture_id)]
            if len(hit)!=1 or hit.iloc[0].home_goals is None or pd.isna(hit.iloc[0].home_goals):
                raise ValueError(f"xG must match a completed fixture: {r.fixture_id}")
            row = hit.iloc[0].to_dict()
            if pd.notna(row["home_xg"]) and not allow_corrections:
                if any(str(row[k])!=str(r[k]) for k in ["home_xg","away_xg","provider"]):
                    raise ValueError("Existing xG differs. Enable corrections explicitly.")
            for key in ["home_xg","away_xg","provider","source_url","xg_available_at"]:
                if key in r:
                    row[key] = r[key]
            changes.append(row)
        normalized = normalize(pd.DataFrame(changes),completed=True)
        return self.import_matches(normalized,allow_corrections=True)

    def save_run(self,run):
        with self.connect() as con:
            con.execute("INSERT INTO runs VALUES(?,?,?,?,?)",
                        (run["run_id"],run["created_at"],run["bundle_id"],run["mode"],
                         json.dumps(run,allow_nan=False)))
    def runs(self):
        with self.connect() as con:
            return [json.loads(r[0]) for r in con.execute("SELECT payload FROM runs ORDER BY created_at DESC")]
