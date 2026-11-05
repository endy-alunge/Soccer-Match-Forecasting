"""Chronological model fitting, market qualification and threshold freezing."""
from __future__ import annotations
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from .data import FORM, XG, MARKETS, features, labels, utc, fingerprint
from .models import DixonColes, fit_xgb, calibrate, scale, log_loss

@dataclass
class TrainingConfig:
    train_end: str = "2024-08-01"
    cal_end: str = "2024-12-01"
    rank_mid: str = "2025-01-15"
    rank_end: str = "2025-02-15"
    threshold_end: str = "2025-04-01"
    use_xg: bool = False
    min_window: int = 25
    min_picks: int = 15
    min_coverage: float = .20
    hit_target: float = .60

    def validate(self):
        ds = [utc(getattr(self, k)) for k in ("train_end","cal_end","rank_mid","rank_end","threshold_end")]
        if not all(a < b for a,b in zip(ds,ds[1:])):
            raise ValueError("Cutoffs must be strictly increasing.")
        if self.min_window < 10 or self.min_picks < 1 or not 0 < self.min_coverage <= 1 or not 0 < self.hit_target < 1:
            raise ValueError("Invalid validation support settings.")
        return ds

def pick_rows(candidates, rankings, policy, threshold):
    """At most one eligible market per fixture; alphabetical ties match research."""
    if not candidates or threshold is None:
        return []
    eligible = {(r["league"],r["market"]) for r in rankings if r[policy + "_eligible"]}
    best = {}
    for r in sorted(candidates, key=lambda r: (-r["confidence"], r["market"], r["fixture_id"])):
        key = (r["league"],r["fixture_id"])
        if (r["league"],r["market"]) in eligible and r["confidence"] >= threshold:
            best.setdefault(key, r)
    return list(best.values())

def train_bundle(history, config, root, progress=lambda msg: None):
    """Writes a bundle only after all leagues and checks succeed. No notebook execution."""
    bounds = config.validate()
    history = history.loc[history.home_goals.notna() & (history.kickoff < bounds[-1])
                          & (history.available_at < bounds[-1])].copy()
    if history.empty:
        raise ValueError("No completed matches before the policy freeze date.")
    if config.use_xg:
        providers = set(history.loc[history.home_xg.notna(),"provider"])
        if len(providers) != 1:
            raise ValueError("The xG arm requires exactly one labelled provider across its training data.")
    else:
        providers = set()
    bundle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / ("." + bundle_id + ".tmp")
    tmp.mkdir()
    manifest = {"schema_version": 1, "bundle_id": bundle_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": asdict(config), "data_hash": fingerprint(history),
                "feature_names": FORM + (XG if config.use_xg else []),
                "xg_provider": next(iter(providers), ""),
                "external_xg_validation": "Pending; not certified by this application",
                "research_status": "New prototype fit; not the published notebook results",
                "packages": {n:importlib.metadata.version(n) for n in ["numpy","pandas","scipy","xgboost"]},
                "leagues": {}, "rankings": [], "thresholds": {}, "validation_grid": []}
    candidates, validation_count = [], 0
    try:
        for li, (league, hist) in enumerate(history.groupby("league", sort=True)):
            progress(f"Training {league}")
            rows = features(hist, hist, use_xg=config.use_xg)
            # Statistics must also be available by the end of the partition that uses them.
            d = rows.kickoff
            windows = [(d < bounds[0]), (d >= bounds[0]) & (d < bounds[1]),
                       (d >= bounds[1]) & (d < bounds[2]), (d >= bounds[2]) & (d < bounds[3]),
                       (d >= bounds[3]) & (d < bounds[4])]
            masks = [w & rows.ready & (rows.available_at < end)
                     for w,end in zip(windows,bounds)]
            counts = dict(zip(["train","cal","rank1","rank2","threshold"], [int(m.sum()) for m in masks]))
            if min(counts.values()) < config.min_window:
                raise ValueError(f"{league}: insufficient eligible fixtures: {counts}. Each partition needs {config.min_window}.")
            train = rows.loc[masks[0]]
            future = rows.loc[masks[1] | masks[2] | masks[3] | masks[4]].copy().reset_index(drop=True)
            d = future.kickoff
            cal = d < bounds[1]
            rank1 = (d >= bounds[1]) & (d < bounds[2])
            rank2 = (d >= bounds[2]) & (d < bounds[3])
            val = d >= bounds[3]
            validation_count += int(val.sum())
            # Same score-model training pool as the notebook: all valid early scores.
            dc_train = hist.loc[(hist.kickoff < bounds[0]) & (hist.available_at < bounds[0])]
            dc = DixonColes.fit(dc_train)
            dp, _, _, _ = dc.predict(future)
            league_meta = {"counts": counts, "dc": asdict(dc), "markets": {},
                           "train_teams": dc.teams}
            for mi,market in enumerate(MARKETS):
                y = labels(future,market)
                yt = labels(train,market)
                k = len(MARKETS[market])
                freq = np.bincount(yt, minlength=k)
                if len(freq) != k or not freq.all():
                    raise ValueError(f"{league}/{market}: at least one training outcome is absent.")
                freq = freq/freq.sum()
                reference = np.tile(freq,(len(future),1))
                xgb = fit_xgb(train[manifest["feature_names"]],yt)
                filename = f"league_{li}_market_{mi}.ubj"
                xgb.save_model(tmp/filename)
                xp = xgb.predict_proba(future[manifest["feature_names"]])
                options, temps = {}, {}
                for name,p in [("Dixon–Coles",dp[market]),("XGBoost",xp)]:
                    temp = calibrate(p[cal],y[cal])
                    temps[name] = temp
                    options[(name,"Raw")] = (p,1.)
                    options[(name,"Scaled")] = (scale(p,temp),temp)
                def skill(p):
                    return [1-log_loss(p[w],y[w])/log_loss(reference[w],y[w]) for w in [rank1,rank2]]
                name,variant = max(options,key=lambda key:np.mean(skill(options[key][0])))
                p,temp = options[(name,variant)]
                s1,s2 = skill(p)
                ranking = {"league":league,"market":market,"model":name,"variant":variant,
                           "temperature":float(temp),"skill_window_1":float(s1),"skill_window_2":float(s2),
                           "league_eligible":bool(min(s1,s2)>0)}
                manifest["rankings"].append(ranking)
                league_meta["markets"][market] = {"xgb_file":filename, "frequency":freq.tolist(),
                    "temperatures":temps, "chosen_model":name,"chosen_variant":variant,"chosen_temperature":float(temp)}
                for i in np.flatnonzero(val):
                    best = int(p[i].argmax())
                    candidates.append({"league":league,"fixture_id":future.iloc[i].fixture_id,"market":market,
                                       "confidence":float(p[i].max()),"correct":bool(best==y[i]),
                                       "baseline_correct":bool(freq.argmax()==y[i])})
            manifest["leagues"][league] = league_meta
        for market in MARKETS:
            rs = [r for r in manifest["rankings"] if r["market"]==market]
            eligible = all(np.mean([r[k] for r in rs]) > 0 for k in ["skill_window_1","skill_window_2"])
            for r in rs:
                r["common_eligible"] = bool(eligible)
        for policy in ["league","common"]:
            grid = []
            for t in np.round(np.arange(.5,.901,.025),3):
                picks = pick_rows(candidates,manifest["rankings"],policy,float(t))
                n = len(picks)
                grid.append({"policy":policy,"threshold":float(t),"picks":n,"coverage":n/validation_count,
                             "hit_rate":float(np.mean([r["correct"] for r in picks])) if n else None})
            supported = [r for r in grid if r["picks"]>=config.min_picks and r["coverage"]>=config.min_coverage]
            meeting = [r for r in supported if r["hit_rate"]>=config.hit_target]
            if meeting:
                choice = sorted(meeting,key=lambda r:(-r["coverage"],-r["hit_rate"],r["threshold"]))[0]
                reason = "Validation target met"
            elif supported:
                choice = sorted(supported,key=lambda r:(-r["hit_rate"],-r["coverage"],r["threshold"]))[0]
                reason = "Supported fallback"
            else:
                choice,reason = None,"No supported threshold"
            manifest["thresholds"][policy] = {"value":choice["threshold"] if choice else None,"reason":reason}
            manifest["validation_grid"].extend(grid)
        (tmp/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
        tmp.rename(root/bundle_id)
        return manifest
    except Exception:
        shutil.rmtree(tmp,ignore_errors=True)
        raise

def list_bundles(root):
    return sorted(Path(root).glob("*/manifest.json"),reverse=True) if Path(root).exists() else []

def load_bundle(path):
    path = Path(path)
    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported bundle schema.")
    models = {}
    for league,info in manifest["leagues"].items():
        for market,m in info["markets"].items():
            filename = m["xgb_file"]
            if Path(filename).name != filename:
                raise ValueError("Invalid model filename.")
            x = XGBClassifier()
            x.load_model(path/filename)
            models[(league,market)] = x
    return manifest,models
