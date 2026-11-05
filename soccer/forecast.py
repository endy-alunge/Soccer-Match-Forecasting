"""Create immutable forecast batches and truthful model-specific explanations."""
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .data import MARKETS, features, utc, fingerprint
from .models import DixonColes, scale, contributions
from .training import pick_rows

def generate(history, targets, manifest, models, cutoff, policy="league", mode="Live"):
    if policy not in ["league","common"] or mode not in ["Live","Historical replay"]:
        raise ValueError("Unknown policy or mode.")
    cutoff = utc(cutoff)
    now = pd.Timestamp.now(tz="UTC")
    freeze = utc(manifest["config"]["threshold_end"])
    if cutoff < freeze:
        raise ValueError("Forecast cutoff precedes the end of policy validation.")
    if mode=="Live":
        if abs((now-cutoff).total_seconds())>300:
            raise ValueError("Live forecasts must use the current time.")
        if (targets.kickoff <= now).any() or targets.home_goals.notna().any():
            raise ValueError("Live forecasts need unplayed upcoming fixtures.")
    if targets.empty:
        raise ValueError("No fixtures selected.")
    if targets.duplicated(["league","fixture_id"]).any():
        raise ValueError("Duplicate target fixtures.")
    if (targets.kickoff < cutoff).any():
        raise ValueError("Target fixtures must be at or after the common forecast cutoff.")
    use_xg = manifest["config"]["use_xg"]
    run = {"run_id":uuid.uuid4().hex, "created_at":datetime.now(timezone.utc).isoformat(),
           "bundle_id":manifest["bundle_id"],"cutoff":cutoff.isoformat(),"mode":mode,"policy":policy,
           "threshold":manifest["thresholds"][policy]["value"],
           "threshold_reason":manifest["thresholds"][policy]["reason"],
           "data_hash":fingerprint(history),"fixtures":[]}
    all_candidates = []
    for _,target in targets.iterrows():
        league = target.league
        item = {k:str(target[k]) for k in ["league","fixture_id","season","home_team","away_team"]}
        item.update(kickoff=target.kickoff.isoformat(),markets={},selection=None)
        run["fixtures"].append(item)
        if league not in manifest["leagues"]:
            item.update(ready=False,reason="No saved model for this league.")
            continue
        target_frame = pd.DataFrame([target])
        # Provider filter turns incompatible xG into missing data, never substitutes silently.
        available = history.copy()
        if use_xg:
            bad = available.provider.ne(manifest["xg_provider"])
            available.loc[bad,["home_xg","away_xg"]] = np.nan
        row = features(available,target_frame,cutoff=cutoff,use_xg=use_xg)
        if not row.iloc[0].ready:
            item.update(ready=False,reason=row.iloc[0].readiness_reason)
            continue
        item.update(ready=True,reason="Ready")
        names = manifest["feature_names"]
        x = row[names]
        item["features"] = {c:float(x.iloc[0][c]) for c in names}
        info = manifest["leagues"][league]
        dc = DixonColes(**info["dc"])
        dp,hr,ar,grid = dc.predict(target_frame)
        item["dc_explanation"] = {"home_rate":float(hr[0]),"away_rate":float(ar[0]),
            "unseen_teams":[t for t in [target.home_team,target.away_team] if t not in dc.teams],
            "score_grid":grid[0].tolist()}
        for market in MARKETS:
            mm = info["markets"][market]
            model = models[(league,market)]
            xp = model.predict_proba(x)
            shap,margin = contributions(model,x)
            raw = {"XGBoost":xp, "Dixon–Coles":dp[market]}
            selected = scale(raw[mm["chosen_model"]],mm["chosen_temperature"])[0]
            ranking = next(r for r in manifest["rankings"] if r["league"]==league and r["market"]==market)
            best = int(selected.argmax())
            item["markets"][market] = {
                "labels":MARKETS[market],"probabilities":selected.tolist(),"baseline":mm["frequency"],
                "model":mm["chosen_model"],"variant":mm["chosen_variant"],"temperature":mm["chosen_temperature"],
                "eligible":ranking[policy+"_eligible"],
                "xgb_raw":xp[0].tolist(),"dc_raw":dp[market][0].tolist(),
                "xgb_temperature":mm["temperatures"]["XGBoost"],
                "xgb_scaled":scale(xp,mm["temperatures"]["XGBoost"])[0].tolist(),
                "shap":shap[0].tolist(),"margins":margin[0].tolist(),
                "additivity_error":float(np.max(np.abs(shap.sum(axis=-1)-margin))),
            }
            all_candidates.append({"league":league,"fixture_id":target.fixture_id,"market":market,
                                   "outcome":MARKETS[market][best],"confidence":float(selected[best])})
    picks = pick_rows(all_candidates,manifest["rankings"],policy,run["threshold"])
    pickmap = {(p["league"],p["fixture_id"]):p for p in picks}
    for item in run["fixtures"]:
        item["selection"] = pickmap.get((item["league"],item["fixture_id"]))
        if item["ready"]:
            if not any(m["eligible"] for m in item["markets"].values()):
                item["reason"] = "No markets qualified for this league/policy."
            elif run["threshold"] is None:
                item["reason"] = "No supported validation threshold."
            elif item["selection"] is None:
                item["reason"] = "Eligible market probabilities are below the frozen threshold."
            else:
                item["reason"] = "Eligible market meets the frozen threshold."
    return run

def forecast_table(run):
    rows=[]
    for f in run["fixtures"]:
        for market,m in f["markets"].items():
            for label,p in zip(m["labels"],m["probabilities"]):
                rows.append({"run_id":run["run_id"],"mode":run["mode"],"cutoff":run["cutoff"],
                             "league":f["league"],"fixture_id":f["fixture_id"],"market":market,
                             "outcome":label,"probability":p,"model":m["model"],"variant":m["variant"],
                             "eligible":m["eligible"],
                             "selected":bool(f["selection"] and f["selection"]["market"]==market and f["selection"]["outcome"]==label)})
    return pd.DataFrame(rows)
