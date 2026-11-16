"""Two-phase forecasting: chronological assessment, then future fixture inference."""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import hashlib
import json
import uuid
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from models import MARKETS, DixonColes, fit_xgb, calibrate, scale, log_loss, brier

FEATURES = [f"{side}_{name}_last_3" for side in ("home","away")
            for name in ("points","goals_scored","goals_conceded")]

def read_matches(payload, completed=True):
    """Football-Data headers; match dates are local calendar dates, not UTC instants."""
    raw = pd.read_csv(BytesIO(payload),encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()
    required = ["Date","HomeTeam","AwayTeam"] + (["FTHG","FTAG"] if completed else [])
    missing = set(required)-set(raw.columns)
    if missing:
        raise ValueError("Missing CSV columns: "+", ".join(sorted(missing)))
    if raw.empty:
        raise ValueError("The CSV has no matches.")
    f = pd.DataFrame()
    # Explicit ISO or day-first parsing avoids mixed-date inference across files.
    def parse_date(value):
        value=str(value).strip()
        if len(value)>=10 and value[4]=="-":
            return pd.to_datetime(value,format="%Y-%m-%d",errors="raise")
        return pd.to_datetime(value,dayfirst=True,errors="raise")
    f["date"] = raw.Date.map(parse_date)
    if f.date.isna().any():
        raise ValueError("Every row needs a valid Date.")
    f["date"] = f.date.dt.normalize()
    for source,target in [("HomeTeam","home_team"),("AwayTeam","away_team")]:
        if raw[source].isna().any():
            raise ValueError("Missing team names.")
        f[target] = raw[source].astype(str).str.strip()
        if f[target].eq("").any():
            raise ValueError("Blank team names.")
    if f.home_team.eq(f.away_team).any():
        raise ValueError("A team cannot play itself.")
    f["division"] = raw.Div.fillna("").astype(str).str.strip() if "Div" in raw else "Single league"
    if f.division.eq("").any():
        raise ValueError("Div contains blank league codes.")
    for source,target in [("FTHG","home_goals"),("FTAG","away_goals")]:
        if completed:
            v=pd.to_numeric(raw[source],errors="raise")
            if v.isna().any() or (~np.isfinite(v)).any() or (v<0).any() or (v%1!=0).any():
                raise ValueError(f"{source} must contain nonnegative whole-number goals.")
            f[target]=v.astype(int)
        else:
            if source in raw and raw[source].notna().any():
                raise ValueError("Upcoming CSV contains results. Remove FTHG/FTAG values.")
            f[target]=np.nan
    if completed and "FTR" in raw:
        expected=np.where(f.home_goals>f.away_goals,"H",np.where(f.home_goals==f.away_goals,"D","A"))
        if not np.all(raw.FTR==expected):
            raise ValueError("FTR does not match the full-time goals.")
    for col in ["AvgCH","AvgCD","AvgCA"]:
        f[col]=pd.to_numeric(raw[col],errors="coerce") if col in raw else np.nan
    return validate_matches(f)

def validate_matches(f):
    f=f.copy().sort_values(["date","division","home_team","away_team"]).reset_index(drop=True)
    if f.duplicated(["division","date","home_team","away_team"]).any():
        raise ValueError("Duplicate fixtures detected, possibly across uploaded CSVs.")
    appearances=pd.concat([f[["division","date","home_team"]].rename(columns={"home_team":"team"}),
                           f[["division","date","away_team"]].rename(columns={"away_team":"team"})])
    if appearances.duplicated(["division","date","team"]).any():
        raise ValueError("A team appears more than once on the same date. Check the uploaded files.")
    f["fixture_id"]=f.division+"|"+f.date.dt.strftime("%Y-%m-%d")+"|"+f.home_team+"|"+f.away_team
    return f

def make_features(history,targets):
    entries=defaultdict(list)
    for r in history.sort_values("date").itertuples():
        if pd.isna(r.home_goals) or pd.isna(r.away_goals):
            continue
        for team,gf,ga in [(r.home_team,r.home_goals,r.away_goals),(r.away_team,r.away_goals,r.home_goals)]:
            entries[(r.division,team)].append((r.date,3 if gf>ga else 1 if gf==ga else 0,gf,ga))
    rows=[]
    for _,t in targets.iterrows():
        row=t.to_dict(); reasons=[]
        for side in ["home","away"]:
            prior=[e for e in entries[(t.division,t[side+"_team"])] if e[0]<t.date][-3:]
            if len(prior)<3:
                reasons.append(f"{t[side+'_team']}: only {len(prior)} prior matches (needs 3)")
            for i,name in enumerate(["points","goals_scored","goals_conceded"],1):
                row[f"{side}_{name}_last_3"]=sum(e[i] for e in prior) if len(prior)==3 else np.nan
        row["ready"]=not reasons
        row["reason"]="; ".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)

def target_labels(frame,market):
    h,a=frame.home_goals.to_numpy(),frame.away_goals.to_numpy()
    if market=="1X2": return np.where(h>a,0,np.where(h==a,1,2))
    if market=="O/U 1.5": return (h+a>=2).astype(int)
    return ((h>0)&(a>0)).astype(int)

def split_data(history):
    rows=make_features(history,history)
    ready=rows.loc[rows.ready].reset_index(drop=True)
    n=len(ready)
    if n<70:
        raise ValueError(f"Only {n} matches have enough prior history. At least 70 feature-ready matches are required.")
    # Find the nearest feasible 60/20/20 split without separating a match date.
    # Minimum support: 30 training, 10 calibration and 10 test matches.
    days=ready.groupby("date",sort=True).size()
    cuts=days.cumsum().to_numpy()[:-1]
    best=None
    for a in cuts[cuts>=30]:
        bs=cuts[(cuts>=a+10)&(cuts<=n-10)]
        if not len(bs):
            continue
        errors=(a/n-.6)**2+((bs-a)/n-.2)**2+((n-bs)/n-.2)**2
        j=int(np.argmin(errors))
        candidate=(float(errors[j]),int(a),int(bs[j]))
        if best is None or candidate<best:
            best=candidate
    if best is None:
        raise ValueError("Not enough distinct match dates for chronological blocks of at least 30 training, 10 calibration and 10 test matches. Add matches from more dates.")
    _,a,b=best
    train=ready.iloc[:a]
    cal=ready.iloc[a:b]
    test=ready.iloc[b:]
    return rows,train,cal,test

def assess(history,progress=lambda s:None):
    if history.division.nunique()!=1:
        raise ValueError("Select one league before training.")
    rows,train,cal,test=split_data(history)
    for market,labels in MARKETS.items():
        if len(np.unique(target_labels(train,market)))!=len(labels):
            raise ValueError(f"Training does not contain every {market} outcome. Upload more history.")
    progress("Fitting Dixon–Coles…")
    dc=DixonColes.fit(history.loc[history.date<cal.date.min()])
    dc_cal,*_=dc.predict(cal); dc_test,*_=dc.predict(test)
    models={}; probabilities={}; scores=[]; choices={}; temperatures={}; frequencies={}
    for market,labels in MARKETS.items():
        progress(f"Fitting XGBoost and calibrating {market}…")
        model=fit_xgb(train[FEATURES],target_labels(train,market))
        models[market]=model
        freq=np.bincount(target_labels(train,market),minlength=len(labels)).astype(float)
        freq/=freq.sum();frequencies[market]=freq.tolist()
        yc,yt=target_labels(cal,market),target_labels(test,market)
        probabilities[market]={"Historical frequency":np.tile(freq,(len(test),1))}
        cal_options={}
        temperatures[market]={}
        for name,cp,tp in [("XGBoost",model.predict_proba(cal[FEATURES]),model.predict_proba(test[FEATURES])),
                            ("Dixon–Coles",dc_cal[market],dc_test[market])]:
            t=calibrate(cp,yc);temperatures[market][name]=t
            for suffix,cq,tq in [("raw",cp,tp),("calibrated",scale(cp,t),scale(tp,t))]:
                key=name+" · "+suffix
                probabilities[market][key]=tq
                cal_options[key]=log_loss(cq,yc)
        # Selection uses development outcomes only, never the displayed test scores.
        choices[market]=min(cal_options,key=cal_options.get)
        for name,p in probabilities[market].items():
            scores.append({"market":market,"model":name,"fixtures":len(test),
                           "log_loss":log_loss(p,yt),"brier":brier(p,yt),
                           "accuracy":float((p.argmax(axis=1)==yt).mean())})
    metadata={"version":1,"created_at":datetime.now(timezone.utc).isoformat(),
              "division":str(history.division.iloc[0]),"features":FEATURES,
              "source_hash":hashlib.sha256(history.to_csv(index=False).encode()).hexdigest(),
              "choices":choices,"temperatures":temperatures,"frequencies":frequencies,
              "split":[{"block":name,"fixtures":len(f),"first_date":str(f.date.min().date()),
                        "last_date":str(f.date.max().date())} for name,f in [("Training",train),("Calibration / model choice",cal),("Test",test)]],
              "skipped":len(rows)-len(train)-len(cal)-len(test)}
    return {"history":history,"test":test.reset_index(drop=True),"models":models,"dc":dc,
            "probabilities":probabilities,"scores":pd.DataFrame(scores),"meta":metadata}

def predict_upcoming(result,fixtures):
    history=result["history"];meta=result["meta"];fixtures=fixtures.copy()
    if fixtures.division.eq("Single league").all():
        fixtures["division"]=meta["division"]
        fixtures=validate_matches(fixtures.drop(columns="fixture_id"))
    if not fixtures.division.eq(meta["division"]).all():
        raise ValueError(f"This model was trained for {meta['division']}. Upload matching fixtures.")
    if (fixtures.date<=history.date.max()).any():
        raise ValueError(f"Upcoming dates must be after the last history date ({history.date.max().date()}).")
    ready=make_features(history,fixtures)
    output=[];details={}
    for _,r in ready.iterrows():
        base={"Date":str(r.date.date()),"HomeTeam":r.home_team,"AwayTeam":r.away_team}
        if not r.ready:
            output.append(dict(base,Market="Unavailable",Model="",Outcome="",Probability=np.nan,Status=r.reason))
            continue
        x=pd.DataFrame([r])[FEATURES]
        dp,hr,ar,grid=result["dc"].predict(pd.DataFrame([r]))
        details[r.fixture_id]={"features":x.iloc[0].to_dict(),"rates":[float(hr[0]),float(ar[0])],
                              "unseen_teams":[t for t in [r.home_team,r.away_team] if t not in result["dc"].teams]}
        for market,labels in MARKETS.items():
            choice=meta["choices"][market]
            name,suffix=choice.split(" · ")
            p=result["models"][market].predict_proba(x) if name=="XGBoost" else dp[market]
            if suffix=="calibrated":p=scale(p,meta["temperatures"][market][name])
            for k,label in enumerate(labels):
                output.append(dict(base,Market=market,Model=choice,Outcome=label,Probability=float(p[0,k]),
                                   Status="Most likely" if k==p[0].argmax() else ""))
    return pd.DataFrame(output),ready,details

def save_result(result,root):
    """Native model formats only. Immutable folder; atomic pointer to latest success."""
    root=Path(root);root.mkdir(exist_ok=True,parents=True)
    key=uuid.uuid4().hex;folder=root/key;folder.mkdir()
    manifest=dict(result["meta"]);manifest["dc"]=asdict(result["dc"])
    manifest["probabilities"]={m:{n:p.tolist() for n,p in opts.items()} for m,opts in result["probabilities"].items()}
    for i,(market,model) in enumerate(result["models"].items()):
        model.save_model(folder/f"model_{i}.ubj")
    result["history"].to_csv(folder/"history.csv",index=False)
    result["scores"].to_csv(folder/"scores.csv",index=False)
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
    temp=root/"latest.tmp";temp.write_text(key,encoding="utf-8");temp.replace(root/"latest.txt")

def load_result(root):
    root=Path(root)
    if not (root/"latest.txt").exists():return None
    key=(root/"latest.txt").read_text().strip()
    if len(key)!=32 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid saved model identifier.")
    folder=root/key
    m=json.loads((folder/"manifest.json").read_text())
    if m.get("version")!=1:raise ValueError("Unsupported saved model format.")
    h=pd.read_csv(folder/"history.csv");h["date"]=pd.to_datetime(h.date)
    # Reconstruct the original saved test interval, even if split rules change.
    rows=make_features(h,h)
    interval=next(block for block in m["split"] if block["block"]=="Test")
    test=rows.loc[rows.ready & rows.date.ge(pd.Timestamp(interval["first_date"]))
                  & rows.date.le(pd.Timestamp(interval["last_date"]))]
    models={}
    for i,market in enumerate(MARKETS):
        model=XGBClassifier();model.load_model(folder/f"model_{i}.ubj");models[market]=model
    probabilities={market:{n:np.array(p) for n,p in opts.items()} for market,opts in m.pop("probabilities").items()}
    dc=DixonColes(**m.pop("dc"))
    return dict(history=h,test=test.reset_index(drop=True),models=models,dc=dc,
                probabilities=probabilities,scores=pd.read_csv(folder/"scores.csv"),meta=m)

def team_table(history):
    rows=[]
    for t in sorted(set(history.home_team)|set(history.away_team)):
        games=history.loc[history.home_team.eq(t)|history.away_team.eq(t)]
        home=games.home_team.eq(t)
        gf=np.where(home,games.home_goals,games.away_goals)
        ga=np.where(home,games.away_goals,games.home_goals)
        rows.append({"Team":t,"Played":len(games),"Points":int(np.where(gf>ga,3,np.where(gf==ga,1,0)).sum()),
                     "Goals scored":int(gf.sum()),"Goals conceded":int(ga.sum())})
    return pd.DataFrame(rows).sort_values(["Points","Goals scored"],ascending=False)
