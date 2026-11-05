import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from soccer.data import normalize, features, FORM, labels
from soccer.demo import synthetic_data
from soccer.models import DixonColes, scale
from soccer.training import TrainingConfig, train_bundle, load_bundle, pick_rows
from soccer.forecast import generate, forecast_table
from soccer.storage import Store
from soccer.evaluation import evaluate, block_interval

@pytest.fixture(scope="session")
def demo():
    h,f=synthetic_data()
    return h.loc[h.league.eq("Demo England")].copy(),f.loc[f.league.eq("Demo England")].copy()

@pytest.fixture(scope="session")
def fitted(demo,tmp_path_factory):
    h,_=demo
    root=tmp_path_factory.mktemp("models")
    manifest=train_bundle(h,TrainingConfig(),root)
    restored,models=load_bundle(root/manifest["bundle_id"])
    return root,restored,models

def test_input_validation():
    f=pd.DataFrame([dict(league="A",season="2025/26",kickoff="2025-08-01",home_team="H",away_team="A",home_goals=2,away_goals=1)])
    n=normalize(f)
    assert n.fixture_id.iloc[0]=="2025/26|H|A"
    with pytest.raises(ValueError,match="Duplicate"):
        normalize(pd.concat([f,f]))
    with pytest.raises(ValueError):
        normalize(f.assign(home_goals=-1))
    with pytest.raises(ValueError):
        normalize(f.assign(home_goals=.5))
    with pytest.raises(ValueError):
        normalize(f.assign(away_team="H"))

def test_football_data_timezone():
    raw=pd.DataFrame([dict(Date="16/08/2025",Time="15:00",HomeTeam="H",AwayTeam="A",FTHG=2,FTAG=1,FTR="H")])
    f=normalize(raw,"Premier League","2025/26","Europe/London")
    assert f.kickoff.iloc[0]==pd.Timestamp("2025-08-16T14:00:00Z")

def test_future_mutation_does_not_leak(demo):
    h,f=demo
    target=h.iloc[[150]]
    boundary=target.kickoff.iloc[0]
    before=features(h,target)[FORM].to_numpy()
    changed=h.copy()
    changed.loc[changed.kickoff>=boundary,["home_goals","away_goals"]]=99
    after=features(changed,target)[FORM].to_numpy()
    np.testing.assert_array_equal(before,after)

def test_same_day_excluded_and_availability(demo):
    h,_=demo
    target=h.iloc[[150]]
    boundary=target.kickoff.iloc[0]
    old=h.loc[h.kickoff<boundary].copy()
    shifted=old.copy()
    shifted.loc[:,"available_at"]=boundary+pd.Timedelta(days=1)
    assert not features(shifted,target).iloc[0].ready
    same=old.copy()
    same.loc[:,"kickoff"]=boundary-pd.Timedelta(hours=1)
    same.loc[:,"available_at"]=boundary-pd.Timedelta(minutes=1)
    assert not features(same,target).iloc[0].ready

def test_xg_missing_blocks_and_provider_required(demo):
    h,f=demo
    assert features(h,f,use_xg=True).ready.all()
    h=h.copy();h["home_xg"]=np.nan
    assert not features(h,f,use_xg=True).ready.any()
    with pytest.raises(ValueError,match="provider"):
        normalize(pd.DataFrame([dict(league="L",season="S",kickoff="2025-08-01",home_team="A",away_team="B",home_goals=1,away_goals=0,home_xg=1,away_xg=.5)]))

def test_temperature_preserves_argmax():
    p=np.array([[.1,.2,.7],[.45,.4,.15]])
    q=scale(p,2)
    np.testing.assert_array_equal(q.argmax(1),p.argmax(1))
    np.testing.assert_allclose(q.sum(1),1)

def test_all_score_markets_and_order(demo,fitted):
    _,m,_=fitted
    _,f=demo
    dc=DixonColes(**m["leagues"]["Demo England"]["dc"])
    p,_,_,g=dc.predict(f)
    for market in p:
        assert (p[market]>=0).all()
        np.testing.assert_allclose(p[market].sum(1),1)
    assert p["1X2"].shape[1]==3
    np.testing.assert_allclose(g.sum((1,2)),1)

def test_roundtrip_forecast_and_shap(demo,fitted):
    h,f=demo
    _,m,models=fitted
    r=generate(h,f,m,models,pd.Timestamp.now(tz="UTC"))
    assert len(r["fixtures"])==8
    assert all(t["ready"] for t in r["fixtures"])
    for t in r["fixtures"]:
        for market,v in t["markets"].items():
            assert v["additivity_error"]<2e-5
            assert abs(sum(v["probabilities"])-1)<1e-6
    assert len(forecast_table(r))==8*7
    json.dumps(r,allow_nan=False)

def test_pre_freeze_and_live_past_rejected(demo,fitted):
    h,f=demo;_,m,models=fitted
    with pytest.raises(ValueError,match="precedes"):
        generate(h,f,m,models,"2024-01-01",mode="Historical replay")
    with pytest.raises(ValueError):
        generate(h,h.tail(8),m,models,pd.Timestamp.now(tz="UTC"))

def test_selection_abstention_and_ties():
    cs=[dict(league="L",fixture_id="f",market=m,confidence=.7) for m in ["BTTS","1X2"]]
    rs=[dict(league="L",market=m,league_eligible=True,common_eligible=True) for m in ["BTTS","1X2"]]
    assert len(pick_rows(cs,rs,"league",.65))==1
    assert pick_rows(cs,rs,"league",.65)[0]["market"]=="1X2"
    assert pick_rows(cs,rs,"league",None)==[]
    assert pick_rows(cs,rs,"league",.8)==[]
    for r in rs:r["league_eligible"]=False
    assert pick_rows(cs,rs,"league",.5)==[]

def test_storage_snapshot_and_results(demo,fitted,tmp_path):
    h,_=demo;_,m,models=fitted
    target=h.loc[h.kickoff>=pd.Timestamp("2025-08-01",tz="UTC")].head(8)
    run=generate(h,target,m,models,"2025-08-01",mode="Historical replay")
    s=Store(tmp_path/"db.sqlite3")
    s.import_matches(h)
    s.save_run(run)
    with pytest.raises(Exception):
        s.save_run(run)
    scores,picks,exclusions=evaluate(s.runs()[0],s.matches())
    assert len(scores)==8*3*2
    assert not exclusions
    snapshot=json.dumps(s.runs()[0])
    correction=h.iloc[[0]].copy();correction["home_goals"]+=1
    with pytest.raises(ValueError): s.import_matches(correction)
    s.import_matches(correction,allow_corrections=True)
    assert json.dumps(s.runs()[0])==snapshot

def test_odds_match_and_postponed_exclusion(demo,fitted):
    h,_=demo;_,m,models=fitted
    target=h.loc[h.kickoff>=pd.Timestamp("2025-08-01",tz="UTC")].head(8)
    run=generate(h,target,m,models,"2025-08-01",mode="Historical replay")
    changed=h.copy()
    changed[["AvgCH","AvgCD","AvgCA"]]=[2.,3.,4.]
    scores,_,_=evaluate(run,changed)
    book=scores.loc[scores.forecast.eq("Bookmaker")]
    assert len(book)==8
    assert book.market.eq("1X2").all()
    np.testing.assert_allclose(np.array(book.probabilities.tolist()).sum(1),1)
    changed.loc[changed.fixture_id.eq(target.fixture_id.iloc[0]),"kickoff"]+=pd.Timedelta(days=1)
    _,_,ex=evaluate(run,changed)
    assert len(ex)==1

def test_bad_dates_and_insufficient_data(demo,tmp_path):
    h,_=demo
    with pytest.raises(ValueError): TrainingConfig(cal_end="2023-01-01").validate()
    with pytest.raises(ValueError,match="insufficient"):
        train_bundle(h.head(10),TrainingConfig(),tmp_path)
    assert not list(tmp_path.glob("*/manifest.json"))

def test_xg_attach(demo,tmp_path):
    h,_=demo;s=Store(tmp_path/"db.sqlite3")
    base=h.head(4).copy();base[["home_xg","away_xg"]]=np.nan;base["provider"]=""
    s.import_matches(base)
    x=h.head(4)[["league","fixture_id","home_xg","away_xg","provider"]]
    assert s.attach_xg(x)==4
    assert s.matches().home_xg.notna().all()

def test_block_interval():
    f=pd.DataFrame({"kickoff":pd.date_range("2025-01-01",periods=30,freq="7D",tz="UTC"),
                    "correct":[1,0,1]*10,"baseline_correct":[1,0,1]*10})
    ci=block_interval(f,100)
    np.testing.assert_allclose(ci[:,1],0)

def test_populated_screens_and_replay_button(demo,fitted,tmp_path,monkeypatch):
    import shutil
    from streamlit.testing.v1 import AppTest
    h,f=demo;root,m,models=fitted
    monkeypatch.setenv("SOCCER_DATA_DIR",str(tmp_path))
    shutil.copytree(root,tmp_path/"models")
    s=Store(tmp_path/"soccer.sqlite3");s.import_matches(h);s.import_matches(f)
    target=h.loc[h.kickoff>=pd.Timestamp("2025-08-01",tz="UTC")].head(8)
    s.save_run(generate(h,target,m,models,"2025-08-01",mode="Historical replay"))
    path=Path(__file__).resolve().parents[1]/"app.py"
    for page in ["Data & readiness","Train & freeze","Forecasts","Explanations","Performance"]:
        app=AppTest.from_file(str(path),default_timeout=60).run()
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, str(app.exception)
        assert not app.error, str(app.error)
    app=AppTest.from_file(str(path),default_timeout=60).run()
    app.sidebar.radio[0].set_value("Forecasts").run()
    next(b for b in app.button if b.label=="Generate and save forecasts").click().run()
    assert not app.exception
    assert not app.error
    assert len(s.runs())==2

def test_xg_training_and_forecast(demo,tmp_path):
    h,f=demo
    m=train_bundle(h,TrainingConfig(use_xg=True),tmp_path)
    m,models=load_bundle(tmp_path/m["bundle_id"])
    assert len(m["feature_names"])==10
    run=generate(h,f,m,models,pd.Timestamp.now(tz="UTC"))
    assert all(t["ready"] for t in run["fixtures"])
    bad=h.copy();bad["provider"]="Incompatible"
    rejected=generate(bad,f,m,models,pd.Timestamp.now(tz="UTC"))
    assert not any(t["ready"] for t in rejected["fixtures"])
