from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from engine import *
from models import contributions

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def history():
    return read_matches((ROOT/"examples/historical_example.csv").read_bytes())

@pytest.fixture(scope="session")
def result(history):
    return assess(history)

def test_uploaded_example(history):
    assert len(history)==380
    assert history.division.unique().tolist()==["E0"]
    assert history.date.min()==pd.Timestamp("2023-08-11")

def test_split_no_date_overlap(history,result):
    _,train,cal,test=split_data(history)
    assert train.date.max()<cal.date.min()
    assert cal.date.max()<test.date.min()
    assert sum(len(t) for t in [train,cal,test])+result["meta"]["skipped"]==380

def test_future_goals_do_not_leak(history):
    target=history.iloc[[100]]
    old=make_features(history,target)[FEATURES].to_numpy()
    h=history.copy();h.loc[h.date>=target.date.iloc[0],["home_goals","away_goals"]]=99
    np.testing.assert_array_equal(make_features(h,target)[FEATURES].to_numpy(),old)

def test_invalid_inputs(history):
    raw=pd.read_csv(ROOT/"examples/historical_example.csv")
    with pytest.raises(ValueError,match="Duplicate"):
        read_matches(pd.concat([raw,raw.iloc[[0]]]).to_csv(index=False).encode())
    with pytest.raises(ValueError):
        read_matches(raw.assign(FTHG=-1).to_csv(index=False).encode())
    with pytest.raises(ValueError,match="Missing"):
        read_matches(raw.drop(columns="HomeTeam").to_csv(index=False).encode())
    with pytest.raises(ValueError,match="contains results"):
        read_matches(raw.to_csv(index=False).encode(),False)

def test_probabilities_and_shap(result):
    for market,options in result["probabilities"].items():
        for p in options.values():
            np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
            assert (p>=0).all()
        v,m=contributions(result["models"][market],result["test"][FEATURES])
        np.testing.assert_allclose(v.sum(-1),m,atol=2e-5,rtol=1e-5)
    assert len(result["scores"])==15

def test_save_load_and_predict(result,tmp_path):
    save_result(result,tmp_path)
    restored=load_result(tmp_path)
    future=read_matches((ROOT/"examples/upcoming_example.csv").read_bytes(),False)
    before,_,_=predict_upcoming(result,future)
    after,_,_=predict_upcoming(restored,future)
    np.testing.assert_allclose(before.Probability,after.Probability)
    assert len(after)==14
    assert after.Status.eq("Most likely").sum()==6

def test_unknown_and_wrong_dates(result):
    f=read_matches((ROOT/"examples/upcoming_example.csv").read_bytes(),False)
    f.loc[0,"home_team"]="Unknown team"
    table,_,_=predict_upcoming(result,f)
    assert (table.Market=="Unavailable").sum()==1
    f["date"]=pd.Timestamp("2023-01-01")
    with pytest.raises(ValueError,match="after"):
        predict_upcoming(result,f)

def test_wrong_league(result):
    f=read_matches((ROOT/"examples/upcoming_example.csv").read_bytes(),False)
    f["division"]="SP1"
    with pytest.raises(ValueError,match="matching"):
        predict_upcoming(result,f)

def test_empty_screen(tmp_path,monkeypatch):
    monkeypatch.setenv("SOCCER_SIMPLE_DATA",str(tmp_path))
    a=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60).run()
    assert not a.exception
    assert not a.error
    assert len(a.tabs)==2

def test_populated_screen(result,tmp_path,monkeypatch):
    save_result(result,tmp_path)
    monkeypatch.setenv("SOCCER_SIMPLE_DATA",str(tmp_path))
    a=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60).run()
    assert not a.exception
    assert not a.error
    for market in MARKETS:
        a.selectbox(key="eval_market").set_value(market).run()
        assert not a.exception
        assert not a.error

def test_full_upload_train_predict_ui(tmp_path,monkeypatch):
    monkeypatch.setenv("SOCCER_SIMPLE_DATA",str(tmp_path))
    a=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60).run()
    a.get("file_uploader")[0].upload("historical.csv",(ROOT/"examples/historical_example.csv").read_bytes(),"text/csv").run()
    assert not a.error
    next(b for b in a.button if b.label=="Train models & show results").click().run()
    assert not a.error
    assert not a.exception
    assert (tmp_path/"latest.txt").exists()
    uploader=next(f for f in a.get("file_uploader") if f.key=="upcoming")
    uploader.upload("upcoming.csv",(ROOT/"examples/upcoming_example.csv").read_bytes(),"text/csv").run()
    next(b for b in a.button if b.label=="Predict upcoming matches").click().run()
    assert not a.error
    assert not a.exception
    table=a.session_state["predictions"][0]
    assert len(table)==14
    assert table.Status.eq("Most likely").sum()==6
    for market in MARKETS:
        a.selectbox(key="future_market").set_value(market).run()
        assert not a.error
        assert not a.exception

def test_seventy_ready_matches_can_train(history):
    rows=make_features(history,history)
    end=rows.loc[rows.ready].iloc[69].date
    short=history.loc[history.date<=end].copy()
    # Retain exactly the first 70 ready fixtures, even if a date has more fixtures.
    ids=set(rows.loc[rows.ready].head(70).fixture_id)
    short=short.loc[short.fixture_id.isin(ids)|~short.fixture_id.isin(set(rows.loc[rows.ready].fixture_id))]
    _,train,cal,test=split_data(short)
    assert len(train)+len(cal)+len(test)==70
    assert len(train)>=30 and len(cal)>=10 and len(test)>=10
    assert train.date.max()<cal.date.min()<test.date.min()
    r=assess(short)
    assert len(r["scores"])==15

def test_fewer_than_seventy_rejected(history):
    rows=make_features(history,history)
    ids=set(rows.loc[rows.ready].head(69).fixture_id)
    short=history.loc[history.fixture_id.isin(ids)|~history.fixture_id.isin(set(rows.loc[rows.ready].fixture_id))]
    with pytest.raises(ValueError,match="At least 70"):
        split_data(short)
