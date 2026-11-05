"""Score saved snapshots against later outcomes, never regenerate old forecasts."""
import numpy as np
import pandas as pd
from .data import labels, utc
from .models import log_loss, brier

ODDS = {"1X2":["AvgCH","AvgCD","AvgCA"],
        "O/U 1.5":["odds_under_1_5","odds_over_1_5"], "BTTS":["odds_btts_no","odds_btts_yes"]}

def evaluate(run, history):
    rows,picks,excluded = [],[],[]
    for f in run["fixtures"]:
        h = history.loc[history.league.eq(f["league"]) & history.fixture_id.eq(f["fixture_id"])]
        if h.empty or pd.isna(h.iloc[0].home_goals) or not f["ready"]:
            continue
        actual = h.iloc[0]
        if actual.kickoff != utc(f["kickoff"]):
            excluded.append(f["fixture_id"]+": kickoff changed after forecast")
            continue
        if actual.available_at > pd.Timestamp.now(tz="UTC"):
            excluded.append(f["fixture_id"]+": result availability is in the future")
            continue
        for market,m in f["markets"].items():
            y = int(labels(h,market)[0])
            p = np.array(m["probabilities"])
            ref = np.array(m["baseline"])
            odds = actual[ODDS[market]].to_numpy(dtype=float)
            book = 1/odds if np.all(np.isfinite(odds)&(odds>1)) else None
            if book is not None:
                book /= book.sum()
            common = {"league":f["league"],"fixture_id":f["fixture_id"],"kickoff":actual.kickoff,
                      "market":market,"observed":m["labels"][y],"class_id":y}
            for name,q in [("Chosen forecast",p),("Historical frequency",ref)] + ([("Bookmaker",book)] if book is not None else []):
                rows.append(dict(common,forecast=name,log_loss=log_loss(q[None,:],np.array([y])),
                                 brier=brier(q[None,:],np.array([y])), probabilities=q.tolist(),
                                 odds_matched=book is not None))
            if f["selection"] and f["selection"]["market"]==market:
                picks.append(dict(common,correct=int(p.argmax()==y),baseline_correct=int(ref.argmax()==y),
                                  confidence=float(p.max())))
    return pd.DataFrame(rows),pd.DataFrame(picks),excluded

def reliability(scores,forecast,market):
    group = scores.loc[scores.forecast.eq(forecast)&scores.market.eq(market)]
    result=[]
    if group.empty:
        return pd.DataFrame()
    p=np.array(group.probabilities.tolist()); y=group.class_id.to_numpy()
    for k in range(p.shape[1]):
        bins=np.minimum((p[:,k]*10).astype(int),9)
        for b in range(10):
            mask=bins==b
            if mask.any():
                result.append({"outcome":k,"bin":b,"count":int(mask.sum()),
                               "mean_probability":float(p[mask,k].mean()),
                               "observed_frequency":float((y[mask]==k).mean())})
    return pd.DataFrame(result)

def block_interval(picks,replicates=2000):
    """Descriptive four-week calendar block resampling of selected outcomes."""
    if len(picks)<2:
        return None
    days=(pd.to_datetime(picks.kickoff,utc=True)-pd.Timestamp("1970-01-05",tz="UTC")).dt.days//28
    block=picks.assign(block=days).groupby("block").agg(n=("correct","size"), hit=("correct","sum"), base=("baseline_correct","sum"))
    if len(block)<2:
        return None
    arr=block.to_numpy();rng=np.random.default_rng(42)
    sampled=arr[rng.integers(0,len(arr),size=(replicates,len(arr)))].sum(axis=1)
    values=np.column_stack([sampled[:,1]/sampled[:,0],(sampled[:,1]-sampled[:,2])/sampled[:,0]])
    return np.quantile(values,[.025,.975],axis=0)
