"""Synthetic data for interface testing only; no research performance claims."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .data import normalize

def synthetic_data():
    rng=np.random.default_rng(42)
    today=pd.Timestamp.now(tz="UTC").normalize()
    rows=[]
    for league,prefix in [("Demo England","E"),("Demo Spain","S")]:
        teams=[f"Demo {prefix}{i:02}" for i in range(1,17)]
        strengths=np.linspace(-.4,.4,16)
        for day in pd.date_range(pd.Timestamp("2023-08-01",tz="UTC"),today-pd.Timedelta(days=1),freq="7D",tz="UTC"):
            order=rng.permutation(16)
            season=f"{day.year if day.month>=7 else day.year-1}/{str(day.year+1 if day.month>=7 else day.year)[-2:]}"
            for j in range(0,16,2):
                h,a=order[j:j+2]
                hx=np.exp(.3+.18+strengths[h]-strengths[a])
                ax=np.exp(.3+strengths[a]-strengths[h])
                rows.append(dict(league=league,season=season,fixture_id=f"demo-{prefix}-{day.date()}-{j}",
                    kickoff=(day+pd.Timedelta(hours=15)).isoformat(),home_team=teams[h],away_team=teams[a],
                    home_goals=int(rng.poisson(hx)),away_goals=int(rng.poisson(ax)),
                    home_xg=round(hx,3),away_xg=round(ax,3),provider="Synthetic demo",
                    source_url="synthetic://seed-42"))
    history=normalize(pd.DataFrame(rows))
    upcoming=[]
    day=today+pd.Timedelta(days=7,hours=15)
    for league,prefix in [("Demo England","E"),("Demo Spain","S")]:
        for j in range(1,17,2):
            upcoming.append(dict(league=league,season=f"{day.year}/{str(day.year+1)[-2:]}",
                fixture_id=f"demo-next-{prefix}-{day.date()}-{j}",kickoff=day.isoformat(),
                home_team=f"Demo {prefix}{j:02}",away_team=f"Demo {prefix}{j+1:02}"))
    return history,normalize(pd.DataFrame(upcoming),completed=False)
