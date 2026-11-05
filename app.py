"""Run with: python -m streamlit run app.py"""
from pathlib import Path
import io
import json
import os
import traceback
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from soccer.data import normalize, read_csv, MARKETS, TIMEZONES
from soccer.storage import Store
from soccer.training import TrainingConfig, train_bundle, list_bundles, load_bundle
from soccer.forecast import generate, forecast_table
from soccer.evaluation import evaluate, reliability, block_interval

ROOT=Path(__file__).resolve().parent
WORK=Path(os.environ.get("SOCCER_DATA_DIR",str(ROOT/"workspace"))).resolve()
MODELS=WORK/"models"
store=Store(WORK/"soccer.sqlite3")
st.set_page_config(page_title="MatchLab · Soccer forecasting",page_icon="⚽",layout="wide")
st.markdown("""<style>
.block-container {padding-top:2rem;max-width:1400px}
[data-testid="stMetric"] {background:#13263a;padding:18px;border-radius:10px}
</style>""",unsafe_allow_html=True)
st.sidebar.title("⚽ MatchLab")
st.sidebar.caption("Forecast · Explain · Evaluate")
page=st.sidebar.radio("Workspace",["Overview","Data & readiness","Train & freeze","Forecasts","Explanations","Performance"])
st.sidebar.divider()
st.sidebar.caption("Local research prototype · v1.0")
st.sidebar.caption("1X2 · O/U 1.5 · BTTS")
history=store.matches()
bundles=list_bundles(MODELS)

def csv_download(label,frame,name):
    st.download_button(label,frame.to_csv(index=False).encode(),name,"text/csv")

def choose_bundle():
    if not bundles:
        st.info("Train and freeze a model version first.")
        return None
    path=st.selectbox("Model version",bundles,format_func=lambda p:p.parent.name)
    return path.parent

def choose_run():
    runs=store.runs()
    if not runs:
        st.info("Generate and save a forecast batch first.")
        return None
    i=st.selectbox("Saved batch",range(len(runs)),format_func=lambda i:
        f'{runs[i]["created_at"][:19]} · {runs[i]["mode"]} · {runs[i]["policy"]} · {runs[i]["run_id"][:8]}')
    return runs[i]

def display_rankings(manifest):
    st.dataframe(pd.DataFrame(manifest["rankings"]),use_container_width=True,hide_index=True)
    st.json({"thresholds":manifest["thresholds"],"cutoffs":manifest["config"],
             "feature_names":manifest["feature_names"],"xg_provider":manifest["xg_provider"]})

def overview():
    st.title("Soccer forecasting workspace")
    st.write("Turn historical match data into traceable pre-match forecasts.")
    c=st.columns(4)
    c[0].metric("Completed matches",0 if history.empty else int(history.home_goals.notna().sum()))
    c[1].metric("Upcoming fixtures",0 if history.empty else int(history.home_goals.isna().sum()))
    c[2].metric("Model versions",len(bundles))
    c[3].metric("Saved batches",len(store.runs()))
    st.subheader("Start here")
    st.markdown("""
1. **Data & readiness:** import completed matches and upcoming fixtures, or load the synthetic demo.
2. **Train & freeze:** select historical cutoffs and create a versioned model bundle.
3. **Forecasts:** generate and save a live batch or a clearly labelled historical replay.
4. **Explanations:** inspect the exact XGBoost contributions and Dixon–Coles score probabilities.
5. **Performance:** import results, then evaluate the stored batch.
""")
    st.info("A high hit rate alone is not evidence of useful forecasting skill. The dashboard compares probabilities and selections against their matched historical-frequency baselines.")
    st.caption("This application retrains a new version using the research recipe. It does not ship fitted research models or assert that new fits reproduce the report.")
    with st.expander("Data timing and xG limitations"):
        st.write("Canonical timestamps use UTC. Without an explicit availability timestamp, results and xG are assumed available three hours after kickoff. Same-UTC-day matches are excluded from form features. Retrospective provider revisions cannot be undone; historical replay is not prospective evidence. Independently published team-xG validation remains pending.")
    if st.button("Load synthetic demo data",type="primary"):
        from soccer.demo import synthetic_data
        h,f=synthetic_data()
        store.import_matches(h)
        store.import_matches(f)
        st.success("Synthetic history and upcoming fixtures loaded. Go to Train & freeze and keep the default dates.")

def data_page():
    st.title("Data & readiness")
    tab1,tab2,tab3=st.tabs(["Import matches","Attach xG","Inspect data"])
    with tab1:
        st.write("Upload canonical CSVs, or Football-Data files with Date, Time, HomeTeam, AwayTeam, FTHG and FTAG.")
        kind=st.radio("Import type",["Completed matches / results","Upcoming fixtures"],horizontal=True)
        a,b,c=st.columns(3)
        league=a.text_input("Fallback league (files without league column)","Premier League")
        season=b.text_input("Fallback season","2025/26")
        zone=c.selectbox("Football-Data local timezone",["Europe/London","Europe/Madrid","UTC"])
        st.caption("Canonical kickoff values must be ISO 8601; values without an offset are interpreted as UTC. Football-Data Date/Time uses the timezone above. One fallback league/season applies to the whole batch.")
        files=st.file_uploader("Match CSV files",type="csv",accept_multiple_files=True)
        corrections=st.checkbox("Allow explicit corrections to existing kickoffs or results")
        if st.button("Validate and import",type="primary",disabled=not files):
            frames=[normalize(read_csv(f.getvalue()),league,season,zone,completed=kind.startswith("Completed")) for f in files]
            merged=pd.concat(frames,ignore_index=True)
            if merged.duplicated(["league","fixture_id"]).any():
                raise ValueError("Duplicate fixtures across uploaded files.")
            n=store.import_matches(merged,corrections)
            st.success(f"Imported {n} rows. Open Inspect data to review, or change pages to refresh.")
            st.dataframe(merged.head(20),use_container_width=True)
        for name in ["completed_template.csv","upcoming_template.csv"]:
            st.download_button("Download "+name,(ROOT/"examples"/name).read_bytes(),name,"text/csv")
    with tab2:
        st.write("Attach team xG using fixture_id, home_xg, away_xg and provider. Include league where IDs are ambiguous. source_url and xg_available_at are optional.")
        st.caption("Keep one provider definition per xG model version. Player-summed xG should be labelled as such. This import does not establish external validity.")
        f=st.file_uploader("Team xG CSV",type="csv",key="xg")
        allow=st.checkbox("Allow explicit xG corrections")
        if st.button("Validate and attach xG",disabled=f is None):
            n=store.attach_xg(read_csv(f.getvalue()),allow)
            st.success(f"Attached xG to {n} completed fixtures.")
        st.download_button("Download xG template",(ROOT/"examples"/"xg_template.csv").read_bytes(),"xg_template.csv","text/csv")
    with tab3:
        current=store.matches()
        if current.empty:
            st.info("No data imported.")
            return
        st.dataframe(current.groupby(["league","season"]).agg(
            fixtures=("fixture_id","size"),completed=("home_goals","count"),with_xg=("home_xg","count"),
            earliest=("kickoff","min"),latest=("kickoff","max")).reset_index(),use_container_width=True,hide_index=True)
        league=st.selectbox("Inspect league",sorted(current.league.unique()))
        subset=current.loc[current.league.eq(league)]
        st.dataframe(subset,use_container_width=True,hide_index=True)
        csv_download("Export canonical data",current,"canonical_matches.csv")
        st.caption("Updates preserve prior forecast snapshots. A changed kickoff is excluded from evaluation of the old snapshot.")

def train_page():
    st.title("Train & freeze")
    st.write("Fit XGBoost and Dixon–Coles, calibrate on a later block, rank each market on two more blocks, then freeze the selection threshold on the final validation block.")
    st.warning("Use chronological cutoffs chosen before evaluating future outcomes. Repeatedly changing these settings after seeing test results creates exploratory results.")
    if history.empty:
        st.info("Import completed matches first.")
        return
    completed=history.loc[history.home_goals.notna()]
    leagues=st.multiselect("Leagues to include in this version",sorted(completed.league.unique()),default=sorted(completed.league.unique()))
    cfg=TrainingConfig()
    cols=st.columns(5)
    fields=["train_end","cal_end","rank_mid","rank_end","threshold_end"]
    titles=["Training end","Calibration end","Ranking window 1 end","Ranking window 2 end","Threshold validation end"]
    for col,key,title in zip(cols,fields,titles):
        setattr(cfg,key,str(col.date_input(title,pd.Timestamp(getattr(cfg,key)).date(),key=key)))
    cfg.use_xg=st.checkbox("Use lagged xG alongside the six form features")
    if cfg.use_xg:
        st.info("This creates a new form+xG calibrated configuration, distinct from the raw xG ablation in the report. It needs one provider and complete lagged inputs.")
    with st.expander("Validation support settings"):
        c=st.columns(4)
        cfg.min_window=int(c[0].number_input("Minimum fixtures / block",min_value=10,value=25))
        cfg.min_picks=int(c[1].number_input("Minimum threshold picks",min_value=1,value=15))
        cfg.min_coverage=float(c[2].number_input("Minimum coverage",min_value=.01,max_value=1.,value=.2,step=.05))
        cfg.hit_target=float(c[3].number_input("Validation hit-rate target",min_value=.01,max_value=.99,value=.6,step=.05))
    st.caption("Ends are exclusive at 00:00 UTC. Models remain fitted to the training block; no automatic refit after validation. Selection uses at most one market per fixture.")
    if st.button("Train and save frozen version",type="primary",disabled=not leagues):
        status=st.empty()
        with st.spinner("Fitting models and validation rules…"):
            manifest=train_bundle(completed.loc[completed.league.isin(leagues)],cfg,MODELS,progress=status.info)
        status.success("Saved version "+manifest["bundle_id"])
        display_rankings(manifest)
        st.dataframe(pd.DataFrame(manifest["validation_grid"]),hide_index=True)
    st.divider()
    path=choose_bundle()
    if path:
        manifest,_=load_bundle(path)
        display_rankings(manifest)
        st.caption("Zero eligible markets is a valid outcome. Common policy eligibility pools the included leagues; a one-league bundle makes the policies equivalent.")
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as z:
            for p in path.iterdir():
                if p.is_file(): z.write(p,p.name)
        st.download_button("Export this model bundle",buffer.getvalue(),path.name+".zip","application/zip")

def forecasts_page():
    st.title("Upcoming forecasts")
    path=choose_bundle()
    if not path or history.empty:
        return
    manifest,models=load_bundle(path)
    st.caption(f"Version {manifest['bundle_id']} · {'Form + xG' if manifest['config']['use_xg'] else 'Form only'}")
    mode=st.radio("Forecast mode",["Live","Historical replay"],horizontal=True)
    now=pd.Timestamp.now(tz="UTC")
    if mode=="Live":
        cutoff=now
        targets=history.loc[history.home_goals.isna() & (history.kickoff>now)]
        st.caption("Live mode uses the current UTC time. No future results enter the feature history.")
    else:
        st.warning("Historical replay uses today’s stored dataset and is retrospective. It must not be reported as a live forecast.")
        cutoff=pd.to_datetime(st.text_input("Replay forecast cutoff (UTC)",manifest["config"]["threshold_end"]),utc=True)
        targets=history.loc[history.kickoff>=cutoff]
    policy=st.selectbox("Frozen selection policy",["league","common"],format_func=lambda v:"League-specific" if v=="league" else "Common markets")
    if targets.empty:
        st.info("No target fixtures. Upload upcoming fixtures or use a historical replay cutoff.")
        return
    leagues=st.multiselect("Target leagues",sorted(targets.league.unique()),default=sorted(targets.league.unique()))
    targets=targets.loc[targets.league.isin(leagues)].copy()
    if targets.empty:
        return
    end=st.date_input("Include fixtures through (UTC date)",(cutoff+pd.Timedelta(days=14)).date())
    targets=targets.loc[targets.kickoff.dt.date<=end]
    st.dataframe(targets[["league","fixture_id","kickoff","home_team","away_team"]],hide_index=True,use_container_width=True)
    st.caption("The batch uses one common cutoff: later fixtures do not gain information from games that had not finished at that cutoff. Generate a new batch nearer kickoff when appropriate.")
    if st.button("Generate and save forecasts",type="primary",disabled=targets.empty):
        with st.spinner("Building pre-cutoff features and explanations…"):
            run=generate(history,targets,manifest,models,cutoff,policy,mode)
            store.save_run(run)
            st.session_state["last_run"]=run
        st.success("Saved batch "+run["run_id"][:8])
    run=st.session_state.get("last_run")
    if run:
        st.subheader("Last generated batch")
        st.caption(f'{run["mode"]} · cutoff {run["cutoff"]} · version {run["bundle_id"]} · policy {run["policy"]}')
        st.dataframe(pd.DataFrame([{"league":f["league"],"fixture":f["home_team"]+" vs "+f["away_team"],
            "ready":f["ready"],"selection":f["selection"]["market"]+" · "+f["selection"]["outcome"] if f["selection"] else "No selection",
            "confidence":f["selection"]["confidence"] if f["selection"] else None,"reason":f["reason"]} for f in run["fixtures"]]),use_container_width=True,hide_index=True)
        st.dataframe(forecast_table(run),use_container_width=True,hide_index=True)
        csv_download("Export forecast probabilities",forecast_table(run),"forecasts_"+run["run_id"][:8]+".csv")
        st.download_button("Export complete snapshot",json.dumps(run,indent=2),run["run_id"]+".json","application/json")

def explanations_page():
    st.title("Explain a saved forecast")
    run=choose_run()
    if not run: return
    fixtures=[f for f in run["fixtures"] if f["ready"]]
    if not fixtures:
        st.info("No ready fixtures in this batch.")
        return
    i=st.selectbox("Fixture",range(len(fixtures)),format_func=lambda i:fixtures[i]["league"]+" · "+fixtures[i]["home_team"]+" vs "+fixtures[i]["away_team"])
    f=fixtures[i]
    market=st.selectbox("Market",list(MARKETS))
    m=f["markets"][market]
    st.caption(f'{f["fixture_id"]} · snapshot cutoff {run["cutoff"]} · model version {run["bundle_id"]}')
    st.subheader(f'{market}: {m["model"]} · {m["variant"]}')
    fig=go.Figure(go.Bar(x=m["labels"],y=m["probabilities"],text=[f"{p:.1%}" for p in m["probabilities"]],textposition="auto",marker_color="#3dd6aa"))
    fig.update_layout(yaxis=dict(range=[0,1],tickformat=".0%"),height=300,yaxis_title="Forecast probability")
    st.plotly_chart(fig,use_container_width=True)
    st.write("Selection:",f["selection"] if f["selection"] else "No selection")
    st.caption(f["reason"])
    tab1,tab2=st.tabs(["XGBoost · TreeSHAP","Dixon–Coles · score model"])
    with tab1:
        st.info("This explanation belongs to XGBoost. It does not explain Dixon–Coles, even when Dixon–Coles supplies the selected forecast. Contributions add to the raw margin, not the calibrated probability.")
        options=m["labels"] if len(m["labels"])==3 else [m["labels"][1]]
        outcome=st.selectbox("Explain XGBoost outcome",options)
        k=m["labels"].index(outcome) if len(m["labels"])==3 else 0
        vals=np.array(m["shap"][k]);names=list(f["features"])
        order=np.argsort(-np.abs(vals[:-1]))
        fig=go.Figure(go.Waterfall(x=["Base"]+[names[j] for j in order]+["Raw margin"],
                 y=[vals[-1]]+[vals[j] for j in order]+[0],
                 measure=["absolute"]+["relative"]*len(order)+["total"],
                 increasing=dict(marker_color="#3dd6aa"),decreasing=dict(marker_color="#ff7b7b")))
        fig.update_layout(height=450,yaxis_title="Raw model margin",showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
        st.caption(f'Additivity error: {m["additivity_error"]:.2e}. Binary margins are log-odds for the positive outcome; 1X2 uses class margins transformed jointly by softmax.')
        st.dataframe(pd.DataFrame({"outcome":m["labels"],"XGBoost raw":m["xgb_raw"],"XGBoost temperature-scaled":m["xgb_scaled"]}),hide_index=True)
        st.dataframe(pd.DataFrame({"feature":names,"value":list(f["features"].values()),"SHAP contribution":vals[:-1]}),hide_index=True)
        st.subheader("Aggregate contributions in this saved batch")
        same=[x for x in fixtures if x["league"]==f["league"]]
        mean=np.mean([np.abs(x["markets"][market]["shap"][k][:-1]) for x in same],axis=0)
        summary=pd.DataFrame({"feature":names,"mean_absolute_shap":mean}).sort_values("mean_absolute_shap")
        st.plotly_chart(px.bar(summary,x="mean_absolute_shap",y="feature",orientation="h"),use_container_width=True)
        st.caption(f'{len(same)} fixtures · {f["league"]} · one model version. Importance describes model behaviour, not causal effects.')
    with tab2:
        d=f["dc_explanation"]
        c=st.columns(2)
        c[0].metric("Home goal rate",f'{d["home_rate"]:.2f}')
        c[1].metric("Away goal rate",f'{d["away_rate"]:.2f}')
        if d["unseen_teams"]:
            st.warning("Neutral attack/defence effects used for unseen teams: "+", ".join(d["unseen_teams"]))
        grid=np.array(d["score_grid"]); visible=grid[:8,:8]
        fig=go.Figure(go.Heatmap(z=visible,x=list(range(visible.shape[1])),y=list(range(visible.shape[0])),colorscale="Teal",colorbar=dict(tickformat=".0%")))
        fig.update_layout(xaxis_title="Away goals",yaxis_title="Home goals",height=420)
        st.plotly_chart(fig,use_container_width=True)
        st.caption(f'Score distribution before market-level calibration. Displayed scorelines contain {visible.sum():.2%} of total mass. Goal rates come from Dixon–Coles, not lagged provider xG.')

def performance_page():
    st.title("Evaluate saved forecasts")
    run=choose_run()
    if not run or history.empty: return
    st.caption(f'{run["mode"]} · one batch only · version {run["bundle_id"]}')
    scores,picks,excluded=evaluate(run,history)
    if excluded: st.warning("\n".join(excluded))
    if scores.empty:
        st.info("No matching completed outcomes yet. Import results through Data & readiness.")
        return
    ready=sum(f["ready"] for f in run["fixtures"])
    settled=scores[["league","fixture_id"]].drop_duplicates().shape[0]
    c=st.columns(4)
    c[0].metric("Ready / requested",f'{ready} / {len(run["fixtures"])}')
    c[1].metric("Settled ready fixtures",settled)
    c[2].metric("Settled selections",len(picks))
    c[3].metric("Coverage on settled ready",f"{len(picks)/settled:.1%}" if settled else "—")
    if not picks.empty:
        c=st.columns(3)
        c[0].metric("Selection hit rate",f"{picks.correct.mean():.1%}")
        c[1].metric("Matched frequency hit rate",f"{picks.baseline_correct.mean():.1%}")
        c[2].metric("Paired difference",f"{100*(picks.correct-picks.baseline_correct).mean():+.1f} pp")
        interval=block_interval(picks)
        if interval is not None:
            st.caption(f"Descriptive 95% four-week block interval: hit rate {interval[0,0]:.1%}–{interval[1,0]:.1%}; paired difference {100*interval[0,1]:+.1f}–{100*interval[1,1]:+.1f} pp. Conditional on the saved models and selections.")
        st.dataframe(picks.groupby(["league","market"]).agg(picks=("correct","size"),correct=("correct","sum"),hit_rate=("correct","mean"),matched_baseline=("baseline_correct","mean")).reset_index(),hide_index=True)
    st.subheader("Probability quality")
    odds_only=st.checkbox("Compare on exact-market odds-matched fixtures only")
    sample=scores.loc[scores.odds_matched] if odds_only else scores.loc[scores.forecast.ne("Bookmaker")]
    if sample.empty:
        st.info("No complete odds for these exact markets. O/U 2.5 odds cannot benchmark O/U 1.5.")
    else:
        st.dataframe(sample.groupby(["league","market","forecast"]).agg(fixtures=("fixture_id","size"),log_loss=("log_loss","mean"),brier=("brier","mean")).reset_index(),hide_index=True,use_container_width=True)
        st.caption("Lower is better. Brier is the sum across outcomes, including both classes for binary markets. Bookmaker probabilities normalize reciprocal decimal odds; closing odds are a retrospective benchmark, not a live input.")
    st.subheader("Reliability")
    leagues=sorted(scores.league.unique())
    league=st.selectbox("Reliability league",leagues)
    market=st.selectbox("Reliability market",list(MARKETS))
    bins=reliability(scores.loc[scores.league.eq(league)],"Chosen forecast",market)
    if len(bins):
        bins["label"]=bins.outcome.map(dict(enumerate(MARKETS[market])))
        fig=px.line(bins,x="mean_probability",y="observed_frequency",color="label",markers=True,hover_data=["count"])
        fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(dash="dash",color="gray"))
        fig.update_layout(xaxis=dict(range=[0,1]),yaxis=dict(range=[0,1]))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(bins,hide_index=True)
        st.caption("Ten fixed-width bins. Small bins are unstable; this plot is not proof of calibration.")
    csv_download("Export per-fixture scores",scores.drop(columns="probabilities"),"evaluation_scores.csv")
    if not picks.empty: csv_download("Export scored selections",picks,"evaluation_selections.csv")

try:
    {"Overview":overview,"Data & readiness":data_page,"Train & freeze":train_page,
     "Forecasts":forecasts_page,"Explanations":explanations_page,"Performance":performance_page}[page]()
except Exception as exc:
    st.error(str(exc))
    with st.expander("Technical details for debugging"):
        st.code(traceback.format_exc())
