"""Start with python -m streamlit run app.py."""
from pathlib import Path
import os
import traceback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix
from engine import (read_matches,validate_matches,assess,predict_upcoming,save_result,
                    load_result,team_table,target_labels,FEATURES)
from models import MARKETS,contributions,log_loss,brier

ROOT=Path(__file__).resolve().parent
SAVED=Path(os.environ.get("SOCCER_SIMPLE_DATA",str(ROOT/"saved_models")))
st.set_page_config(page_title="Soccer Forecast · Simple",page_icon="⚽",layout="wide")
st.title("⚽ Soccer Forecast")
st.caption("Upload your results. Understand the models. Predict the next fixtures.")
if "result" not in st.session_state:
    try: st.session_state.result=load_result(SAVED)
    except Exception as e:
        st.session_state.result=None
        st.warning("Could not load the saved model. Train again from your CSV. "+str(e))
phase1,phase2=st.tabs(["1 · Train & explore","2 · Predict upcoming matches"])

def download(frame,label,filename):
    st.download_button(label,frame.to_csv(index=False).encode(),filename,"text/csv")

def explain(result,frame,market,key):
    """Explain XGBoost specifically, regardless of the selected prediction model."""
    st.caption("TreeSHAP explains XGBoost's raw margin. It does not explain a Dixon–Coles forecast or directly add up to a calibrated probability.")
    vals,margins=contributions(result["models"][market],frame[FEATURES])
    options=MARKETS[market] if market=="1X2" else [MARKETS[market][1]]
    label=st.selectbox("XGBoost outcome to explain",options,key=key+"label")
    k=MARKETS[market].index(label) if market=="1X2" else 0
    avg=np.abs(vals[:,k,:-1]).mean(axis=0)
    importance=pd.DataFrame({"Feature":FEATURES,"Mean absolute SHAP":avg}).sort_values("Mean absolute SHAP")
    st.plotly_chart(px.bar(importance,x="Mean absolute SHAP",y="Feature",orientation="h",
                          title="Which inputs influenced XGBoost most?"),use_container_width=True)
    j=st.selectbox("Inspect one match",range(len(frame)),format_func=lambda i:
        f'{frame.iloc[i].date.date()} · {frame.iloc[i].home_team} vs {frame.iloc[i].away_team}',key=key+"fixture")
    value=vals[j,k];order=np.argsort(-np.abs(value[:-1]))
    fig=go.Figure(go.Waterfall(x=["Base"]+[FEATURES[i] for i in order]+["Raw margin"],
          y=[float(value[-1])]+[float(value[i]) for i in order]+[0],
          measure=["absolute"]+["relative"]*len(order)+["total"]))
    fig.update_layout(title="How each feature moved this prediction",height=430,yaxis_title="Raw margin")
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f"Additivity error: {abs(value.sum()-margins[j,k]):.2e}. Binary margins are positive-outcome log-odds; 1X2 uses class margins with a joint softmax.")
    st.dataframe(pd.DataFrame({"Feature":FEATURES,"Observed lagged value":frame.iloc[j][FEATURES].to_numpy(),
                             "SHAP contribution":value[:-1]}),hide_index=True,use_container_width=True)

def phase_one():
    st.subheader("Upload historical match results")
    st.write("Use your existing football CSV: Date, HomeTeam, AwayTeam, FTHG and FTAG. Extra columns can stay.")
    files=st.file_uploader("Historical CSV — one season or several",type="csv",accept_multiple_files=True,key="history")
    history=None
    if files:
        frames=[read_matches(f.getvalue()) for f in files]
        history=validate_matches(pd.concat(frames,ignore_index=True))
        divisions=sorted(history.division.unique())
        division=st.selectbox("Choose a league",divisions) if len(divisions)>1 else divisions[0]
        history=history.loc[history.division.eq(division)].reset_index(drop=True)
        c=st.columns(3)
        c[0].metric("Matches",len(history))
        c[1].metric("Teams",len(set(history.home_team)|set(history.away_team)))
        c[2].metric("League",division)
        st.caption(f"History: {history.date.min().date()} to {history.date.max().date()}. Dates and leagues come directly from your CSV.")
        with st.expander("Explore the uploaded matches",expanded=True):
            counts=pd.Series(target_labels(history,"1X2")).value_counts().reindex([0,1,2],fill_value=0)
            a,b=st.columns(2)
            a.plotly_chart(px.bar(x=["Home win","Draw","Away win"],y=counts.values,
                                labels={"x":"Outcome","y":"Matches"},title="Observed match outcomes"),use_container_width=True)
            goals=history.home_goals+history.away_goals
            b.plotly_chart(px.histogram(x=goals,nbins=int(goals.max())+1,
                          labels={"x":"Total goals"},title="Goals per match"),use_container_width=True)
            teams=team_table(history)
            st.plotly_chart(px.bar(teams,x="Team",y=["Goals scored","Goals conceded"],barmode="group",
                                   title="Team scoring patterns"),use_container_width=True)
            st.dataframe(teams,hide_index=True,use_container_width=True)
            st.caption("These descriptive charts use the complete uploaded dataset. Model evaluation below uses only the final test block.")
    st.info("Automatic chronological split: roughly 60% training, 20% calibration/model choice, 20% testing. Matches on the same date stay together.")
    if st.button("Train models & show results",type="primary",disabled=history is None):
        message=st.empty()
        with st.spinner("Training and evaluating…"):
            result=assess(history,progress=message.info)
            save_result(result,SAVED)
            st.session_state.result=result
            st.session_state.pop("predictions",None)
        message.success("Models trained and saved. Results are below; Phase Two is ready.")
    result=st.session_state.result
    if result is None:
        st.caption("At least 70 feature-ready matches are needed; a full season is optional. No date settings or xG file are needed.")
        return
    st.divider()
    st.subheader("Model results on later, unseen matches")
    meta=result["meta"]
    st.caption(f"Saved model: {meta['division']} · history through {result['history'].date.max().date()} · trained {meta['created_at'][:10]}. Uploading a different file does not change this model until you click Train.")
    st.dataframe(pd.DataFrame(meta["split"]),hide_index=True,use_container_width=True)
    if any(block["fixtures"]<30 for block in meta["split"][1:]):
        st.warning("Small-sample evaluation: calibration or testing has fewer than 30 matches. You can train and predict, but scores, model choice and reliability plots are preliminary.")

    st.caption(f"{meta['skipped']} early matches excluded because a team lacked three prior matches.")
    market=st.selectbox("Market",list(MARKETS),key="eval_market")
    scores=result["scores"].loc[result["scores"].market.eq(market)]
    a,b=st.columns(2)
    a.plotly_chart(px.bar(scores,x="model",y="log_loss",title="Log loss · lower is better"),use_container_width=True)
    b.plotly_chart(px.bar(scores,x="model",y="accuracy",title="Accuracy · higher is better").update_yaxes(tickformat=".0%"),use_container_width=True)
    st.dataframe(scores.style.format({"log_loss":"{:.4f}","brier":"{:.4f}","accuracy":"{:.1%}"}),hide_index=True,use_container_width=True)
    choice=meta["choices"][market]
    st.write("Model used for upcoming forecasts:",choice)
    st.caption("Chosen using the calibration block, not these test results. A higher hit rate can simply reflect a common outcome; compare log loss and Brier against historical frequency.")
    p=result["probabilities"][market][choice]
    test=result["test"];y=target_labels(test,market);labels=MARKETS[market]
    a,b=st.columns(2)
    cm=confusion_matrix(y,p.argmax(1),labels=np.arange(len(labels)))
    a.plotly_chart(px.imshow(cm,x=labels,y=labels,text_auto=True,labels={"x":"Predicted","y":"Actual"},
                            title="Where the chosen model was right or wrong",color_continuous_scale="Blues"),use_container_width=True)
    rel=[]
    for k,label in enumerate(labels):
        bins=np.minimum((p[:,k]*10).astype(int),9)
        for j in range(10):
            mask=bins==j
            if mask.any():
                rel.append({"Outcome":label,"Predicted probability":float(p[mask,k].mean()),
                            "Observed frequency":float((y[mask]==k).mean()),"Matches":int(mask.sum())})
    rel=pd.DataFrame(rel)
    fig=px.line(rel,x="Predicted probability",y="Observed frequency",color="Outcome",markers=True,
                hover_data=["Matches"],title="Do the probabilities match what happened?")
    fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(dash="dash",color="gray"))
    fig.update_xaxes(range=[0,1]);fig.update_yaxes(range=[0,1])
    b.plotly_chart(fig,use_container_width=True)
    st.caption("The dashed line represents perfect reliability. Small test sets and sparsely populated bins make calibration estimates uncertain.")
    if market=="1X2":
        with st.expander("Compare with available closing odds"):
            odds=test[["AvgCH","AvgCD","AvgCA"]].to_numpy(dtype=float)
            mask=np.isfinite(odds).all(1)&(odds>1).all(1)
            if mask.any():
                book=1/odds[mask];book/=book.sum(1,keepdims=True)
                baseline=result["probabilities"][market]["Historical frequency"][mask]
                table=pd.DataFrame([dict(Forecast=n,Fixtures=int(mask.sum()),Log_loss=log_loss(q,y[mask]),Brier=brier(q,y[mask]))
                    for n,q in [(choice,p[mask]),("Historical frequency",baseline),("Closing odds",book)]])
                st.dataframe(table,hide_index=True)
                st.caption("Normalized reciprocal AvgCH/AvgCD/AvgCA odds on exactly the same test fixtures. Closing odds are not model inputs.")
            else: st.info("No complete AvgCH/AvgCD/AvgCA odds in the test block.")
    with st.expander("Why did XGBoost make these predictions?"):
        explain(result,test,market,"test_")
    download(result["scores"],"Download model comparison","model_comparison.csv")
    with st.expander("How this simple version works"):
        st.write("The models use each team's last three completed matches: points, goals scored, and goals conceded. Same-day games are excluded. Shots, corners, current-match scores and odds do not enter the predictors. Dixon–Coles uses earlier scores to estimate team attack/defence.")
        st.write("Calibration adjusts probabilities on a later development block. The final block is tested chronologically using results from earlier test dates as they become historical. Models are not refitted on the test block. Small one-season experiments are exploratory, not a replacement for the research study's full protocol.")

def phase_two():
    result=st.session_state.result
    st.subheader("Upload upcoming fixtures")
    if result is None:
        st.info("Train your historical CSV in Phase One first.")
        return
    last=result["history"].date.max()
    st.write(f"Using the saved **{result['meta']['division']}** model and history through **{last.date()}**.")
    st.caption("Fixture dates must be after the uploaded history. Team names must match exactly. Upload current history for current forecasts; historical files produce retrospective demonstrations.")
    # Template uses real known team spelling but is explicitly an illustrative pairing.
    teams=sorted(set(result["history"].home_team)|set(result["history"].away_team))
    template=pd.DataFrame({"Date":[(last+pd.Timedelta(days=7)).strftime("%d/%m/%Y")],
                           "HomeTeam":[teams[0]],"AwayTeam":[teams[1]]})
    download(template,"Download upcoming CSV template","upcoming_template.csv")
    st.caption("The template pairing and date are illustrative, not a published fixture.")
    upload=st.file_uploader("Upcoming CSV: Date, HomeTeam, AwayTeam (Time and Div optional)",type="csv",key="upcoming")
    fixtures=None
    if upload:
        fixtures=read_matches(upload.getvalue(),completed=False)
        st.dataframe(fixtures[["date","home_team","away_team"]],hide_index=True,use_container_width=True)
    if st.button("Predict upcoming matches",type="primary",disabled=fixtures is None):
        table,ready,details=predict_upcoming(result,fixtures)
        st.session_state.predictions=(table,ready,details)
    saved=st.session_state.get("predictions")
    if saved is None:return
    table,ready,details=saved
    st.subheader("Forecast probabilities")
    st.caption("These are the last generated predictions. Click Predict again after changing the fixture upload.")
    good=table.loc[table.Market.ne("Unavailable")]
    if good.empty:
        st.warning("No fixtures have enough matching team history.")
    else:
        top=good.loc[good.Status.eq("Most likely")]
        st.dataframe(top.style.format({"Probability":"{:.1%}"}),hide_index=True,use_container_width=True)
        st.caption("Most likely means the highest-probability outcome in each market. It is not a guaranteed result or the research study's validation-qualified betting selection.")
        market=st.selectbox("View market probabilities",list(MARKETS),key="future_market")
        plot=good.loc[good.Market.eq(market)].copy()
        plot["Fixture"]=plot.HomeTeam+" vs "+plot.AwayTeam+" · "+plot.Date
        st.plotly_chart(px.bar(plot,x="Fixture",y="Probability",color="Outcome",barmode="group",
                              title=market+" probabilities").update_yaxes(range=[0,1],tickformat=".0%"),use_container_width=True)
        with st.expander("Explain the upcoming XGBoost forecasts"):
            explain(result,ready.loc[ready.ready].reset_index(drop=True),market,"future_")
        unseen=sorted({t for d in details.values() for t in d["unseen_teams"]})
        if unseen:
            st.warning("Dixon–Coles uses neutral fitted team effects for: "+", ".join(unseen))
    bad=table.loc[table.Market.eq("Unavailable")]
    if len(bad):st.dataframe(bad[["Date","HomeTeam","AwayTeam","Status"]],hide_index=True)
    download(table,"Download all predictions","upcoming_predictions.csv")
    st.caption("Every fixture uses only completed matches in the uploaded history. Predictions are not used as results for later fixtures.")

for container,render in [(phase1,phase_one),(phase2,phase_two)]:
    with container:
        try:render()
        except Exception as exc:
            st.error(str(exc))
            with st.expander("Technical details"):st.code(traceback.format_exc())
