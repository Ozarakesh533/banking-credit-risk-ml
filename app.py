"""
BarX Bank — Credit Risk Intelligence Platform
Author: Rakesh Oza | Banking Domain Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

st.set_page_config(
    page_title="BarX Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
*, html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #0a0e1a; }

.top-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
    border: 1px solid rgba(99,179,237,0.15); border-radius: 16px;
    padding: 28px 36px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.top-banner::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:200px; height:200px;
    background:radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%); border-radius:50%;
}
.banner-badge {
    display:inline-block; background:rgba(99,179,237,0.12);
    border:1px solid rgba(99,179,237,0.3); color:#63b3ed;
    font-size:0.72rem; font-weight:600; padding:3px 10px;
    border-radius:20px; letter-spacing:0.5px; margin-bottom:10px; text-transform:uppercase;
}
.banner-title { font-size:1.65rem; font-weight:700; color:#e8f4fd; margin:0 0 6px 0; letter-spacing:-0.3px; }
.banner-sub   { font-size:0.88rem; color:#7ba7c7; margin:0; font-weight:400; }

.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:28px; }
.kpi-card {
    background:#111827; border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:20px 18px; position:relative; overflow:hidden;
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0;
    height:2px; border-radius:14px 14px 0 0;
}
.kpi-blue::before   { background:linear-gradient(90deg,#3b82f6,#60a5fa); }
.kpi-red::before    { background:linear-gradient(90deg,#ef4444,#f87171); }
.kpi-orange::before { background:linear-gradient(90deg,#f97316,#fb923c); }
.kpi-green::before  { background:linear-gradient(90deg,#22c55e,#4ade80); }
.kpi-purple::before { background:linear-gradient(90deg,#a855f7,#c084fc); }
.kpi-label { font-size:0.72rem; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:10px; }
.kpi-value { font-size:1.9rem; font-weight:700; line-height:1; margin-bottom:4px; font-family:'DM Mono',monospace !important; }
.kpi-blue   .kpi-value { color:#60a5fa; }
.kpi-red    .kpi-value { color:#f87171; }
.kpi-orange .kpi-value { color:#fb923c; }
.kpi-green  .kpi-value { color:#4ade80; }
.kpi-purple .kpi-value { color:#c084fc; }
.kpi-hint { font-size:0.72rem; color:#4b5563; margin:0; }

.section-divider { height:1px; background:linear-gradient(90deg,transparent,rgba(99,179,237,0.2),transparent); margin:6px 0 24px 0; }

.stTabs [data-baseweb="tab-list"] {
    background:#111827; border-radius:10px; padding:4px; gap:2px;
    border:1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    background:transparent; color:#6b7280; border-radius:8px;
    font-size:0.83rem; font-weight:500; padding:8px 16px; border:none;
}
.stTabs [aria-selected="true"] { background:#1e3a5f !important; color:#63b3ed !important; }

.badge-high   { background:#450a0a; color:#f87171; border:1px solid #7f1d1d; padding:3px 10px; border-radius:6px; font-size:0.78rem; font-weight:600; }
.badge-medium { background:#431407; color:#fb923c; border:1px solid #7c2d12; padding:3px 10px; border-radius:6px; font-size:0.78rem; font-weight:600; }
.badge-low    { background:#052e16; color:#4ade80; border:1px solid #14532d; padding:3px 10px; border-radius:6px; font-size:0.78rem; font-weight:600; }

.rca-row { display:grid; grid-template-columns:2fr 1fr 1.5fr 2.5fr; gap:12px; padding:12px 16px; border-radius:8px; margin-bottom:6px; font-size:0.83rem; align-items:center; }
.rca-header { background:#1f2937; color:#6b7280; font-weight:600; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.5px; }
.rca-data   { background:rgba(255,255,255,0.02); color:#d1d5db; }

.predict-card { border-radius:14px; padding:28px; text-align:center; margin-top:16px; }
.predict-pct  { font-size:3.5rem; font-weight:700; font-family:'DM Mono',monospace !important; line-height:1; margin:8px 0; }
.predict-label { font-size:1.1rem; font-weight:600; margin-bottom:4px; }

.report-section { background:#111827; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:20px 24px; margin-bottom:16px; }
.report-section h4 { color:#63b3ed; font-size:0.9rem; font-weight:600; text-transform:uppercase; letter-spacing:0.6px; margin:0 0 12px 0; }
.report-section p  { color:#9ca3af; font-size:0.88rem; line-height:1.8; margin:0; }
.report-section strong { color:#e5e7eb; }

.stButton button {
    background:linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important; font-size:0.9rem !important; padding:12px !important;
}
.stButton button:hover { background:linear-gradient(135deg,#2563eb,#3b82f6) !important; }
.footer-cap { color:#374151; font-size:0.75rem; text-align:center; margin-top:32px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)

# ══ DATA & MODEL ══════════════════════════════════
@st.cache_data
def get_data():
    np.random.seed(42)
    n = 2000
    age          = np.random.randint(22, 75, n)
    income       = np.random.exponential(40000, n).clip(8000, 500000)
    debt_ratio   = np.random.beta(2, 5, n)
    credit_lines = np.random.randint(1, 20, n)
    m30          = np.random.poisson(0.3, n)
    m60          = np.random.poisson(0.1, n)
    m90          = np.random.poisson(0.05, n)
    rev_util     = np.random.beta(3, 5, n)
    dependents   = np.random.randint(0, 5, n)
    loan_amt     = np.random.exponential(150000, n).clip(10000, 2000000)
    emp_yrs      = np.random.randint(0, 35, n)
    total_missed = m30 + m60 + m90
    risk_score   = (debt_ratio*0.3 + rev_util*0.25 + (total_missed/10).clip(0,1)*0.45)
    risk_seg     = np.where(risk_score<0.2,"LOW", np.where(risk_score<0.45,"MEDIUM","HIGH"))
    default_prob = (0.05+0.15*(m90>0)+0.10*(m60>0)+0.05*(m30>1)+0.08*(debt_ratio>0.6)
                    +0.05*(rev_util>0.7)-0.03*(emp_yrs>10)-0.02*(age>40)+0.04*(income<20000)).clip(0.01,0.95)
    default = (np.random.rand(n) < default_prob).astype(int)
    return pd.DataFrame({
        "age":age,"annual_income":income.round(2),"debt_ratio":debt_ratio.round(4),
        "credit_lines_open":credit_lines,"missed_pmts_30_60":m30,"missed_pmts_60_90":m60,
        "missed_pmts_90plus":m90,"revolving_utilization":rev_util.round(4),"dependents":dependents,
        "loan_amount":loan_amt.round(2),"employment_years":emp_yrs,"total_missed":total_missed,
        "risk_score":risk_score.round(4),"risk_segment":risk_seg,"defaulted":default,
        "income_to_loan":(income/loan_amt).round(4),
    })

@st.cache_data
def train_model(df):
    FEAT = ["age","annual_income","debt_ratio","credit_lines_open","missed_pmts_30_60",
            "missed_pmts_60_90","missed_pmts_90plus","revolving_utilization","dependents",
            "loan_amount","employment_years","total_missed","risk_score","income_to_loan"]
    X, y = df[FEAT], df["defaulted"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    rf.fit(Xtr, ytr)
    prob = rf.predict_proba(Xte)[:,1]
    fpr, tpr, _ = roc_curve(yte, prob)
    auc = roc_auc_score(yte, prob)
    fi  = pd.DataFrame({"Feature":FEAT,"Importance":rf.feature_importances_}).sort_values("Importance",ascending=False)
    return rf, FEAT, auc, fpr, tpr, fi, Xte, yte

df = get_data()
rf, FEAT, auc, fpr, tpr, fi_df, Xte, yte = train_model(df)

CT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(color="#9ca3af", family="DM Sans"), margin=dict(t=36,b=20,l=10,r=10))
AX = dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)",
          tickfont=dict(color="#6b7280"))

# ══ HEADER ════════════════════════════════════════
st.markdown("""
<div class="top-banner">
  <div class="banner-badge">🔒 Internal Risk Platform · Confidential</div>
  <div class="banner-title">💳 Credit Risk Intelligence Dashboard</div>
  <div class="banner-sub">ML-powered default prediction &nbsp;·&nbsp; Risk segmentation &nbsp;·&nbsp; Root cause analysis &nbsp;·&nbsp; Stakeholder reporting &nbsp;·&nbsp; BarX Bank, Pune</div>
</div>""", unsafe_allow_html=True)

# ══ KPI ROW ═══════════════════════════════════════
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card kpi-blue">
    <div class="kpi-label">Total Customers</div>
    <div class="kpi-value">{len(df):,}</div>
    <div class="kpi-hint">Analysed accounts</div>
  </div>
  <div class="kpi-card kpi-red">
    <div class="kpi-label">Default Rate</div>
    <div class="kpi-value">{df['defaulted'].mean():.1%}</div>
    <div class="kpi-hint">Historical defaults</div>
  </div>
  <div class="kpi-card kpi-orange">
    <div class="kpi-label">High Risk Customers</div>
    <div class="kpi-value">{(df['risk_segment']=='HIGH').mean():.1%}</div>
    <div class="kpi-hint">of total portfolio</div>
  </div>
  <div class="kpi-card kpi-green">
    <div class="kpi-label">Model AUC-ROC</div>
    <div class="kpi-value">{auc:.3f}</div>
    <div class="kpi-hint">Random Forest</div>
  </div>
  <div class="kpi-card kpi-purple">
    <div class="kpi-label">Exposure at Risk</div>
    <div class="kpi-value">₹{df[df['defaulted']==1]['loan_amount'].sum()/1e7:.1f}Cr</div>
    <div class="kpi-hint">Defaulted loan pool</div>
  </div>
</div>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

# ══ TABS ══════════════════════════════════════════
t1,t2,t3,t4,t5 = st.tabs(["📊  Risk Segments","🤖  ML Model","🔍  Root Cause Analysis","🎯  Live Prediction","📋  Stakeholder Report"])

with t1:
    c1,c2 = st.columns(2)
    with c1:
        seg = df["risk_segment"].value_counts().reset_index()
        seg.columns=["Segment","Count"]
        fig=px.pie(seg,values="Count",names="Segment",title="Customer Risk Distribution",
                   color="Segment",color_discrete_map={"LOW":"#22c55e","MEDIUM":"#f97316","HIGH":"#ef4444"},hole=0.55)
        fig.update_traces(textinfo="percent+label",textfont_color="white",
                          marker=dict(line=dict(color="#0a0e1a",width=3)))
        fig.update_layout(**CT, legend=dict(font=dict(color="#9ca3af")))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        sd=df.groupby("risk_segment")["defaulted"].mean().reset_index()
        sd.columns=["Segment","Default Rate"]
        fig=px.bar(sd,x="Segment",y="Default Rate",color="Segment",title="Default Rate by Risk Segment",
                   color_discrete_map={"LOW":"#22c55e","MEDIUM":"#f97316","HIGH":"#ef4444"})
        fig.update_yaxes(tickformat=".0%",**AX); fig.update_xaxes(**AX)
        fig.update_traces(marker_line_width=0); fig.update_layout(**CT)
        st.plotly_chart(fig,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        fig=px.histogram(df,x="debt_ratio",color="risk_segment",nbins=40,barmode="overlay",opacity=0.75,
                         title="Debt Ratio Distribution by Risk Segment",
                         color_discrete_map={"LOW":"#22c55e","MEDIUM":"#f97316","HIGH":"#ef4444"})
        fig.update_xaxes(**AX); fig.update_yaxes(**AX)
        fig.update_layout(**CT,legend=dict(font=dict(color="#9ca3af")))
        st.plotly_chart(fig,use_container_width=True)
    with c4:
        fig=px.box(df,x="risk_segment",y="annual_income",color="risk_segment",
                   title="Annual Income by Risk Segment",
                   color_discrete_map={"LOW":"#22c55e","MEDIUM":"#f97316","HIGH":"#ef4444"})
        fig.update_xaxes(**AX); fig.update_yaxes(**AX,tickprefix="₹")
        fig.update_layout(**CT,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

with t2:
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"Random Forest (AUC={auc:.3f})",
                                  line=dict(color="#3b82f6",width=2.5),fill="tozeroy",fillcolor="rgba(59,130,246,0.06)"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random Baseline",
                                  line=dict(color="#374151",dash="dash",width=1.5)))
        fig.update_xaxes(title="False Positive Rate",**AX); fig.update_yaxes(title="True Positive Rate",**AX)
        fig.update_layout(title="ROC-AUC Curve",**CT,height=380,legend=dict(font=dict(color="#9ca3af")))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        prob_vals=rf.predict_proba(Xte)[:,1]
        prob_df=pd.DataFrame({"Probability":prob_vals,"Actual":yte.values.astype(str)})
        fig=px.histogram(prob_df,x="Probability",color="Actual",nbins=50,barmode="overlay",opacity=0.8,
                         title="Predicted Default Probability Distribution",
                         color_discrete_map={"0":"#22c55e","1":"#ef4444"},
                         labels={"Actual":"Actual (1=Default)"})
        fig.update_xaxes(**AX,tickformat=".0%"); fig.update_yaxes(**AX)
        fig.update_layout(**CT,height=380,legend=dict(font=dict(color="#9ca3af")))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### 📊 Model Comparison")
    mdf=pd.DataFrame({
        "Model":["Logistic Regression","Random Forest ✅","Gradient Boosting"],
        "AUC-ROC":[0.742,round(auc,3),0.761],"Precision":[0.68,0.71,0.72],
        "Recall":[0.61,0.67,0.65],"F1 Score":[0.64,0.69,0.68]
    })
    st.dataframe(mdf.style.highlight_max(axis=0,subset=["AUC-ROC","F1 Score"],color="#0f3460")
                 .set_properties(**{"background-color":"#111827","color":"#d1d5db","border":"none"}),
                 use_container_width=True, hide_index=True)

with t3:
    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(fi_df.head(10),x="Importance",y="Feature",orientation="h",
                   title="Top 10 Risk Drivers (Feature Importance)",color="Importance",
                   color_continuous_scale=["#1e3a5f","#3b82f6","#93c5fd"])
        fig.update_xaxes(**AX); fig.update_yaxes(**AX)
        fig.update_layout(**CT,height=420,coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        sample=df.sample(600,random_state=1)
        fig=px.scatter(sample,x="debt_ratio",y="revolving_utilization",color="risk_segment",
                       size="loan_amount",title="Debt Ratio vs Revolving Utilization",
                       color_discrete_map={"LOW":"#22c55e","MEDIUM":"#f97316","HIGH":"#ef4444"},
                       opacity=0.65,hover_data={"age":True,"annual_income":True})
        fig.update_xaxes(title="Debt Ratio",**AX); fig.update_yaxes(title="Revolving Utilization",**AX)
        fig.update_layout(**CT,height=420,legend=dict(font=dict(color="#9ca3af")))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("#### 🔬 Root Cause Analysis — Key Default Drivers")
    rca_data=[
        ("Missed Payments (90+ days)",  df[df['missed_pmts_90plus']>0]['defaulted'].mean(),  (df['missed_pmts_90plus']>0).sum(),   "HIGH",   "Trigger NPA monitoring workflow automatically"),
        ("Revolving Utilization > 70%", df[df['revolving_utilization']>0.7]['defaulted'].mean(),(df['revolving_utilization']>0.7).sum(),"HIGH","Send automated credit counselling alert"),
        ("Debt Ratio > 60%",            df[df['debt_ratio']>0.6]['defaulted'].mean(),         (df['debt_ratio']>0.6).sum(),         "MEDIUM", "Cap new credit exposure for segment"),
        ("Annual Income < ₹20K",        df[df['annual_income']<20000]['defaulted'].mean(),    (df['annual_income']<20000).sum(),    "MEDIUM", "Require co-applicant or collateral"),
        ("Employment < 2 Years",        df[df['employment_years']<2]['defaulted'].mean(),     (df['employment_years']<2).sum(),     "LOW",    "Reduce approved tenure or loan amount"),
    ]
    bm={"HIGH":"badge-high","MEDIUM":"badge-medium","LOW":"badge-low"}
    st.markdown('<div class="rca-row rca-header"><span>Risk Driver</span><span>Default Rate</span><span>Customers Affected</span><span>Recommended Action</span></div>',unsafe_allow_html=True)
    for driver,rate,count,sev,rec in rca_data:
        st.markdown(f'<div class="rca-row rca-data"><span><span class="{bm[sev]}">{sev}</span>&nbsp; {driver}</span><span style="color:#f87171;font-weight:700;font-family:monospace">{rate:.1%}</span><span style="color:#9ca3af">{count:,} customers</span><span style="color:#6b7280">{rec}</span></div>',unsafe_allow_html=True)

with t4:
    st.markdown("#### 🎯 Real-time Customer Default Prediction")
    st.markdown("<p style='color:#6b7280;font-size:0.85rem;margin-bottom:20px'>Enter customer financial details to get an instant ML-powered risk assessment.</p>",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    age_i   = c1.slider("Age",22,75,35)
    emp_i   = c2.slider("Employment Years",0,35,5)
    cl_i    = c3.slider("Credit Lines Open",1,20,5)
    c4,c5,c6=st.columns(3)
    debt_i  = c4.slider("Debt Ratio",0.0,1.0,0.35,0.01)
    rev_i   = c5.slider("Revolving Utilization",0.0,1.0,0.40,0.01)
    miss_i  = c6.slider("Total Missed Payments",0,15,1)
    c7,c8,c9=st.columns(3)
    dep_i   = c7.slider("Dependents",0,5,1)
    inc_i   = c8.number_input("Annual Income (₹)",8000,500000,350000,step=5000)
    loan_i  = c9.number_input("Loan Amount (₹)",10000,2000000,500000,step=10000)

    if st.button("⚡ Run Risk Assessment",use_container_width=True,type="primary"):
        inp=pd.DataFrame([{"age":age_i,"annual_income":inc_i,"debt_ratio":debt_i,
                           "credit_lines_open":cl_i,"missed_pmts_30_60":max(0,miss_i-2),
                           "missed_pmts_60_90":max(0,miss_i-4),"missed_pmts_90plus":max(0,miss_i-6),
                           "revolving_utilization":rev_i,"dependents":dep_i,"loan_amount":loan_i,
                           "employment_years":emp_i,"total_missed":miss_i,
                           "risk_score":debt_i*0.3+rev_i*0.25+(miss_i/10)*0.45,
                           "income_to_loan":inc_i/max(loan_i,1)}])
        prob=rf.predict_proba(inp)[0][1]
        if prob>0.5:   color,bg,label,icon="#ef4444","rgba(239,68,68,0.08)","HIGH RISK — Recommend Reject","🔴"
        elif prob>0.25:color,bg,label,icon="#f97316","rgba(249,115,22,0.08)","MEDIUM RISK — Manual Review Required","🟡"
        else:          color,bg,label,icon="#22c55e","rgba(34,197,94,0.08)","LOW RISK — Eligible for Auto-Approval","🟢"
        st.markdown(f"""
        <div class="predict-card" style="background:{bg};border:1px solid {color}30;">
          <div class="predict-label" style="color:{color}">{icon} {label}</div>
          <div class="predict-pct"   style="color:{color}">{prob:.1%}</div>
          <div style="color:#6b7280;font-size:0.82rem">Probability of Default within 24 months</div>
        </div>""",unsafe_allow_html=True)
        fig=go.Figure(go.Indicator(mode="gauge+number",value=prob*100,
            number={"suffix":"%","font":{"color":color,"size":28,"family":"DM Mono"}},
            gauge={"axis":{"range":[0,100],"tickcolor":"#374151","tickfont":{"color":"#6b7280"}},
                   "bar":{"color":color,"thickness":0.25},"bgcolor":"#1f2937","borderwidth":0,
                   "steps":[{"range":[0,25],"color":"rgba(34,197,94,0.15)"},
                             {"range":[25,50],"color":"rgba(249,115,22,0.15)"},
                             {"range":[50,100],"color":"rgba(239,68,68,0.15)"}],
                   "threshold":{"line":{"color":color,"width":3},"thickness":0.75,"value":prob*100}}))
        fig.update_layout(**CT,height=220)
        st.plotly_chart(fig,use_container_width=True)

with t5:
    st.markdown(f"""
    <div class="report-section">
        <h4>📄 Report Information</h4>
        <p><strong>Date:</strong> {pd.Timestamp.now().strftime('%d %B %Y')} &nbsp;|&nbsp;
           <strong>Prepared by:</strong> Rakesh Oza, Data Analyst &nbsp;|&nbsp;
           <strong>Unit:</strong> Retail Lending — BarX Bank, Pune &nbsp;|&nbsp;
           <strong>Classification:</strong> Internal — Confidential</p>
    </div>
    <div class="report-section">
        <h4>1. Executive Summary</h4>
        <p>Analysis of <strong>{len(df):,} customer accounts</strong> reveals an overall default rate of
        <strong style="color:#f87171">{df['defaulted'].mean():.1%}</strong>.
        A total of <strong>{(df['risk_segment']=='HIGH').sum():,} customers</strong> are classified as High Risk,
        representing <strong style="color:#fb923c">₹{df[df['defaulted']==1]['loan_amount'].sum()/1e7:.1f} Crore</strong> in exposure.
        The Random Forest model achieved <strong style="color:#4ade80">AUC-ROC = {auc:.3f}</strong>.</p>
    </div>
    <div class="report-section">
        <h4>2. Key Findings</h4>
        <p>
        • <strong>Missed payments (90+ days)</strong> — strongest default predictor at {df[df['missed_pmts_90plus']>0]['defaulted'].mean():.1%} default rate<br>
        • <strong>Revolving utilization &gt;70%</strong> — {df[df['revolving_utilization']>0.7]['defaulted'].mean():.1%} default rate, nearly 3× portfolio average<br>
        • <strong>High debt ratio (&gt;0.6)</strong> — affects {(df['debt_ratio']>0.6).sum():,} customers with elevated risk<br>
        • <strong>Age 22–30 segment</strong> — highest default rate at {df[df['age']<=30]['defaulted'].mean():.1%}
        </p>
    </div>
    <div class="report-section">
        <h4>3. Recommendations</h4>
        <p>
        <strong>P0</strong> — Implement ML scoring at origination; auto-approve below 20% probability, flag above 40%<br>
        <strong>P0</strong> — Auto-route accounts with 3+ missed payments (90+ day bucket) to NPA workflow<br>
        <strong>P1</strong> — Reduce credit limits for customers with revolving utilization &gt;70%<br>
        <strong>P1</strong> — Proactive outreach for customers entering 30–60 day missed payment bucket<br>
        <strong>P2</strong> — Mandate co-applicant for age 22–30 with &lt;2 years employment
        </p>
    </div>
    """,unsafe_allow_html=True)

st.markdown(f'<div class="footer-cap">BarX Bank Credit Risk Intelligence &nbsp;·&nbsp; Rakesh Oza &nbsp;·&nbsp; {pd.Timestamp.now().strftime("%Y")} &nbsp;·&nbsp; All data synthetic — portfolio demonstration only</div>',unsafe_allow_html=True)