# =============================================================================
# CRM ANALYTICAL DASHBOARD — FINAL PROJECT
# Mata Kuliah  : Analytical CRM
# Framework    : Streamlit
# Jalankan     : streamlit run app.py
# =============================================================================
# INSTALL:
# pip install streamlit pandas numpy scikit-learn plotly scipy
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import io

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="CRM Analytical Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stMultiSelect > div,
section[data-testid="stSidebar"] .stSelectbox > div {
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
}
section[data-testid="stSidebar"] hr { border-color: #334155; }

.main { background-color: #f1f5f9; }

/* HERO HEADER */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f3460 100%);
    padding: 32px 40px; border-radius: 20px; margin-bottom: 24px;
    box-shadow: 0 12px 40px rgba(15,23,42,0.35);
    position: relative; overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,179,237,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

/* KPI */
.kpi-card {
    background: white; padding: 20px 22px; border-radius: 16px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    border-top: 4px solid #3b82f6; margin-bottom: 8px;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-title { font-size: 11px; font-weight: 700; color: #64748b;
             text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { font-size: 28px; font-weight: 800; color: #0f172a;
             font-family: 'Space Grotesk', sans-serif; margin: 4px 0; }
.kpi-sub   { font-size: 12px; color: #94a3b8; }

/* SECTION LABEL */
.sec-label {
    display: inline-block; background: linear-gradient(90deg,#3b82f6,#06b6d4);
    color: white; padding: 4px 16px; border-radius: 20px;
    font-size: 12px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 6px;
}
.sec-title {
    font-size: 20px; font-weight: 800; color: #0f172a;
    font-family: 'Space Grotesk', sans-serif;
}

/* INSIGHT CARD */
.ins-card {
    background: white; border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); height: 100%;
}
.ins-card h4 { font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 10px; }
.ins-card li { font-size: 13px; color: #374151; line-height: 2; }

/* STRATEGY CARDS */
.strat { border-radius: 14px; padding: 18px 20px; height: 100%; }
.strat h4 { font-size: 14px; font-weight: 700; margin-bottom: 10px; }
.strat p  { font-size: 13px; line-height: 1.85; }
.s-vip    { background: #fffbeb; border-top: 4px solid #f59e0b; }
.s-reg    { background: #eff6ff; border-top: 4px solid #3b82f6; }
.s-risk   { background: #fff1f2; border-top: 4px solid #ef4444; }
.s-lost   { background: #f8fafc; border-top: 4px solid #64748b; }

/* TAG BADGE */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600;
}
.badge-blue   { background:#dbeafe; color:#1d4ed8; }
.badge-green  { background:#dcfce7; color:#15803d; }
.badge-red    { background:#fee2e2; color:#dc2626; }
.badge-yellow { background:#fef9c3; color:#a16207; }

.divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    margin: 28px 0;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD & CLEAN DATA
# =============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("E-commerce_Customer_Behavior_-_Sheet1.csv")

    # FIX 1: gunakan assignment bukan inplace (pandas CoW-safe)
    df['Satisfaction Level'] = df['Satisfaction Level'].fillna(
        df['Satisfaction Level'].dropna().mode()[0]
    )
    df = df.drop_duplicates().reset_index(drop=True)
    return df

df = load_data()

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
le = LabelEncoder()
df['Gender_enc']       = le.fit_transform(df['Gender'])
df['Membership_enc']   = le.fit_transform(df['Membership Type'])
df['Satisfaction_enc'] = le.fit_transform(df['Satisfaction Level'])

# Customer Lifetime Value (CLV) — simplified model: Monetary * Frequency / Recency
df['CLV_Score'] = (
    df['Total Spend'] * df['Items Purchased'] /
    (df['Days Since Last Purchase'].replace(0, 1))
).round(2)

# =============================================================================
# RFM ANALYSIS
# =============================================================================
@st.cache_data
def build_rfm(df):
    rfm = df[['Customer ID','Days Since Last Purchase',
              'Items Purchased','Total Spend']].copy()
    rfm.columns = ['CustomerID','Recency','Frequency','Monetary']

    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4,3,2,1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1,2,3,4], duplicates='drop')
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=[1,2,3,4], duplicates='drop')

    rfm['RFM_Score'] = (rfm['R_Score'].astype(int) +
                        rfm['F_Score'].astype(int) +
                        rfm['M_Score'].astype(int))

    def segment(s):
        if s >= 10: return 'VIP / Loyal'
        elif s >= 7: return 'Regular'
        elif s >= 5: return 'At Risk'
        else: return 'Inactive'

    rfm['Segment'] = rfm['RFM_Score'].apply(segment)
    return rfm

rfm = build_rfm(df)
df  = df.merge(rfm[['CustomerID','Segment','RFM_Score','R_Score','F_Score','M_Score']],
               left_on='Customer ID', right_on='CustomerID', how='left')

# =============================================================================
# K-MEANS CLUSTERING + ELBOW/SILHOUETTE
# =============================================================================
@st.cache_data
def run_clustering(df):
    features = ['Total Spend','Days Since Last Purchase',
                'Items Purchased','Average Rating',
                'Satisfaction_enc','Membership_enc']
    X       = df[features]
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    # Elbow & Silhouette for k=2..6
    inertias, silhouettes = [], []
    k_range = range(2, 7)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_sc)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_sc, lbl))

    # Final model k=3
    km3     = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels  = km3.fit_predict(X_sc)
    sil3    = silhouette_score(X_sc, labels)
    return labels, sil3, list(k_range), inertias, silhouettes

cluster_labels_arr, sil3, k_range, inertias, silhouettes = run_clustering(df)
df['Cluster'] = cluster_labels_arr

# Auto-label clusters by avg spend
cmap = df.groupby('Cluster')['Total Spend'].mean().sort_values(ascending=False)

rank_to_label = {}

if len(cmap) >= 1:
    rank_to_label[cmap.index[0]] = 'High-Value'

if len(cmap) >= 2:
    rank_to_label[cmap.index[1]] = 'Regular'

if len(cmap) >= 3:
    rank_to_label[cmap.index[2]] = 'Low-Engagement'

df['Cluster_Label'] = df['Cluster'].map(rank_to_label)

# =============================================================================
# RETENTION RISK (multi-factor)
# =============================================================================
df['Risk_Score'] = 0
df['Risk_Score'] += (df['Days Since Last Purchase'] > 30).astype(int) * 2
df['Risk_Score'] += df['Satisfaction Level'].isin(['Unsatisfied']).astype(int) * 2
df['Risk_Score'] += df['Satisfaction Level'].isin(['Neutral']).astype(int)
df['Risk_Score'] += (df['Average Rating'] < 3).astype(int)
df['Risk_Score'] += (df['Items Purchased'] < df['Items Purchased'].quantile(0.25)).astype(int)

df['Retention_Risk']  = (df['Risk_Score'] >= 3).astype(int)
df['Risk_Level']      = pd.cut(df['Risk_Score'], bins=[-1,1,3,5,10],
                               labels=['Low','Medium','High','Critical'])

# =============================================================================
# CHURN PREDICTION — Logistic Regression
# =============================================================================
@st.cache_data
def train_churn_model(df):
    feats = ['Total Spend','Days Since Last Purchase','Items Purchased',
             'Average Rating','Satisfaction_enc','Membership_enc','Gender_enc']
    X  = df[feats]
    y  = df['Retention_Risk']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr)
    Xte_s  = scaler.transform(Xte)
    model  = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(Xtr_s, ytr)
    proba  = model.predict_proba(scaler.transform(X))[:,1]
    report = classification_report(yte, model.predict(Xte_s), output_dict=True)
    cm     = confusion_matrix(yte, model.predict(Xte_s))
    coef   = pd.Series(model.coef_[0], index=feats).sort_values(key=abs, ascending=False)
    return proba, report, cm, coef

churn_proba, churn_report, churn_cm, churn_coef = train_churn_model(df)
df['Churn_Probability'] = (churn_proba * 100).round(1)

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("## ⚙️ Filter & Konfigurasi")
st.sidebar.markdown("---")

# FIX 2: dropna() sebelum sorted() agar tidak error float vs str
membership_filter   = st.sidebar.multiselect("🏅 Membership Type",
    options=sorted(df['Membership Type'].dropna().unique()),
    default=sorted(df['Membership Type'].dropna().unique()))

gender_filter       = st.sidebar.multiselect("👤 Gender",
    options=sorted(df['Gender'].dropna().unique()),
    default=sorted(df['Gender'].dropna().unique()))

satisfaction_filter = st.sidebar.multiselect("😊 Satisfaction Level",
    options=sorted(df['Satisfaction Level'].dropna().unique()),
    default=sorted(df['Satisfaction Level'].dropna().unique()))

segment_filter      = st.sidebar.multiselect("🎯 RFM Segment",
    options=['VIP / Loyal','Regular','At Risk','Inactive'],
    default=['VIP / Loyal','Regular','At Risk','Inactive'])

spend_range = st.sidebar.slider("💰 Total Spend ($)",
    int(df['Total Spend'].min()), int(df['Total Spend'].max()),
    (int(df['Total Spend'].min()), int(df['Total Spend'].max())))

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='background:#1e3a5f;border-radius:10px;padding:12px 14px;'>
<p style='color:#93c5fd;font-size:12px;font-weight:700;margin:0'>📐 Model Metrics</p>
<p style='color:#e2e8f0;font-size:12px;margin:6px 0 2px'>Silhouette Score K-Means</p>
<p style='color:#34d399;font-size:18px;font-weight:800;margin:0'>{sil3:.3f}</p>
<p style='color:#94a3b8;font-size:11px;margin:4px 0 8px'>Mendekati 1 = cluster valid</p>
<p style='color:#e2e8f0;font-size:12px;margin:2px 0'>Accuracy Churn Model</p>
<p style='color:#f59e0b;font-size:18px;font-weight:800;margin:0'>{churn_report['accuracy']*100:.1f}%</p>
<p style='color:#94a3b8;font-size:11px;margin:4px 0 0'>Logistic Regression</p>
</div>""", unsafe_allow_html=True)

# Apply filters
fdf = df[
    df['Membership Type'].isin(membership_filter) &
    df['Gender'].isin(gender_filter) &
    df['Satisfaction Level'].isin(satisfaction_filter) &
    df['Segment'].isin(segment_filter) &
    df['Total Spend'].between(spend_range[0], spend_range[1])
].copy()
if len(fdf) == 0:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
    st.stop()

# =============================================================================
# HERO
# =============================================================================
st.markdown("""
<div class="hero">
  <h1 style="color:white;margin:0;font-size:26px;font-weight:800;
             font-family:'Space Grotesk',sans-serif;">
    📊 CRM Analytical Dashboard
  </h1>
  <p style="color:#93c5fd;margin:6px 0 2px;font-size:14px;">
    Customer Segmentation · Retention Risk · Churn Prediction · CLV Analysis
  </p>
  <p style="color:#475569;font-size:12px;margin:0;">
    Universitas Pembangunan Nasional "Veteran" Jawa Timur &nbsp;·&nbsp; Analytical CRM Final Project
  </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# TAB NAVIGATION
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview & KPI",
    "👥 Segmentasi RFM",
    "🔬 K-Means Clustering",
    "⚠️ Retention & Churn",
    "💰 CLV Analysis",
    "🎯 Strategy & Insight"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW & KPI
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    # KPI Row
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpi_data = [
        (k1, "Total Customer",   f"{len(fdf):,}",        "📊 Data aktif",       "#3b82f6"),
        (k2, "Avg. Total Spend", f"${fdf['Total Spend'].mean():,.0f}", "💵 Per customer", "#10b981"),
        (k3, "Avg. Rating",      f"{fdf['Average Rating'].mean():.2f} ⭐", "Kepuasan produk","#f59e0b"),
        (k4, "VIP Customer",     f"{len(fdf[fdf['Segment']=='VIP / Loyal']):,}", "🏆 Loyal customer","#8b5cf6"),
        (k5, "Retention Risk",   f"{fdf['Retention_Risk'].mean()*100:.1f}%",
                                 f"⚠️ {fdf['Retention_Risk'].sum():,} orang","#ef4444"),
        (k6, "Avg CLV Score",    f"{fdf['CLV_Score'].mean():.1f}", "💎 Customer value","#06b6d4"),
    ]
    for col, title, val, sub, color in kpi_data:
        with col:
            st.markdown(f"""<div class="kpi-card" style="border-top-color:{color};">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Overview charts
    c1, c2 = st.columns(2)
    with c1:
        # Membership distribution
        mem_cnt = fdf['Membership Type'].value_counts().reset_index()
        mem_cnt.columns = ['Membership','Count']
        fig = px.bar(mem_cnt, x='Membership', y='Count', color='Membership',
                     title='Distribusi Membership Type', text='Count',
                     color_discrete_sequence=['#3b82f6','#10b981','#f59e0b'])
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor='white',
                          paper_bgcolor='white', title_font_size=14,
                          margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Satisfaction distribution
        sat_cnt = fdf['Satisfaction Level'].value_counts().reset_index()
        sat_cnt.columns = ['Level','Count']
        fig2 = px.pie(sat_cnt, names='Level', values='Count',
                      hole=0.5, title='Distribusi Satisfaction Level',
                      color_discrete_sequence=['#10b981','#3b82f6','#ef4444','#f59e0b'])
        fig2.update_traces(textinfo='percent+label', pull=[0.04]*len(sat_cnt))
        fig2.update_layout(paper_bgcolor='white', title_font_size=14,
                           margin=dict(t=50,b=10,l=10,r=10),
                           legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Spend distribution histogram
        fig3 = px.histogram(fdf, x='Total Spend', nbins=30, color='Membership Type',
                            title='Distribusi Total Spend per Membership',
                            color_discrete_sequence=['#3b82f6','#10b981','#f59e0b'],
                            opacity=0.8, barmode='overlay')
        fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                           title_font_size=14, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Gender × Membership sunburst
        sun_data = fdf.groupby(['Gender','Membership Type']).size().reset_index(name='Count')
        fig4 = px.sunburst(sun_data, path=['Gender','Membership Type'], values='Count',
                           title='Komposisi Gender × Membership',
                           color_discrete_sequence=px.colors.qualitative.Set2)
        fig4.update_layout(paper_bgcolor='white', title_font_size=14,
                           margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig4, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### 🔗 Correlation Matrix Fitur Numerik")
    num_cols = ['Total Spend','Items Purchased','Average Rating',
                'Days Since Last Purchase','CLV_Score','RFM_Score',
                'Satisfaction_enc','Membership_enc']
    corr = fdf[num_cols].corr().round(2)
    fig5 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, title='Correlation Heatmap',
                     aspect='auto')
    fig5.update_layout(paper_bgcolor='white', title_font_size=14,
                       margin=dict(t=50,b=10,l=10,r=10), height=400)
    st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RFM SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<span class="sec-label">RFM ANALYSIS</span>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Recency · Frequency · Monetary Segmentation</p>', unsafe_allow_html=True)
    st.markdown("")

    r1, r2, r3 = st.columns(3)

    with r1:
        seg_order  = ['VIP / Loyal','Regular','At Risk','Inactive']
        seg_colors = ['#8b5cf6','#3b82f6','#f59e0b','#ef4444']
        seg_cnt = (fdf['Segment'].value_counts()
                   .reindex(seg_order, fill_value=0).reset_index())
        seg_cnt.columns = ['Segment','Count']
        seg_cnt['Pct'] = (seg_cnt['Count']/seg_cnt['Count'].sum()*100).round(1)

        fig = px.bar(seg_cnt, x='Segment', y='Count', color='Segment',
                     color_discrete_sequence=seg_colors,
                     title='Jumlah Customer per Segment',
                     text=seg_cnt['Count'].astype(str)+'<br>('+seg_cnt['Pct'].astype(str)+'%)')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor='white',
                          paper_bgcolor='white', title_font_size=14,
                          margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        fig2 = px.box(fdf, x='Segment', y='Total Spend', color='Segment',
                      color_discrete_sequence=seg_colors,
                      title='Distribusi Spend per Segment',
                      points='outliers',
                      category_orders={'Segment': seg_order})
        fig2.update_layout(showlegend=False, plot_bgcolor='white',
                           paper_bgcolor='white', title_font_size=14,
                           margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    with r3:
        heat = (fdf.groupby(['Membership Type','Segment']).size()
                .reset_index(name='Count')
                .pivot(index='Membership Type', columns='Segment', values='Count')
                .fillna(0).reindex(columns=seg_order))
        fig3 = px.imshow(heat, text_auto=True,
                         color_continuous_scale='Blues',
                         title='Heatmap Membership × Segment', aspect='auto')
        fig3.update_layout(paper_bgcolor='white', title_font_size=14,
                           coloraxis_showscale=False,
                           margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig3, use_container_width=True)

    # RFM 3D Scatter
    rfm_full = rfm.merge(fdf[['Customer ID','Segment']],
                         left_on='CustomerID', right_on='Customer ID', how='inner')
    fig4 = px.scatter_3d(rfm_full, x='Recency', y='Frequency', z='Monetary',
                         color='Segment_x' if 'Segment_x' in rfm_full.columns else 'Segment',
                         color_discrete_sequence=seg_colors,
                         title='3D RFM Scatter — Recency × Frequency × Monetary',
                         opacity=0.7, size_max=6,
                         labels={'Recency':'Recency (hari)','Frequency':'Frequency','Monetary':'Monetary ($)'})
    fig4.update_layout(paper_bgcolor='white', title_font_size=14, height=500,
                       legend=dict(orientation='h', y=-0.1),
                       margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig4, use_container_width=True)

    # RFM Stats Table
    st.markdown("#### 📋 Statistik RFM per Segment")
    rfm_stat = rfm.groupby('Segment').agg(
        Jumlah_Customer=('CustomerID','count'),
        Avg_Recency=('Recency','mean'),
        Avg_Frequency=('Frequency','mean'),
        Avg_Monetary=('Monetary','mean'),
        Avg_RFM_Score=('RFM_Score','mean')
    ).round(2).reindex(seg_order)
    rfm_stat.columns = ['Jumlah Customer','Avg Recency (hari)','Avg Frequency',
                        'Avg Monetary ($)','Avg RFM Score']
    st.dataframe(rfm_stat.style.background_gradient(cmap='Blues', axis=0)
                 .format({'Avg Recency (hari)':'{:.1f}','Avg Frequency':'{:.1f}',
                          'Avg Monetary ($)':'${:,.1f}','Avg RFM Score':'{:.2f}'}),
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<span class="sec-label">UNSUPERVISED LEARNING</span>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">K-Means Clustering Analysis</p>', unsafe_allow_html=True)
    st.markdown("")

    e1, e2 = st.columns(2)
    with e1:
        # Elbow chart
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                                       line=dict(color='#3b82f6', width=2.5),
                                       marker=dict(size=8, color='#3b82f6'),
                                       name='Inertia (WCSS)'))
        fig_elbow.update_layout(title='Elbow Method — Optimal K',
                                xaxis_title='Jumlah Cluster (K)',
                                yaxis_title='Within-Cluster Sum of Squares',
                                plot_bgcolor='white', paper_bgcolor='white',
                                title_font_size=14, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with e2:
        # Silhouette chart
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=list(k_range), y=silhouettes, mode='lines+markers',
                                     line=dict(color='#10b981', width=2.5),
                                     marker=dict(size=8, color='#10b981'),
                                     name='Silhouette Score'))
        fig_sil.add_vline(x=3, line_dash='dash', line_color='#ef4444',
                          annotation_text=f'K=3 dipilih ({sil3:.3f})',
                          annotation_position='top right')
        fig_sil.update_layout(title='Silhouette Score per K — Validasi Cluster',
                              xaxis_title='K', yaxis_title='Silhouette Score',
                              plot_bgcolor='white', paper_bgcolor='white',
                              title_font_size=14, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_sil, use_container_width=True)

    c1, c2 = st.columns([1.3,1])
    with c1:
        # Scatter Spend vs Days colored by cluster
        fig_sc = px.scatter(fdf, x='Total Spend', y='Days Since Last Purchase',
                            color='Cluster_Label', size='Items Purchased',
                            hover_data=['Membership Type','Satisfaction Level','Segment'],
                            color_discrete_sequence=['#3b82f6','#10b981','#f59e0b'],
                            title='Customer Behavior Map — K-Means Clusters',
                            opacity=0.75,
                            labels={'Total Spend':'Total Spend ($)',
                                    'Days Since Last Purchase':'Hari Sejak Pembelian'})
        fig_sc.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                             title_font_size=14, margin=dict(t=50,b=10,l=10,r=10),
                             legend=dict(orientation='h',y=-0.2))
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        # Radar chart cluster profile
        profile_cols = ['Total Spend','Days Since Last Purchase',
                        'Items Purchased','Average Rating','CLV_Score']
        cp = fdf.groupby('Cluster_Label')[profile_cols].mean().reset_index()
        for col in profile_cols:
            mx = cp[col].max()
            cp[col+'_n'] = cp[col]/mx if mx > 0 else 0
        cats  = profile_cols
        colors= ['#3b82f6','#10b981','#f59e0b']
        fig_r = go.Figure()
        for i, row in cp.iterrows():
            vals = [row[c+'_n'] for c in cats] + [row[cats[0]+'_n']]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats+[cats[0]],
                fill='toself', name=row['Cluster_Label'],
                line_color=colors[i%3], opacity=0.7
            ))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                            title='Radar — Profil Cluster (Normalized)',
                            font_family='Inter', paper_bgcolor='white',
                            title_font_size=14, height=380,
                            margin=dict(t=50,b=30,l=10,r=10))
        st.plotly_chart(fig_r, use_container_width=True)

    # Cluster stats table
    st.markdown("#### 📋 Statistik Cluster")
    cs = fdf.groupby('Cluster_Label').agg(
        Jumlah=('Customer ID','count'),
        Avg_Spend=('Total Spend','mean'),
        Avg_Recency=('Days Since Last Purchase','mean'),
        Avg_Items=('Items Purchased','mean'),
        Avg_Rating=('Average Rating','mean'),
        Avg_CLV=('CLV_Score','mean')
    ).round(2)
    cs.columns = ['Jumlah Customer','Avg Spend ($)','Avg Recency (hari)',
                  'Avg Items','Avg Rating','Avg CLV Score']
    st.dataframe(cs.style.background_gradient(cmap='Blues', axis=0)
                 .format({'Avg Spend ($)':'${:,.1f}','Avg Recency (hari)':'{:.1f}',
                          'Avg Items':'{:.1f}','Avg Rating':'{:.2f}','Avg CLV Score':'{:.1f}'}),
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RETENTION & CHURN PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<span class="sec-label">PREDICTIVE ANALYTICS</span>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Retention Risk & Churn Prediction Model</p>', unsafe_allow_html=True)
    st.markdown("")

    # Risk level breakdown
    rk1, rk2, rk3, rk4 = st.columns(4)
    for col, level, color in [
        (rk1,'Low','#10b981'), (rk2,'Medium','#f59e0b'),
        (rk3,'High','#f97316'), (rk4,'Critical','#ef4444')
    ]:
        cnt = len(fdf[fdf['Risk_Level']==level])
        pct = cnt/len(fdf)*100 if len(fdf) > 0 else 0
        with col:
            st.markdown(f"""<div class="kpi-card" style="border-top-color:{color};">
                <div class="kpi-title">{level} Risk</div>
                <div class="kpi-value" style="font-size:24px;">{cnt:,}</div>
                <div class="kpi-sub">{pct:.1f}% dari total</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    ch1, ch2 = st.columns(2)

    with ch1:
        # Risk level by membership
        rl_mem = (fdf.groupby(['Membership Type','Risk_Level']).size()
                  .reset_index(name='Count'))
        fig = px.bar(rl_mem, x='Membership Type', y='Count', color='Risk_Level',
                     barmode='stack', title='Risk Level per Membership Type',
                     color_discrete_map={'Low':'#10b981','Medium':'#f59e0b',
                                         'High':'#f97316','Critical':'#ef4444'},
                     category_orders={'Risk_Level':['Low','Medium','High','Critical']})
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          title_font_size=14, margin=dict(t=50,b=10,l=10,r=10),
                          legend=dict(orientation='h',y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        # Churn probability distribution
        fig2 = px.histogram(fdf, x='Churn_Probability', nbins=20,
                            color='Risk_Level',
                            color_discrete_map={'Low':'#10b981','Medium':'#f59e0b',
                                                'High':'#f97316','Critical':'#ef4444'},
                            title='Distribusi Churn Probability (Logistic Regression)',
                            labels={'Churn_Probability':'Churn Probability (%)'},
                            opacity=0.85)
        fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                           title_font_size=14, margin=dict(t=50,b=10,l=10,r=10),
                           legend=dict(orientation='h',y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

    ch3, ch4 = st.columns(2)
    with ch3:
        # Confusion matrix
        fig_cm = px.imshow(churn_cm, text_auto=True,
                           labels=dict(x='Predicted', y='Actual'),
                           x=['No Risk','High Risk'], y=['No Risk','High Risk'],
                           color_continuous_scale='Blues',
                           title='Confusion Matrix — Churn Model')
        fig_cm.update_layout(paper_bgcolor='white', title_font_size=14,
                             margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with ch4:
        # Feature importance (coefficients)
        coef_df = churn_coef.reset_index()
        coef_df.columns = ['Feature','Coefficient']
        coef_df['Direction'] = coef_df['Coefficient'].apply(lambda x: 'Positif (↑ Risk)' if x>0 else 'Negatif (↓ Risk)')
        fig_coef = px.bar(coef_df, x='Coefficient', y='Feature',
                          orientation='h', color='Direction',
                          color_discrete_map={'Positif (↑ Risk)':'#ef4444','Negatif (↓ Risk)':'#10b981'},
                          title='Feature Importance — Logistic Regression Coefficients')
        fig_coef.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               title_font_size=14, margin=dict(t=50,b=10,l=10,r=10),
                               legend=dict(orientation='h',y=-0.2),
                               yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_coef, use_container_width=True)

    # Model performance metrics
    st.markdown("#### 📊 Classification Report")
    cr = churn_report
    cr_df = pd.DataFrame({
    'Class': ['No Risk (0)', 'High Risk (1)', 'Macro Avg', 'Weighted Avg'],
    'Precision': [
        cr['0']['precision'],
        cr['1']['precision'],
        cr['macro avg']['precision'],
        cr['weighted avg']['precision']
    ],
    'Recall': [
        cr['0']['recall'],
        cr['1']['recall'],
        cr['macro avg']['recall'],
        cr['weighted avg']['recall']
    ],
    'F1-Score': [
        cr['0']['f1-score'],
        cr['1']['f1-score'],
        cr['macro avg']['f1-score'],
        cr['weighted avg']['f1-score']
    ],
    'Support': [
        int(cr['0']['support']),
        int(cr['1']['support']),
        np.nan,
        np.nan
    ]
})
    st.dataframe(cr_df.style.format({'Precision':'{:.3f}','Recall':'{:.3f}','F1-Score':'{:.3f}'},
                                    na_rep='')
                 .background_gradient(cmap='Greens', subset=['Precision','Recall','F1-Score']),
                 use_container_width=True)

    # Top at-risk customers
    st.markdown("#### 🔴 Top 15 Customer Berisiko Churn Tertinggi")
    at_risk = (fdf.sort_values('Churn_Probability', ascending=False)
               [['Customer ID','Membership Type','Satisfaction Level',
                 'Total Spend','Days Since Last Purchase','Segment',
                 'Risk_Level','Churn_Probability']]
               .head(15).reset_index(drop=True))
    st.dataframe(at_risk.style
                 .background_gradient(cmap='Reds', subset=['Churn_Probability'])
                 .format({'Total Spend':'${:,.0f}','Churn_Probability':'{:.1f}%'}),
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CLV ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<span class="sec-label">VALUE ANALYSIS</span>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Customer Lifetime Value (CLV) Analysis</p>', unsafe_allow_html=True)
    st.markdown("")

    cl1, cl2 = st.columns(2)
    with cl1:
        # CLV per segment
        clv_seg = fdf.groupby('Segment')['CLV_Score'].mean().reindex(
            ['VIP / Loyal','Regular','At Risk','Inactive']).reset_index()
        clv_seg.columns = ['Segment','Avg_CLV']
        fig = px.funnel(clv_seg, x='Avg_CLV', y='Segment',
                        title='CLV Funnel per RFM Segment',
                        color_discrete_sequence=['#8b5cf6'])
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          title_font_size=14, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with cl2:
        # CLV vs spend scatter
        fig2 = px.scatter(
    fdf,
    x='Total Spend',
    y='CLV_Score',
    color='Segment',
    size='Items Purchased',
    color_discrete_sequence=[
        '#8b5cf6',
        '#3b82f6',
        '#f59e0b',
        '#ef4444'
    ],
    title='CLV Score vs Total Spend',
    labels={
        'CLV_Score':'CLV Score',
        'Total Spend':'Total Spend ($)'
    }
)
        fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                           title_font_size=14, margin=dict(t=50,b=10,l=10,r=10),
                           legend=dict(orientation='h',y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

    cl3, cl4 = st.columns(2)
    with cl3:
        # CLV per membership
        clv_mem = fdf.groupby('Membership Type')['CLV_Score'].describe()[['mean','std','min','50%','max']].round(2)
        clv_mem.columns = ['Mean CLV','Std Dev','Min','Median','Max']
        st.markdown("#### 📋 CLV Statistik per Membership")
        st.dataframe(clv_mem.style.background_gradient(cmap='Purples', axis=0)
                     .format('{:.2f}'), use_container_width=True)

    with cl4:
        # CLV distribution by membership
        fig3 = px.violin(fdf, x='Membership Type', y='CLV_Score',
                         color='Membership Type', box=True,
                         color_discrete_sequence=['#3b82f6','#10b981','#f59e0b'],
                         title='Distribusi CLV per Membership (Violin + Box)')
        fig3.update_layout(showlegend=False, plot_bgcolor='white',
                           paper_bgcolor='white', title_font_size=14,
                           margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig3, use_container_width=True)

    # Pareto: top 20% customer → berapa % revenue?
    st.markdown("#### 📊 Pareto Analysis — Customer Value Distribution")
    pareto = fdf[['Customer ID','Total Spend']].sort_values('Total Spend', ascending=False).copy()
    pareto['Cumulative_Spend'] = pareto['Total Spend'].cumsum()
    pareto['Cumulative_Pct']   = pareto['Cumulative_Spend'] / pareto['Total Spend'].sum() * 100
    pareto['Customer_Rank']    = range(1, len(pareto)+1)
    pareto['Customer_Pct']     = pareto['Customer_Rank'] / len(pareto) * 100

    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(x=pareto['Customer_Pct'], y=pareto['Total Spend'],
                           name='Individual Spend', marker_color='#dbeafe'))
    fig_p.add_trace(go.Scatter(x=pareto['Customer_Pct'], y=pareto['Cumulative_Pct'],
                               name='Cumulative Spend %', yaxis='y2',
                               line=dict(color='#3b82f6', width=2.5)))
    fig_p.add_vline(x=20, line_dash='dash', line_color='#ef4444',
                    annotation_text='Top 20% Customer')
    fig_p.update_layout(
        title='Pareto Chart — Revenue Contribution',
        xaxis_title='Customer Rank (%)', yaxis_title='Total Spend ($)',
        yaxis2=dict(title='Cumulative Spend (%)', overlaying='y', side='right', range=[0,110]),
        plot_bgcolor='white', paper_bgcolor='white',
        title_font_size=14, margin=dict(t=50,b=10,l=10,r=10),
        legend=dict(orientation='h', y=-0.2), height=380
    )
    top20_pct = pareto[pareto['Customer_Pct'] <= 20]['Cumulative_Pct'].max()
    st.plotly_chart(fig_p, use_container_width=True)
    st.info(f"📌 **Pareto Finding**: Top 20% customer berkontribusi pada **{top20_pct:.1f}%** total revenue — "
            f"strategi retensi harus diprioritaskan untuk kelompok ini.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — STRATEGY & INSIGHT
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<span class="sec-label">BUSINESS INTELLIGENCE</span>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">CRM Strategy & Business Insight</p>', unsafe_allow_html=True)
    st.markdown("")

    # Dynamic insight cards
    top_seg   = fdf['Segment'].value_counts().idxmax() if len(fdf) > 0 else '-'
    top_clust = fdf['Cluster_Label'].value_counts().idxmax() if len(fdf) > 0 else '-'
    risk_mem  = (fdf.groupby('Membership Type')['Retention_Risk'].mean().idxmax()
                 if len(fdf) > 0 else '-')
    avg_churn_vip  = fdf[fdf['Segment']=='VIP / Loyal']['Churn_Probability'].mean()
    avg_churn_risk = fdf[fdf['Segment']=='At Risk']['Churn_Probability'].mean()
    top20_rev  = pareto[pareto['Customer_Pct'] <= 20]['Cumulative_Pct'].max()

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(f"""<div class="ins-card">
        <h4>🎯 Segmentasi Customer</h4><ul>
        <li>Segmen terbesar: <b>{top_seg}</b></li>
        <li>Cluster dominan: <b>{top_clust}</b></li>
        <li>VIP customer: <b>{len(fdf[fdf['Segment']=='VIP / Loyal']):,}</b> orang</li>
        <li>Silhouette K-Means: <b>{sil3:.3f}</b> ✅ cluster valid</li>
        <li>Top 20% customer → <b>{top20_rev:.1f}%</b> revenue (Pareto)</li>
        </ul></div>""", unsafe_allow_html=True)

    with i2:
        st.markdown(f"""<div class="ins-card">
        <h4>⚠️ Retention & Churn Risk</h4><ul>
        <li>Membership paling berisiko: <b>{risk_mem}</b></li>
        <li>Churn prob. VIP customer: <b>{avg_churn_vip:.1f}%</b> (aman)</li>
        <li>Churn prob. At Risk: <b>{avg_churn_risk:.1f}%</b> (kritis)</li>
        <li>Accuracy churn model: <b>{churn_report['accuracy']*100:.1f}%</b></li>
        <li>Total customer high risk: <b>{fdf['Retention_Risk'].sum():,}</b></li>
        </ul></div>""", unsafe_allow_html=True)

    with i3:
        st.markdown(f"""<div class="ins-card">
        <h4>💰 Customer Value</h4><ul>
        <li>Avg CLV score: <b>{fdf['CLV_Score'].mean():.1f}</b></li>
        <li>Avg spend VIP: <b>${fdf[fdf['Segment']=='VIP / Loyal']['Total Spend'].mean():,.0f}</b></li>
        <li>Avg spend Inactive: <b>${fdf[fdf['Segment']=='Inactive']['Total Spend'].mean():,.0f}</b></li>
        <li>Tren: Days Since Purchase ↑ → CLV ↓</li>
        <li>Membership Gold → CLV tertinggi</li>
        </ul></div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 🎯 CRM Strategy Matrix per Segment")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""<div class="strat s-vip">
        <h4 style="color:#92400e;">🏆 VIP / Loyal Customer</h4>
        <p>✦ Loyalty Points Program<br>
        ✦ Priority Customer Service<br>
        ✦ Exclusive Early Access<br>
        ✦ VIP Event & Birthday Reward<br>
        ✦ Personalized Recommendation<br>
        ✦ NPS Survey Berkala</p>
        </div>""", unsafe_allow_html=True)

    with s2:
        st.markdown("""<div class="strat s-reg">
        <h4 style="color:#1e40af;">🛒 Regular Customer</h4>
        <p>✦ Product Recommendation Engine<br>
        ✦ Bundle & Cross-sell Promo<br>
        ✦ Membership Upselling (→ Gold)<br>
        ✦ Weekly Personalized Newsletter<br>
        ✦ Referral Bonus Program<br>
        ✦ Flash Sale Early Access</p>
        </div>""", unsafe_allow_html=True)

    with s3:
        st.markdown("""<div class="strat s-risk">
        <h4 style="color:#991b1b;">⚠️ At Risk Customer</h4>
        <p>✦ Personalized Discount (15-20%)<br>
        ✦ Automated Reminder Email<br>
        ✦ Cashback & Voucher Campaign<br>
        ✦ Satisfaction Survey + Follow-up<br>
        ✦ Customer Service Outreach<br>
        ✦ Re-engagement Push Notif</p>
        </div>""", unsafe_allow_html=True)

    with s4:
        st.markdown("""<div class="strat s-lost">
        <h4 style="color:#374151;">🔄 Inactive / Lost</h4>
        <p>✦ Win-Back Campaign<br>
        ✦ Reactivation Voucher (25-30%)<br>
        ✦ "We Miss You" Email Series<br>
        ✦ Sunset Policy & Data Review<br>
        ✦ Exit Survey / Feedback Form<br>
        ✦ Retargeting Ads</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Project limitation
    with st.expander("⚠️ Keterbatasan Proyek & Metodologi", expanded=False):
        col_l, col_m = st.columns(2)
        with col_l:
            st.markdown("""
**Keterbatasan:**
- Dataset tidak memiliki label churn asli → unsupervised approach
- Retention Risk dibuat menggunakan multi-factor business rule
- Churn model hanya Logistic Regression → bisa ditingkatkan dengan Random Forest / XGBoost
- CLV menggunakan simplified formula (bukan BG/NBD model)
- Tidak ada data temporal → cohort analysis tidak dapat dilakukan penuh
- K-Means sensitif terhadap outlier dan inisialisasi awal
""")
        with col_m:
            st.markdown("""
**Metodologi:**
- **RFM**: Quartile-based scoring → 4 segment (VIP/Regular/At Risk/Inactive)
- **K-Means**: 3 cluster, StandardScaler, n_init=10, validasi Silhouette Score
- **Retention Risk**: Multi-factor rule (Recency, Satisfaction, Rating, Frequency)
- **Churn Prediction**: Logistic Regression, train/test split 75:25, stratified
- **CLV**: Simplified → Total Spend × Frequency / Recency
- **Pareto**: Cumulative revenue by sorted customer spend
""")

    # Data download
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### 📂 Export Data Terfilter")
    ec1, ec2 = st.columns(2)
    with ec1:
        csv_data = fdf[['Customer ID','Gender','Membership Type','Total Spend',
                         'Items Purchased','Average Rating','Days Since Last Purchase',
                         'Satisfaction Level','Segment','Cluster_Label',
                         'Risk_Level','Churn_Probability','CLV_Score']].to_csv(index=False)
        st.download_button("⬇️ Download CSV (Filtered Data)",
                           data=csv_data, file_name='crm_filtered_data.csv',
                           mime='text/csv', use_container_width=True)
    with ec2:
        at_risk_csv = (fdf[fdf['Retention_Risk']==1]
                       .sort_values('Churn_Probability', ascending=False)
                       [['Customer ID','Membership Type','Satisfaction Level',
                         'Total Spend','Churn_Probability','Risk_Level']]
                       .to_csv(index=False))
        st.download_button("⬇️ Download High-Risk Customer List",
                           data=at_risk_csv, file_name='crm_high_risk_customers.csv',
                           mime='text/csv', use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);
            border-radius:16px;padding:20px 32px;text-align:center;">
  <p style="color:#93c5fd;margin:0;font-size:13px;font-weight:700;">
    📊 CRM Analytical Dashboard — Final Project
  </p>
  <p style="color:#475569;margin:4px 0 0;font-size:12px;">
    Universitas Pembangunan Nasional "Veteran" Jawa Timur
    &nbsp;·&nbsp; Mata Kuliah Analytical CRM
  </p>
</div>
""", unsafe_allow_html=True)