"""
╔══════════════════════════════════════════════════════════════╗
║     CardioInsight — Heart Disease Analytics Dashboard        ║
║     Dataset : UCI Heart Disease (Cleveland) — Real Data      ║
║     Deploy  : Hugging Face Spaces (Dash SDK)                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, classification_report,
)

# ──────────────────────────────────────────────────────────────
# 1.  LOAD REAL UCI HEART DISEASE DATASET
#     Primary  : fetch from UCI ML Repository (CSV mirror)
#     Fallback : bundled copy in data/heart.csv
# ──────────────────────────────────────────────────────────────
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target",
]

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

def load_dataset() -> pd.DataFrame:
    """Load UCI Cleveland Heart Disease dataset, clean & return."""
    # Try remote first
    try:
        df = pd.read_csv(UCI_URL, header=None, names=COLUMN_NAMES, na_values="?")
        print("✅  Loaded dataset from UCI repository.")
    except Exception as e:
        print(f"⚠️  Remote fetch failed ({e}). Using bundled copy.")
        local = os.path.join(os.path.dirname(__file__), "data", "heart.csv")
        df = pd.read_csv(local)
        # Normalise columns if bundled file has headers
        if "target" not in df.columns:
            df.columns = COLUMN_NAMES

    # Clean
    df.dropna(inplace=True)
    df = df.astype(float)

    # Binarise target: 0 = no disease, 1 = disease (original: 0-4)
    df["target"] = (df["target"] > 0).astype(int)

    return df.reset_index(drop=True)


df = load_dataset()
FEATURES = COLUMN_NAMES[:-1]   # all except target
X = df[FEATURES]
y = df["target"]

FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex  (1 = Male)",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol":     "Serum Cholesterol (mg/dL)",
    "fbs":      "Fasting Blood Sugar > 120",
    "restecg":  "Resting ECG",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Oldpeak)",
    "slope":    "Slope of ST Segment",
    "ca":       "Major Vessels (Fluoroscopy)",
    "thal":     "Thalassemia",
}

# ──────────────────────────────────────────────────────────────
# 2.  DESIGN TOKENS
# ──────────────────────────────────────────────────────────────
BG, CARD, BORDER = "#0D1117", "#161B22", "#30363D"
C1  = "#58A6FF"   # blue   – primary
C2  = "#3FB950"   # green  – healthy
C3  = "#F85149"   # red    – disease
C4  = "#D2A8FF"   # purple
C5  = "#FFA657"   # orange
TEXT, SUB = "#E6EDF3", "#8B949E"

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
    margin=dict(l=48, r=28, t=48, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
)

ALGO_NAMES  = {"rf": "Random Forest", "gb": "Gradient Boosting",
               "lr": "Logistic Reg.",  "svm": "SVM"}
ALGO_COLORS = {"rf": C1, "gb": C4, "lr": C2, "svm": C5}

# ──────────────────────────────────────────────────────────────
# 3.  HELPERS
# ──────────────────────────────────────────────────────────────
def label_s(): return {
    "color": SUB, "fontSize": "10px", "letterSpacing": "1.2px",
    "textTransform": "uppercase", "marginBottom": "6px", "display": "block",
}
def dd_s(): return {
    "backgroundColor": CARD, "color": TEXT,
    "border": f"1px solid {BORDER}", "borderRadius": "6px",
}
def card_s(**kw): return {
    "background": CARD, "border": f"1px solid {BORDER}",
    "borderRadius": "8px", "padding": "22px", **kw,
}

def kpi_card(title, value, sub, color):
    return html.Div([
        html.P(title, style={"color": SUB, "fontSize": "10px", "margin": "0 0 4px",
                             "letterSpacing": "1.5px", "textTransform": "uppercase"}),
        html.H2(str(value), style={"color": color, "margin": "0",
                                   "fontSize": "30px", "fontWeight": "700"}),
        html.P(sub, style={"color": SUB, "margin": "4px 0 0", "fontSize": "11px"}),
    ], style={"background": CARD, "border": f"1px solid {BORDER}",
              "borderTop": f"3px solid {color}", "borderRadius": "8px",
              "padding": "18px 20px", "flex": "1", "minWidth": "140px"})

# ──────────────────────────────────────────────────────────────
# 4.  APP & LAYOUT
# ──────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server          # ← required by HF Spaces / gunicorn

app.title = "CardioInsight · UCI Heart Disease"

app.layout = html.Div([
    # Google Font
    html.Link(rel="preconnect", href="https://fonts.googleapis.com"),
    html.Link(rel="stylesheet",
              href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&display=swap"),

    # ── Header ──────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("❤️", style={"fontSize": "26px", "marginRight": "12px"}),
            html.H1("CardioInsight",
                    style={"margin": "0", "fontSize": "24px", "fontWeight": "700",
                           "color": TEXT, "letterSpacing": "-0.5px"}),
            html.Span(" Analytics Dashboard",
                      style={"fontSize": "13px", "color": SUB, "marginLeft": "10px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(
            f"UCI Heart Disease · Cleveland · {len(df)} patients · {len(FEATURES)} features",
            style={"color": SUB, "fontSize": "11px"},
        ),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
              "padding": "18px 32px", "borderBottom": f"1px solid {BORDER}",
              "background": CARD}),

    # ── KPI Row ─────────────────────────────────────────────
    html.Div([
        kpi_card("Patients",      len(df),                 "after cleaning",              C1),
        kpi_card("Disease",       int(y.sum()),             f"{y.mean()*100:.1f}% prevalence", C3),
        kpi_card("Healthy",       int((y==0).sum()),        f"{(1-y.mean())*100:.1f}% healthy", C2),
        kpi_card("Features",      len(FEATURES),            "clinical variables",          C4),
        kpi_card("Avg Age",       int(df["age"].mean()),    f"range {int(df.age.min())}–{int(df.age.max())}", C5),
    ], style={"display": "flex", "gap": "14px", "padding": "18px 32px", "flexWrap": "wrap"}),

    # ── Tabs ────────────────────────────────────────────────
    dcc.Tabs(id="tabs", value="eda", children=[
        dcc.Tab(label="📊  Exploration",         value="eda"),
        dcc.Tab(label="🔍  Feature Importance",  value="feat"),
        dcc.Tab(label="🤖  Model Performance",   value="model"),
        dcc.Tab(label="🩺  Predict Patient",     value="pred"),
    ], style={"padding": "0 32px"},
       colors={"border": BORDER, "primary": C1, "background": BG}),

    html.Div(id="content", style={"padding": "22px 32px"}),

], style={"background": BG, "minHeight": "100vh",
          "fontFamily": "JetBrains Mono, monospace", "color": TEXT})


# ──────────────────────────────────────────────────────────────
# 5.  TAB ROUTER
# ──────────────────────────────────────────────────────────────
@app.callback(Output("content", "children"), Input("tabs", "value"))
def route(tab):
    if tab == "eda":   return eda_layout()
    if tab == "feat":  return feat_layout()
    if tab == "model": return model_layout()
    if tab == "pred":  return pred_layout()


# ══════════════════════════════════════════════════════════════
# TAB 1 — EXPLORATION
# ══════════════════════════════════════════════════════════════
def eda_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Feature", style=label_s()),
                dcc.Dropdown(id="eda-feat",
                    options=[{"label": v, "value": k} for k, v in FEATURE_LABELS.items()],
                    value="age", clearable=False, style=dd_s()),
            ], style={"flex": "2"}),
            html.Div([
                html.Label("Chart Type", style=label_s()),
                dcc.RadioItems(id="eda-type",
                    options=[{"label": " Histogram", "value": "hist"},
                             {"label": " Box Plot",  "value": "box"},
                             {"label": " Violin",    "value": "violin"}],
                    value="hist", inline=True,
                    style={"color": TEXT, "display": "flex", "gap": "18px",
                           "alignItems": "center", "height": "38px"}),
            ], style={"flex": "2"}),
            html.Div([
                html.Label("Palette", style=label_s()),
                dcc.Dropdown(id="eda-theme",
                    options=[{"label": "Blue / Red",     "value": "br"},
                             {"label": "Teal / Orange",  "value": "to"},
                             {"label": "Purple / Green", "value": "pg"}],
                    value="br", clearable=False, style=dd_s()),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px", "alignItems": "flex-end",
                  "marginBottom": "18px"}),

        html.Div([
            dcc.Graph(id="eda-main",    style={"flex": "1.6"}),
            dcc.Graph(id="eda-corr",    style={"flex": "1"}),
        ], style={"display": "flex", "gap": "14px"}),

        html.Div([
            dcc.Graph(id="eda-scatter", style={"flex": "1"}),
            dcc.Graph(id="eda-pie",     style={"flex": "1"}),
            dcc.Graph(id="eda-sexage",  style={"flex": "1"}),
        ], style={"display": "flex", "gap": "14px", "marginTop": "14px"}),

        # Data table
        html.Div([
            html.H4("📋  Raw Dataset (first 20 rows)",
                    style={"color": SUB, "fontSize": "11px", "letterSpacing": "1px",
                           "textTransform": "uppercase", "margin": "0 0 12px"}),
            dash_table.DataTable(
                data=df.head(20).round(2).to_dict("records"),
                columns=[{"name": FEATURE_LABELS.get(c, c), "id": c} for c in df.columns],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": CARD, "color": C1,
                               "border": f"1px solid {BORDER}", "fontWeight": "700",
                               "fontSize": "11px"},
                style_cell={"backgroundColor": BG, "color": TEXT, "fontSize": "11px",
                             "border": f"1px solid {BORDER}", "padding": "6px 10px"},
                style_data_conditional=[{
                    "if": {"filter_query": "{target} = 1"},
                    "borderLeft": f"3px solid {C3}",
                }],
                page_size=20,
            ),
        ], style={**card_s(), "marginTop": "20px"}),
    ])


THEME_MAP = {"br": (C2, C3), "to": ("#00B4D8", "#FF6B35"), "pg": ("#B5179E", "#4CC9F0")}

@app.callback(
    Output("eda-main",    "figure"),
    Output("eda-corr",    "figure"),
    Output("eda-scatter", "figure"),
    Output("eda-pie",     "figure"),
    Output("eda-sexage",  "figure"),
    Input("eda-feat",  "value"),
    Input("eda-type",  "value"),
    Input("eda-theme", "value"),
)
def upd_eda(feature, ctype, theme):
    c0, c1 = THEME_MAP.get(theme, THEME_MAP["br"])
    lab = FEATURE_LABELS[feature]
    dff = df.copy()
    dff["Diagnosis"] = dff["target"].map({0: "Healthy", 1: "Disease"})
    cmap = {"Healthy": c0, "Disease": c1}

    if ctype == "hist":
        fig1 = px.histogram(dff, x=feature, color="Diagnosis",
                            color_discrete_map=cmap, barmode="overlay",
                            nbins=25, opacity=0.78, title=f"Distribution — {lab}")
    elif ctype == "box":
        fig1 = px.box(dff, x="Diagnosis", y=feature, color="Diagnosis",
                      color_discrete_map=cmap, points="all",
                      title=f"{lab} by Diagnosis")
    else:
        fig1 = px.violin(dff, x="Diagnosis", y=feature, color="Diagnosis",
                         color_discrete_map=cmap, box=True,
                         title=f"{lab} — Violin")
    fig1.update_layout(**PLOT, legend=dict(bgcolor="rgba(0,0,0,0)"))

    corr = df[FEATURES].corrwith(df["target"]).sort_values()
    fig2 = go.Figure(go.Bar(
        x=corr.values, y=[FEATURE_LABELS[f] for f in corr.index],
        orientation="h",
        marker_color=[c1 if v > 0 else c0 for v in corr.values],
        text=[f"{v:.2f}" for v in corr.values], textposition="outside",
    ))
    fig2.update_layout(**PLOT, title="Correlation with Target", height=380,
                        xaxis_title="Pearson r")

    fig3 = px.scatter(dff, x="age", y="thalach", color="Diagnosis",
                      color_discrete_map=cmap, size="chol",
                      hover_data=["cp", "ca", "oldpeak", "sex"],
                      title="Age vs Max Heart Rate  (size = Cholesterol)",
                      labels={"age": "Age (yr)", "thalach": "Max HR (bpm)"})
    fig3.update_layout(**PLOT, legend=dict(bgcolor="rgba(0,0,0,0)"))

    counts = dff["Diagnosis"].value_counts()
    fig4 = go.Figure(go.Pie(labels=counts.index, values=counts.values,
                             marker_colors=[c0, c1], hole=0.56,
                             textinfo="label+percent", pull=[0.03, 0.03]))
    fig4.update_layout(**PLOT, title="Diagnosis Split", showlegend=False)

    dff["Sex"] = dff["sex"].map({0.0: "Female", 1.0: "Male"})
    fig5 = px.histogram(dff, x="age", color="Sex",
                         color_discrete_map={"Female": C4, "Male": C5},
                         barmode="overlay", nbins=20, opacity=0.80,
                         title="Age Distribution by Sex",
                         labels={"age": "Age (yr)"})
    fig5.update_layout(**PLOT, legend=dict(bgcolor="rgba(0,0,0,0)"))

    return fig1, fig2, fig3, fig4, fig5


# ══════════════════════════════════════════════════════════════
# TAB 2 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
def feat_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Algorithm", style=label_s()),
                dcc.Dropdown(id="fi-algo",
                    options=[{"label": "Random Forest",      "value": "rf"},
                             {"label": "Gradient Boosting",  "value": "gb"},
                             {"label": "Logistic Regression","value": "lr"}],
                    value="rf", clearable=False, style=dd_s()),
            ], style={"flex": "1"}),
            html.Div([
                html.Label(f"Top N Features  (3 – {len(FEATURES)})", style=label_s()),
                dcc.Slider(id="fi-n", min=3, max=len(FEATURES), step=1, value=8,
                           marks={i: str(i) for i in range(3, len(FEATURES)+1)},
                           tooltip={"placement": "bottom"}),
            ], style={"flex": "2.5"}),
        ], style={"display": "flex", "gap": "24px", "alignItems": "flex-end",
                  "marginBottom": "18px"}),

        html.Div([
            dcc.Graph(id="fi-bar",     style={"flex": "1"}),
            dcc.Graph(id="fi-heatmap", style={"flex": "1.3"}),
        ], style={"display": "flex", "gap": "14px"}),

        dcc.Graph(id="fi-parallel", style={"marginTop": "14px"}),
    ])


@app.callback(
    Output("fi-bar",      "figure"),
    Output("fi-heatmap",  "figure"),
    Output("fi-parallel", "figure"),
    Input("fi-algo", "value"),
    Input("fi-n",    "value"),
)
def upd_feat(algo, n):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr)

    if algo == "rf":
        mdl = RandomForestClassifier(n_estimators=200, random_state=42)
        mdl.fit(Xtr, ytr); imp = mdl.feature_importances_
    elif algo == "gb":
        mdl = GradientBoostingClassifier(n_estimators=200, random_state=42)
        mdl.fit(Xtr, ytr); imp = mdl.feature_importances_
    else:
        mdl = LogisticRegression(max_iter=1000, random_state=42)
        mdl.fit(Xtr_sc, ytr)
        imp = np.abs(mdl.coef_[0]); imp /= imp.sum()

    fi   = pd.Series(imp, index=FEATURES).sort_values(ascending=False)
    top  = fi.head(n)

    fig1 = go.Figure(go.Bar(
        x=top.values, y=[FEATURE_LABELS[f] for f in top.index],
        orientation="h",
        marker=dict(color=top.values, colorscale="Blues", showscale=True,
                    colorbar=dict(thickness=10, tickfont=dict(color=TEXT))),
        text=[f"{v:.3f}" for v in top.values], textposition="outside",
    ))
    fig1.update_layout(**PLOT,
        title=f"Top {n} Features — {ALGO_NAMES[algo]}",
        xaxis_title="Importance Score", height=380)

    top_cols = list(top.index) + ["target"]
    cm = df[top_cols].corr()
    labs = [FEATURE_LABELS.get(c, c) for c in cm.columns]
    fig2 = go.Figure(go.Heatmap(
        z=cm.values, x=labs, y=labs,
        colorscale="RdBu_r", zmid=0,
        text=np.round(cm.values, 2), texttemplate="%{text}",
        colorbar=dict(thickness=10, tickfont=dict(color=TEXT)),
    ))
    fig2.update_layout(**PLOT, title="Correlation Matrix (Top Features)", height=380)

    t5   = list(fi.head(5).index)
    dff  = df[t5 + ["target"]].copy()
    norm = dff.copy()
    for c in t5:
        mn, mx = dff[c].min(), dff[c].max()
        norm[c] = (dff[c] - mn) / (mx - mn + 1e-9)
    dims = [dict(label=FEATURE_LABELS[f], values=norm[f]) for f in t5]
    dims.append(dict(label="Diagnosis", values=dff["target"],
                     tickvals=[0,1], ticktext=["Healthy","Disease"]))
    fig3 = go.Figure(go.Parcoords(
        line=dict(color=dff["target"],
                  colorscale=[[0, C2], [1, C3]], cmin=0, cmax=1),
        dimensions=dims,
    ))
    fig3.update_layout(**PLOT, title="Parallel Coordinates — Top 5 Features", height=300)

    return fig1, fig2, fig3


# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
def model_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Models to Compare", style=label_s()),
                dcc.Checklist(id="ml-models",
                    options=[{"label": f"  {v}", "value": k}
                             for k, v in ALGO_NAMES.items()],
                    value=["rf", "gb"],
                    style={"color": TEXT, "display": "flex", "gap": "20px",
                           "flexWrap": "wrap", "alignItems": "center",
                           "height": "38px"}),
            ], style={"flex": "2"}),
            html.Div([
                html.Label("Test Split %", style=label_s()),
                dcc.Slider(id="ml-split", min=10, max=40, step=5, value=20,
                           marks={i: f"{i}%" for i in range(10, 45, 5)},
                           tooltip={"placement": "bottom"}),
            ], style={"flex": "1.5"}),
        ], style={"display": "flex", "gap": "24px", "alignItems": "flex-end",
                  "marginBottom": "18px"}),

        html.Div(id="ml-kpis",
                 style={"display": "flex", "gap": "12px",
                        "flexWrap": "wrap", "marginBottom": "18px"}),

        html.Div([
            dcc.Graph(id="ml-roc", style={"flex": "1"}),
            dcc.Graph(id="ml-cm",  style={"flex": "1"}),
            dcc.Graph(id="ml-cv",  style={"flex": "1"}),
        ], style={"display": "flex", "gap": "14px"}),
    ])


@app.callback(
    Output("ml-kpis", "children"),
    Output("ml-roc",  "figure"),
    Output("ml-cm",   "figure"),
    Output("ml-cv",   "figure"),
    Input("ml-models","value"),
    Input("ml-split", "value"),
)
def upd_model(selected, split_pct):
    empty = go.Figure(); empty.update_layout(**PLOT)
    if not selected:
        return [], empty, empty, empty

    ts = split_pct / 100
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=42)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr); Xte_sc = sc.transform(Xte)

    def build(k):
        return {"rf":  RandomForestClassifier(n_estimators=200, random_state=42),
                "gb":  GradientBoostingClassifier(n_estimators=200, random_state=42),
                "lr":  LogisticRegression(max_iter=1000, random_state=42),
                "svm": SVC(probability=True, random_state=42)}[k]

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color=BORDER),
                                  showlegend=False))
    kpis, primary_cm, cv_traces = [], None, []

    for k in selected:
        mdl = build(k)
        use_sc = k in ("lr", "svm")
        _Xtr, _Xte = (Xtr_sc, Xte_sc) if use_sc else (Xtr, Xte)
        mdl.fit(_Xtr, ytr)
        yp  = mdl.predict(_Xte)
        ypr = mdl.predict_proba(_Xte)[:, 1]
        acc = accuracy_score(yte, yp)
        rep = classification_report(yte, yp, output_dict=True)
        fpr, tpr, _ = roc_curve(yte, ypr)
        ra  = auc(fpr, tpr)
        cvs = cross_val_score(mdl, _Xtr, ytr, cv=5)

        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{ALGO_NAMES[k]} (AUC={ra:.2f})",
            line=dict(color=ALGO_COLORS[k], width=2.5)))

        kpis.append(html.Div([
            html.P(ALGO_NAMES[k],
                   style={"color": ALGO_COLORS[k], "margin": "0 0 4px",
                          "fontSize": "10px", "letterSpacing": "1px",
                          "textTransform": "uppercase"}),
            html.H3(f"{acc:.1%}",
                    style={"margin": "0", "fontSize": "26px",
                           "color": TEXT, "fontWeight": "700"}),
            html.P(f"AUC {ra:.3f}  ·  F1 {rep['weighted avg']['f1-score']:.3f}",
                   style={"margin": "4px 0 0", "fontSize": "11px", "color": SUB}),
        ], style={**card_s(), "borderTop": f"3px solid {ALGO_COLORS[k]}",
                  "minWidth": "155px"}))

        if primary_cm is None:
            primary_cm = (confusion_matrix(yte, yp), k)
        cv_traces.append((k, cvs))

    fig_roc.update_layout(**PLOT, title="ROC Curves",
                           xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate",
                           legend=dict(bgcolor="rgba(0,0,0,0)", x=0.5, y=0.05))

    cm, cm_k = primary_cm
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Pred Healthy", "Pred Disease"],
        y=["Actual Healthy", "Actual Disease"],
        colorscale=[[0, BG], [1, ALGO_COLORS[cm_k]]],
        text=cm, texttemplate="<b>%{text}</b>",
        textfont=dict(size=22), showscale=False,
    ))
    fig_cm.update_layout(**PLOT,
        title=f"Confusion Matrix — {ALGO_NAMES[cm_k]}")

    fig_cv = go.Figure()
    for k, cvs in cv_traces:
        fig_cv.add_trace(go.Box(y=cvs, name=ALGO_NAMES[k],
                                 marker_color=ALGO_COLORS[k], boxmean=True))
    fig_cv.update_layout(**PLOT, title="5-Fold CV Accuracy",
                          yaxis_title="Accuracy", showlegend=False)

    return kpis, fig_roc, fig_cm, fig_cv


# ══════════════════════════════════════════════════════════════
# TAB 4 — PATIENT PREDICTION
# ══════════════════════════════════════════════════════════════
PRED_SLIDERS = [
    ("age",      "Age (years)",                  29, 77,   1,    54, None),
    ("sex",      "Sex",                            0,  1,   1,     1, {0:"Female", 1:"Male"}),
    ("cp",       "Chest Pain Type (0–3)",          0,  3,   1,     0, {0:"Asympt", 1:"Atypical", 2:"Non-ang", 3:"Typical"}),
    ("trestbps", "Resting BP (mmHg)",             94,200,   2,   130, None),
    ("chol",     "Cholesterol (mg/dL)",           126,564,   5,   246, None),
    ("fbs",      "Fasting Sugar > 120 mg/dL",     0,  1,   1,     0, {0:"No", 1:"Yes"}),
    ("restecg",  "Resting ECG (0–2)",              0,  2,   1,     1, {0:"Normal", 1:"ST-T abnorm.", 2:"LV Hypertrophy"}),
    ("thalach",  "Max Heart Rate (bpm)",          71,202,   2,   150, None),
    ("exang",    "Exercise Angina",                0,  1,   1,     0, {0:"No", 1:"Yes"}),
    ("oldpeak",  "ST Depression (0–6)",            0,  6, 0.1,   1.0, None),
    ("slope",    "ST Slope (0–2)",                 0,  2,   1,     1, {0:"Downsloping", 1:"Flat", 2:"Upsloping"}),
    ("ca",       "Major Vessels (0–3)",            0,  3,   1,     0, {0:"0", 1:"1", 2:"2", 3:"3"}),
    ("thal",     "Thalassemia",                    0,  3,   1,     2, {0:"Normal", 1:"Fixed defect", 2:"Reversible", 3:"Other"}),
]
PRED_IDS = [r[0] for r in PRED_SLIDERS]


def pred_layout():
    def row(fid, lbl, mn, mx, step, val, marks):
        m = marks or {mn: str(mn), mx: str(mx)}
        return html.Div([
            html.Label(lbl, style={"color": SUB, "fontSize": "11px",
                                   "width": "210px", "flexShrink": "0"}),
            dcc.Slider(id=f"p-{fid}", min=mn, max=mx, step=step, value=val,
                       marks=m, tooltip={"placement":"top","always_visible":False},
                       style={"flex": "1"}),
            html.Span(id=f"p-{fid}-v", children=str(val),
                      style={"color": C1, "fontSize": "12px", "fontWeight": "700",
                             "width": "48px", "textAlign": "right", "flexShrink": "0"}),
        ], style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"9px"})

    return html.Div([
        html.Div([
            # Input panel
            html.Div([
                html.H3("🩺  Patient Parameters",
                        style={"color": C1, "margin": "0 0 18px", "fontSize": "13px",
                               "letterSpacing": "1px", "textTransform": "uppercase",
                               "borderBottom": f"1px solid {BORDER}",
                               "paddingBottom": "10px"}),
                *[row(*r) for r in PRED_SLIDERS],
                html.Div([
                    html.Label("Prediction Model", style=label_s()),
                    dcc.Dropdown(id="p-algo",
                        options=[{"label":"Random Forest",     "value":"rf"},
                                 {"label":"Gradient Boosting", "value":"gb"},
                                 {"label":"Logistic Regression","value":"lr"}],
                        value="rf", clearable=False, style=dd_s()),
                ], style={"marginTop": "18px"}),
            ], style={**card_s(flex="1")}),

            # Output panel
            html.Div([
                html.H3("📈  Result",
                        style={"color": C1, "margin": "0 0 18px", "fontSize": "13px",
                               "letterSpacing": "1px", "textTransform": "uppercase",
                               "borderBottom": f"1px solid {BORDER}",
                               "paddingBottom": "10px"}),
                html.Div(id="p-banner"),
                dcc.Graph(id="p-gauge",   style={"height": "280px"}),
                dcc.Graph(id="p-contrib", style={"height": "380px"}),
            ], style={**card_s(flex="1")}),
        ], style={"display": "flex", "gap": "20px"}),
    ])


# Live slider display
for _fid, *_ in PRED_SLIDERS:
    @app.callback(Output(f"p-{_fid}-v", "children"), Input(f"p-{_fid}", "value"))
    def _disp(v): return str(v)


@app.callback(
    Output("p-banner",  "children"),
    Output("p-gauge",   "figure"),
    Output("p-contrib", "figure"),
    [Input(f"p-{fid}", "value") for fid in PRED_IDS],
    Input("p-algo", "value"),
)
def upd_pred(*args):
    vals = {fid: float(v) for fid, v in zip(PRED_IDS, args[:len(PRED_IDS)])}
    algo = args[-1]

    pv    = np.array([[vals[f] for f in FEATURES]])
    sc    = StandardScaler().fit(X)
    pv_sc = sc.transform(pv)

    def build(k):
        return {"rf":  RandomForestClassifier(n_estimators=200, random_state=42),
                "gb":  GradientBoostingClassifier(n_estimators=200, random_state=42),
                "lr":  LogisticRegression(max_iter=1000, random_state=42)}[k]

    scaled = algo == "lr"
    mdl = build(algo)
    mdl.fit(sc.transform(X) if scaled else X, y)
    prob = mdl.predict_proba(pv_sc if scaled else pv)[0][1]
    pred = int(prob >= 0.5)

    color = C3 if pred else C2
    icon  = "⚠️" if pred else "✅"
    label = "HIGH RISK — Disease Likely" if pred else "LOW RISK — No Disease Detected"

    banner = html.Div([
        html.Div(f"{icon}  {label}", style={
            "color": color, "fontSize": "15px", "fontWeight": "700",
            "padding": "14px 18px", "borderRadius": "8px",
            "border": f"2px solid {color}", "textAlign": "center",
            "background": f"{color}15", "marginBottom": "10px",
        }),
        html.P(f"Estimated probability of disease: {prob:.1%}",
               style={"color": SUB, "fontSize": "12px",
                      "textAlign": "center", "margin": "0"}),
    ])

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"color": color, "size": 38}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": SUB,
                     "tickfont": dict(color=SUB)},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": CARD, "bordercolor": BORDER,
            "steps": [
                {"range": [0,  35], "color": f"{C2}22"},
                {"range": [35, 65], "color": f"{C5}22"},
                {"range": [65, 100],"color": f"{C3}22"},
            ],
            "threshold": {"line": {"color": TEXT, "width": 3},
                          "thickness": 0.85, "value": 50},
        },
        title={"text": "Risk Score", "font": {"color": SUB, "size": 13}},
    ))
    fig_g.update_layout(**PLOT, height=260)

    # Perturbation-based feature contribution
    X_means = X.mean().to_dict()
    contrib = {}
    for feat in PRED_IDS:
        alt    = pv.copy()
        alt[0, FEATURES.index(feat)] = X_means[feat]
        alt_in = sc.transform(alt) if scaled else alt
        contrib[feat] = prob - mdl.predict_proba(alt_in)[0][1]

    cs   = pd.Series(contrib).sort_values()
    bcol = [C3 if v > 0 else C2 for v in cs.values]
    fig_c = go.Figure(go.Bar(
        x=cs.values, y=[FEATURE_LABELS[f] for f in cs.index],
        orientation="h", marker_color=bcol,
        text=[f"{v:+.3f}" for v in cs.values], textposition="outside",
    ))
    fig_c.update_layout(**PLOT,
        title="Feature Contribution to Risk",
        xaxis_title="Δ Probability", height=360)

    return banner, fig_g, fig_c


# ──────────────────────────────────────────────────────────────
# 6.  ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n  ❤️  CardioInsight →  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
