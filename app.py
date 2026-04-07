"""
╔══════════════════════════════════════════════════════════════╗
║     CardioInsight — Heart Disease Analytics Dashboard        ║
║     Dataset : UCI Heart Disease (Cleveland) — Real Data      ║
║     Deploy  : Hugging Face Spaces (Dash SDK)                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score,
)

# ──────────────────────────────────────────────────────────────
# 1.  LOAD REAL UCI HEART DISEASE DATASET & FEATURE ENGINEERING
#     Source : bundled copy in data/Heart_disease.csv
# ──────────────────────────────────────────────────────────────
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target",
]

def load_dataset() -> pd.DataFrame:
    """Load bundled heart disease dataset from the local data folder."""
    local = os.path.join(os.path.dirname(__file__), "data", "Heart_disease.csv")
    if not os.path.exists(local):
        raise FileNotFoundError(
            f"Dataset not found at '{local}'. Expected data/Heart_disease.csv"
        )

    df = pd.read_csv(local)
    print(f"[OK] Loaded dataset from local file: {local}")

    # Normalise columns if the local file does not include target header.
    if "target" not in df.columns:
        df.columns = COLUMN_NAMES

    # Clean
    df.dropna(inplace=True)
    df = df.astype(float)

    # Binarise target: 0 = no disease, 1 = disease (original: 0-4)
    df["target"] = (df["target"] > 0).astype(int)

    # ─────────────────────────────────────────────────────────────
    # FEATURE ENGINEERING (from notebook)
    # ─────────────────────────────────────────────────────────────
    # Hypertension flag (clinical threshold ≥ 140 mmHg)
    df['hypertension'] = (df['trestbps'] >= 140).astype(int)
    
    # High cholesterol flag (AHA threshold ≥ 240 mg/dL)
    df['high_chol'] = (df['chol'] >= 240).astype(int)
    
    # Heart rate reserve proxy: thalach normalized to age-predicted max (220 - age)
    df['hr_reserve_pct'] = df['thalach'] / (220 - df['age'])

    print("[OK] Feature engineering applied: hypertension, high_chol, hr_reserve_pct")
    
    return df.reset_index(drop=True)


df = load_dataset()

# ─────────────────────────────────────────────────────────────
# BUILD PREPROCESSING PIPELINE (from notebook)
# ─────────────────────────────────────────────────────────────
# Feature groups based on scaling strategy
num_standard = ['age', 'thalach', 'hr_reserve_pct']        # roughly normal
num_robust   = ['trestbps', 'chol', 'oldpeak']             # has outliers
binary_cols  = ['sex', 'fbs', 'exang', 'hypertension', 'high_chol']
ordinal_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']

# Sub-pipelines for each feature group
num_standard_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

num_robust_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler()),
])

binary_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
])

ordinal_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
])

# Column Transformer combining all sub-pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num_std', num_standard_pipe, num_standard),
        ('num_robust', num_robust_pipe, num_robust),
        ('binary', binary_pipe, binary_cols),
        ('ordinal', ordinal_pipe, ordinal_cols),
    ],
    remainder='drop',
    verbose_feature_names_out=True,
)

# Prepare feature columns in correct order
feature_cols = num_standard + num_robust + binary_cols + ordinal_cols
X = df[feature_cols]
y = df["target"]

# Fit the preprocessor on the full dataset once (for static app)
X_preprocessed = preprocessor.fit_transform(X)

print(f"[OK] Preprocessing pipeline built and fitted. Feature matrix shape: {X_preprocessed.shape}")

FEATURES = feature_cols  # Updated to include engineered features

# Feature type categorization for different chart modes
NUMERICAL_FEATURES = num_standard + num_robust
CATEGORICAL_FEATURES = binary_cols + ordinal_cols

# Map each feature to its type
FEATURE_TYPES = {f: "numerical" for f in NUMERICAL_FEATURES}
FEATURE_TYPES.update({f: "categorical" for f in CATEGORICAL_FEATURES})

FEATURE_LABELS = {
    "age":          "Age (years)",
    "sex":          "Sex (1 = Male)",
    "cp":           "Chest Pain Type",
    "trestbps":     "Resting Blood Pressure",
    "chol":         "Serum Cholesterol (mg/dL)",
    "fbs":          "Fasting Blood Sugar > 120",
    "restecg":      "Resting ECG",
    "thalach":      "Max Heart Rate Achieved",
    "exang":        "Exercise-Induced Angina",
    "oldpeak":      "ST Depression (Oldpeak)",
    "slope":        "Slope of ST Segment",
    "ca":           "Major Vessels (Fluoroscopy)",
    "thal":         "Thalassemia",
    "hypertension": "Hypertension (BP ≥ 140)",
    "high_chol":    "High Cholesterol (≥ 240)",
    "hr_reserve_pct": "HR Reserve %",
}

# Category labels for categorical features (for nice pie chart labels)
FEATURE_CATEGORY_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"},
    "fbs": {0: "Normal (<= 120)", 1: "Elevated (> 120)"},
    "restecg": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Downsloping", 1: "Flat", 2: "Upsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal": {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"},
    "hypertension": {0: "Normal (< 140)", 1: "High (>= 140)"},
    "high_chol": {0: "Normal (< 240)", 1: "High (>= 240)"},
    "target": {0: "Healthy", 1: "Disease"},
}

# Feature definitions for the Exploration tab
FEATURE_DEFINITIONS = {
    "age": "Patient's age in years. Older patients generally have higher cardiovascular risk.",
    "sex": "Biological sex (0=Female, 1=Male). Males typically have higher heart disease prevalence in this dataset.",
    "cp": "Type of chest pain: 0=Typical angina, 1=Atypical angina, 2=Non-anginal pain, 3=Asymptomatic. Describes the nature of chest discomfort experienced.",
    "trestbps": "Systolic blood pressure at rest (mmHg). Normal is <120; elevated ≥140 indicates hypertension.",
    "chol": "Serum cholesterol level (mg/dL). Higher levels (≥240) indicate increased cardiac risk.",
    "fbs": "Fasting blood sugar level >120 mg/dL (1=Yes, 0=No). Indicator of glucose metabolism and diabetes risk.",
    "restecg": "Resting electrocardiography results: 0=Normal, 1=ST-T abnormality, 2=Left ventricular hypertrophy. Shows electrical heart activity at rest.",
    "thalach": "Maximum heart rate achieved during exercise (bpm). Lower values may indicate reduced cardiac fitness or disease.",
    "exang": "Exercise-induced angina (1=Yes, 0=No). Presence of chest pain during physical exertion; strong disease predictor.",
    "oldpeak": "ST depression induced by exercise relative to rest (mm). Measures electrical changes under cardiac stress; key diagnostic indicator.",
    "slope": "Slope of peak exercise ST segment: 0=Downsloping, 1=Flat, 2=Upsloping. Describes abnormal changes in heart's electrical activity.",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy. Coronary artery calcification; more vessels = higher disease likelihood.",
    "thal": "Thalassemia type: 1=Normal, 2=Fixed Defect, 3=Reversible Defect. Blood condition affecting oxygen transport.",
    "hypertension": "Binary flag for resting BP ≥140 mmHg. Clinical indicator of hypertension status.",
    "high_chol": "Binary flag for cholesterol ≥240 mg/dL. Identifies patients with elevated cholesterol levels.",
    "hr_reserve_pct": "Heart rate reserve as percentage of age-predicted maximum (thalach / (220-age)). Normalized measure of exercise capacity.",
}

# ──────────────────────────────────────────────────────────────
# 2.  DESIGN TOKENS
# ──────────────────────────────────────────────────────────────
BG, CARD, BORDER = "#0D1117", "#161B22", "#30363D"
GRAPH_BG = "#11161F"  # modern black graph panel
C1  = "#58A6FF"   # blue   – primary
C2  = "#3FB950"   # green  – healthy
C3  = "#F85149"   # red    – disease
C4  = "#D2A8FF"   # purple
C5  = "#FFA657"   # orange
TEXT, SUB = "#E6EDF3", "#8B949E"

PLOT = dict(
    paper_bgcolor=GRAPH_BG,
    plot_bgcolor =GRAPH_BG,
    font=dict(family="JetBrains Mono, monospace", color=TEXT, size=12),
    margin=dict(l=48, r=28, t=48, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
)

ALGO_NAMES  = {"rf": "Random Forest", "gb": "Gradient Boosting",
               "lr": "Logistic Reg.",  "svm": "SVM"}

THEME_MAP = {
    "br": ("#58A6FF", "#F85149"),  # Blue / Red
    "to": ("#14B8A6", "#F59E0B"),  # Teal / Orange
    "pg": ("#A78BFA", "#22C55E"),  # Purple / Green
}

THEME_MODEL_COLORS = {
    "br": {"rf": "#58A6FF", "gb": "#F85149", "lr": "#3FB950", "svm": "#FFA657"},
    "to": {"rf": "#14B8A6", "gb": "#F59E0B", "lr": "#38BDF8", "svm": "#F97316"},
    "pg": {"rf": "#A78BFA", "gb": "#22C55E", "lr": "#C084FC", "svm": "#16A34A"},
}

# ──────────────────────────────────────────────────────────────
# 3.  HELPERS
# ──────────────────────────────────────────────────────────────
def label_s(): return {
    "color": SUB, "fontSize": "12px", "letterSpacing": "1.2px",
    "textTransform": "uppercase", "marginBottom": "8px", "display": "block",
}
def card_s(**kw): return {
    "background": CARD, "border": f"1px solid {BORDER}",
    "borderRadius": "8px", "padding": "22px", **kw,
}

def hex_to_rgba(hex_color, alpha):
    """Convert a hex color to rgba(r,g,b,a) string for translucent Plotly fills."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def blank_fig(height=None):
    """Dark placeholder figure to avoid white flash before callbacks update charts."""
    fig = go.Figure()
    fig.update_layout(**PLOT, showlegend=False)
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    if height is not None:
        fig.update_layout(height=height)
    return fig

def kpi_card(title, value, sub, color):
    return html.Div([
        html.P(title, style={"color": SUB, "fontSize": "12px", "margin": "0 0 6px",
                             "letterSpacing": "1.5px", "textTransform": "uppercase"}),
        html.H2(str(value), style={"color": color, "margin": "0",
                                   "fontSize": "36px", "fontWeight": "700"}),
        html.P(sub, style={"color": SUB, "margin": "6px 0 0", "fontSize": "13px"}),
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
                    style={"margin": "0", "fontSize": "28px", "fontWeight": "700",
                           "color": TEXT, "letterSpacing": "-0.5px"}),
            html.Span(" Analytics Dashboard",
                      style={"fontSize": "15px", "color": SUB, "marginLeft": "10px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Div(
                f"UCI Heart Disease · Cleveland · {len(df)} patients · {len(FEATURES)} features",
                style={"color": SUB, "fontSize": "13px", "marginBottom": "8px"},
            ),
            html.Div([
                html.Span("Palette", style={"color": SUB, "fontSize": "12px", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="global-theme",
                    options=[
                        {"label": "Blue / Red", "value": "br"},
                        {"label": "Teal / Orange", "value": "to"},
                        {"label": "Purple / Green", "value": "pg"},
                    ],
                    value="br", clearable=False, className="ci-dropdown",
                    style={"width": "210px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
        ], style={"display": "flex", "flexDirection": "column", "alignItems": "flex-end"}),
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
    dcc.Tabs(
        id="tabs",
        value="eda",
        parent_className="ci-tabs-parent",
        className="ci-tabs",
        children=[
            dcc.Tab(label="🔬  Exploration", value="eda", className="ci-tab", selected_className="ci-tab-selected"),
            dcc.Tab(label="🏆  Feature Importance", value="feat", className="ci-tab", selected_className="ci-tab-selected"),
            dcc.Tab(label="🤖  Model Performance", value="model", className="ci-tab", selected_className="ci-tab-selected"),
            dcc.Tab(label="🩺  Predict Patient", value="pred", className="ci-tab", selected_className="ci-tab-selected"),
        ],
    ),

    html.Div(id="content", style={"padding": "22px 32px"}),

], className="app-root", style={"background": BG, "minHeight": "100vh",
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
                    value="age", clearable=False, className="ci-dropdown"),
            ], style={"flex": "2"}),
            
            html.Div([
                html.Label("Chart Type", style=label_s()),
                dcc.RadioItems(id="eda-type",
                    options=[],  # Will be populated by callback
                    value="hist", inline=True, className="ci-radio"),
            ], style={"flex": "3"}),
        ], style={"display": "flex", "gap": "20px", "alignItems": "flex-end",
                  "marginBottom": "18px"}),

        # ── Feature Definition Card ─────────────────────────────
        html.Div([
            html.Div([
                html.H4(id="eda-def-title", 
                   style={"color": C1, "margin": "0 0 16px", "fontSize": "20px",
                       "fontWeight": "800", "letterSpacing": "0.4px"}),
                html.P(id="eda-def-text",
                  style={"color": TEXT, "margin": "0", "fontSize": "16px",
                      "lineHeight": "1.95", "opacity": "1", "fontWeight": "500"}),
            ], style=card_s()),
         ], style={"marginBottom": "20px", "paddingLeft": "6px", "borderLeft": f"4px solid {C1}"}),

        html.Div([
            dcc.Graph(id="eda-main", figure=blank_fig(), style={"flex": "1.6"}),
            dcc.Graph(id="eda-corr", figure=blank_fig(), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "14px"}),

        html.Div([
            dcc.Graph(id="eda-sexage", figure=blank_fig(), style={"flex": "1"}),
            dcc.Graph(id="eda-pie", figure=blank_fig(), style={"flex": "1"}),
            dcc.Graph(id="eda-age-sex", figure=blank_fig(), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "14px", "marginTop": "14px"}),
    ])

@app.callback(
    Output("eda-def-title", "children"),
    Output("eda-def-text",  "children"),
    Input("eda-feat", "value"),
)
def upd_eda_def(feature):
    """Update feature definition when feature selection changes."""
    title = f"📌  {FEATURE_LABELS[feature]}"
    definition = FEATURE_DEFINITIONS.get(feature, "No definition available for this feature.")
    return title, definition


@app.callback(
    Output("eda-type", "options"),
    Output("eda-type", "value"),
    Input("eda-feat", "value"),
)
def upd_chart_options(feature):
    """Update chart type options based on feature type (numerical vs categorical)."""
    feat_type = FEATURE_TYPES.get(feature, "numerical")
    
    if feat_type == "numerical":
        # Numerical features: Histogram, Box Plot, KDE
        options = [
            {"label": " Histogram", "value": "hist"},
            {"label": " Box Plot",  "value": "box"},
            {"label": " KDE",       "value": "kde"}
        ]
        default = "hist"
    else:
        # Categorical features: Bar Chart, Stacked, Heatmap, Pie
        options = [
            {"label": " Bar Chart",  "value": "bar"},
            {"label": " Stacked",    "value": "stacked"},
            {"label": " Heatmap",    "value": "heatmap"},
            {"label": " Pie",        "value": "pie"}
        ]
        default = "bar"
    
    return options, default


@app.callback(
    Output("eda-main",    "figure"),
    Output("eda-corr",    "figure"),
    Output("eda-pie",     "figure"),
    Output("eda-sexage",  "figure"),
    Output("eda-age-sex", "figure"),
    Input("eda-feat",  "value"),
    Input("eda-type",  "value"),
    Input("global-theme", "value"),
)
def upd_eda(feature, ctype, theme):
    c0, c1 = THEME_MAP.get(theme, THEME_MAP["br"])
    lab = FEATURE_LABELS[feature]
    dff = df.copy()
    dff["Diagnosis"] = dff["target"].map({0: "Healthy", 1: "Disease"})
    cmap = {"Healthy": c0, "Disease": c1}
    
    feat_type = FEATURE_TYPES.get(feature, "numerical")

    # Chart generation based on feature type
    if feat_type == "numerical":
        # Numerical features: Histogram, Box Plot, KDE
        if ctype == "hist":
            fig1 = px.histogram(dff, x=feature, color="Diagnosis",
                                color_discrete_map=cmap, barmode="overlay",
                                nbins=25, opacity=0.78, title=f"Distribution — {lab}")
        elif ctype == "box":
            fig1 = px.box(dff, x="Diagnosis", y=feature, color="Diagnosis",
                          color_discrete_map=cmap, points="all",
                          title=f"{lab} by Diagnosis")
        else:  # kde
            # KDE plot using plotly
            import scipy.stats as stats
            x_range = np.linspace(dff[feature].min(), dff[feature].max(), 200)
            fig1 = go.Figure()
            for diag, color in cmap.items():
                data = dff[dff["Diagnosis"] == diag][feature]
                if len(data) > 1:
                    kde = stats.gaussian_kde(data)
                    fig1.add_trace(go.Scatter(
                        x=x_range, y=kde(x_range),
                        mode='lines', name=diag, line=dict(color=color, width=2.5),
                        fill='tozeroy', opacity=0.6
                    ))
            fig1.update_layout(**PLOT, title=f"{lab} — Kernel Density Estimate",
                              xaxis_title=lab, yaxis_title="Density")
    else:
        # Categorical features: Bar Chart, Stacked, Heatmap, Pie
        if ctype == "bar":
            # Grouped bar chart with counts and percentages
            freq_data = dff.groupby([feature, "Diagnosis"]).size().reset_index(name="Count")
            # Calculate percentages for each feature category
            freq_data["Total"] = freq_data.groupby(feature)["Count"].transform("sum")
            freq_data["Pct"] = (freq_data["Count"] / freq_data["Total"] * 100).round(1)
            fig1 = px.bar(freq_data, x=feature, y="Count", color="Diagnosis",
                         color_discrete_map=cmap, barmode="group",
                         title=f"{lab} — Frequency by Diagnosis",
                         labels={feature: lab},
                         text="Pct",
                         custom_data=["Pct", "Count"])
            fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              hovertemplate="<b>%{x}</b><br>%{fullData.name}<br>Count: %{customdata[1]}<br>Percentage: %{customdata[0]:.1f}%<extra></extra>")
        elif ctype == "stacked":
            # Stacked bar chart with percentages visible on bars
            freq_data = dff.groupby([feature, "Diagnosis"]).size().reset_index(name="Count")
            freq_data["Total"] = freq_data.groupby(feature)["Count"].transform("sum")
            freq_data["Pct"] = (freq_data["Count"] / freq_data["Total"] * 100).round(1)
            fig1 = px.bar(freq_data, x=feature, y="Count", color="Diagnosis",
                         color_discrete_map=cmap, barmode="stack",
                         title=f"{lab} — Stacked Distribution",
                         labels={feature: lab},
                         text="Pct",
                         custom_data=["Pct", "Count"])
            fig1.update_traces(texttemplate="%{text:.1f}%", textposition="inside",
                              hovertemplate="<b>%{x}</b><br>%{fullData.name}<br>Count: %{customdata[1]}<br>Percentage: %{customdata[0]:.1f}%<extra></extra>")
        elif ctype == "heatmap":
            # Contingency table heatmap: categories vs diagnosis
            contingency = pd.crosstab(dff[feature], dff["Diagnosis"])
            fig1 = go.Figure(data=go.Heatmap(
                z=contingency.values,
                x=contingency.columns,
                y=contingency.index,
                colorscale=[[0, c0], [1, c1]],
                text=contingency.values,
                texttemplate='%{text}',
                colorbar=dict(title="Count")
            ))
            fig1.update_layout(**PLOT, title=f"{lab} — Distribution Heatmap",
                              xaxis_title="Diagnosis", yaxis_title=lab)
        else:  # pie
            # Pie chart: show distribution of selected feature
            if feat_type == "numerical":
                # For numerical features, create bins and show distribution
                feature_dist = pd.cut(dff[feature], bins=5).value_counts().sort_index()
                labels = [f"{interval.left:.1f}-{interval.right:.1f}" for interval in feature_dist.index]
            else:
                # For categorical features, show direct value counts with meaningful labels
                feature_dist = dff[feature].value_counts()
                # Get category labels if they exist, otherwise use raw values
                if feature in FEATURE_CATEGORY_LABELS:
                    label_map = FEATURE_CATEGORY_LABELS[feature]
                    labels = [label_map.get(int(val) if isinstance(val, float) and val.is_integer() else val, str(val)) 
                             for val in feature_dist.index]
                else:
                    labels = [str(int(val)) if isinstance(val, float) and val.is_integer() else str(val) for val in feature_dist.index]
            
            n_slices = len(feature_dist)
            steps = [i / max(1, n_slices - 1) for i in range(n_slices)]
            pie_colors = px.colors.sample_colorscale([c0, c1], steps)

            fig1 = px.pie(values=feature_dist.values,
                         names=labels,
                         hole=0.45,
                         title=f"{lab} — Distribution")
            fig1.update_traces(
                marker=dict(colors=pie_colors),
                textposition="inside",
                textinfo="label+percent+value"
            )

    
    fig1.update_layout(**PLOT, legend=dict(bgcolor="rgba(0,0,0,0)"))

    corr = df[FEATURES].corrwith(df["target"]).sort_values()
    corr_vals = corr.values.astype(float)
    corr_pad = max(0.03, (float(corr_vals.max()) - float(corr_vals.min())) * 0.14)
    fig2 = go.Figure(go.Bar(
        x=corr.values, y=[FEATURE_LABELS[f] for f in corr.index],
        orientation="h",
        marker_color=[c1 if v > 0 else c0 for v in corr.values],
        text=[f"{v:+.2f}" for v in corr.values],
        textposition="outside",
        textfont=dict(color=TEXT, size=16, family="JetBrains Mono, monospace"),
        cliponaxis=False,
    ))
    fig2.update_layout(**PLOT, title="Correlation with Target", height=380,
                        xaxis_title="Pearson r")
    fig2.update_layout(margin=dict(l=250, r=80, t=48, b=40))
    fig2.update_xaxes(range=[float(corr_vals.min()) - corr_pad, float(corr_vals.max()) + corr_pad])

    counts = dff["Diagnosis"].value_counts()
    fig3 = go.Figure(go.Pie(labels=counts.index, values=counts.values,
                             marker_colors=[c0, c1], hole=0.56,
                             textinfo="label+percent+value", pull=[0.05, 0.05],
                             textposition="inside"))
    fig3.update_layout(**PLOT, title="Diagnosis Split", showlegend=False)

    fig4 = px.scatter(
        dff,
        x="age",
        y="thalach",
        color="Diagnosis",
        size="chol",
        color_discrete_map=cmap,
        opacity=0.75,
        size_max=22,
        title="Age vs Max Heart Rate (size = Cholesterol)",
        labels={"age": "Age (yr)", "thalach": "Max HR (bpm)", "chol": "Cholesterol (mg/dL)"},
        hover_data={"age": True, "thalach": True, "chol": True, "Diagnosis": True},
    )
    fig4.update_traces(marker=dict(line=dict(width=1, color="rgba(230,237,243,0.5)")))
    fig4.update_layout(**PLOT, legend=dict(bgcolor="rgba(0,0,0,0)"))

    dff["Sex"] = dff["sex"].map({0.0: "Female", 1.0: "Male"})
    sex_colors = px.colors.sample_colorscale([c0, c1], [0.2, 0.8])
    fig5 = px.histogram(
        dff,
        x="age",
        color="Sex",
        color_discrete_map={"Male": sex_colors[0], "Female": sex_colors[1]},
        barmode="overlay",
        nbins=20,
        opacity=0.75,
        title="Age Distribution by Sex",
        labels={"age": "Age (yr)"},
    )
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
                    value="rf", clearable=False, className="ci-dropdown"),
            ], style={"flex": "1"}),
            html.Div([
                html.Label(f"Top N Features  (3 - {len(FEATURES)})", style=label_s()),
                dcc.Slider(id="fi-n", min=3, max=len(FEATURES), step=1, value=8,
                           marks={i: str(i) for i in range(3, len(FEATURES)+1)},
                           tooltip={"placement": "bottom"}, className="ci-slider"),
            ], style={"flex": "2.5"}),
            
        ], style={"display": "flex", "gap": "24px", "alignItems": "flex-end",
                  "marginBottom": "18px"}),

        html.Div([
            dcc.Graph(id="fi-bar", figure=blank_fig(), style={"flex": "1"}),
            dcc.Graph(id="fi-heatmap", figure=blank_fig(), style={"flex": "1.3"}),
        ], style={"display": "flex", "gap": "14px"}),

        dcc.Graph(id="fi-parallel", figure=blank_fig(), style={"marginTop": "14px"}),
    ])


@app.callback(
    Output("fi-bar",      "figure"),
    Output("fi-heatmap",  "figure"),
    Output("fi-parallel", "figure"),
    Input("fi-algo", "value"),
    Input("fi-n",    "value"),
    Input("global-theme", "value"),
)
def upd_feat(algo, n, theme):
    c0, c1 = THEME_MAP.get(theme, THEME_MAP["br"])
    # Use preprocessed data directly
    Xtr_proc, Xte_proc, ytr, yte = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=42
    )
    
    # For tree-based models, use original feature names; for LR use preprocessed
    if algo == "rf":
        mdl = RandomForestClassifier(n_estimators=200, random_state=42)
        mdl.fit(Xtr_proc, ytr)
        imp = mdl.feature_importances_
        feat_names = preprocessor.get_feature_names_out()
    elif algo == "gb":
        mdl = GradientBoostingClassifier(n_estimators=200, random_state=42)
        mdl.fit(Xtr_proc, ytr)
        imp = mdl.feature_importances_
        feat_names = preprocessor.get_feature_names_out()
    else:  # lr
        mdl = LogisticRegression(max_iter=1000, random_state=42)
        mdl.fit(Xtr_proc, ytr)
        imp = np.abs(mdl.coef_[0])
        imp /= imp.sum()
        feat_names = preprocessor.get_feature_names_out()

    fi   = pd.Series(imp, index=feat_names).sort_values(ascending=False)
    top  = fi.head(n)

    fig1 = go.Figure(go.Bar(
        x=top.values, y=[str(f) for f in top.index],
        orientation="h",
        marker=dict(color=top.values, colorscale=[[0, "#CBD5E1"], [1, c0]], showscale=True,
                    colorbar=dict(thickness=10, tickfont=dict(color=TEXT))),
        text=[f"{v:.3f}" for v in top.values],
        textposition="outside",
        textfont=dict(color=TEXT, size=14, family="JetBrains Mono, monospace"),
        cliponaxis=False,
    ))
    # Reserve right-side headroom so outside labels remain fully visible.
    xmax = float(top.max())
    xpad = max(0.01, xmax * 0.18)
    fig1.update_layout(**PLOT, title=f"Top {n} Features — {ALGO_NAMES[algo]}",
        xaxis_title="Importance Score", height=max(300, 200 + n * 15))
    fig1.update_layout(margin=dict(l=150, r=90, t=48, b=40))
    fig1.update_xaxes(range=[0, xmax + xpad])

    # For correlation, use subset of original features for interpretability
    top_cols = [col for col in FEATURES if any(col in str(f) for f in top.index)] + ["target"]
    top_cols = list(dict.fromkeys(top_cols))[:n+1]  # Remove duplicates, limit to top N
    if "target" not in top_cols:
        top_cols.append("target")
    
    cm = df[top_cols].corr()
    labs = [FEATURE_LABELS.get(c, c) for c in cm.columns]
    fig2 = go.Figure(go.Heatmap(
        z=cm.values, x=labs, y=labs,
        colorscale=[[0, c0], [0.5, "#1F2937"], [1, c1]], zmid=0,
        text=np.round(cm.values, 2), texttemplate="%{text}",
        colorbar=dict(thickness=10, tickfont=dict(color=TEXT)),
    ))
    fig2.update_layout(**PLOT, title="Correlation Matrix (Top Features)", height=380)

    # Parallel coordinates using top original features
    t5_originals = []
    for f in fi.head(5).index:
        for orig in FEATURES:
            if orig in str(f):
                if orig not in t5_originals:
                    t5_originals.append(orig)
                break
    t5_originals = t5_originals[:5]
    
    dff  = df[t5_originals + ["target"]].copy()
    norm = dff.copy()
    for c in t5_originals:
        mn, mx = dff[c].min(), dff[c].max()
        norm[c] = (dff[c] - mn) / (mx - mn + 1e-9)
    dims = [dict(label=FEATURE_LABELS.get(f, f).replace(" ", "<br>"), values=norm[f]) for f in t5_originals]
    dims.append(dict(label="Diagnosis", values=dff["target"],
                     tickvals=[0,1], ticktext=["Healthy","Disease"]))
    fig3 = go.Figure(go.Parcoords(
        line=dict(color=dff["target"],
                  colorscale=[[0, c0], [1, c1]], cmin=0, cmax=1),
        dimensions=dims,
        domain=dict(x=[0, 1], y=[0, 0.88]),
    ))
    fig3.update_layout(**PLOT, title="Parallel Coordinates — Top 5 Features", height=420)
    fig3.update_layout(margin=dict(l=80, r=40, t=90, b=60))

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
                    value=["rf"],
                    inline=True,
                    className="ci-checklist"),
            ], style={"flex": "2"}),
            html.Div([
                html.Label("Test Split %", style=label_s()),
                dcc.Slider(id="ml-split", min=10, max=40, step=5, value=20,
                           marks={i: f"{i}%" for i in range(10, 45, 5)},
                           tooltip={"placement": "bottom"}, className="ci-slider"),
            ], style={"flex": "1.5"}),
        ], style={"display": "flex", "gap": "24px", "alignItems": "flex-end",
                  "marginBottom": "18px"}),

        html.Div([
            html.H4("⚙️  Hyperparameter Tuning",
                    style={"color": C1, "fontSize": "13px", "margin": "0 0 8px",
                           "letterSpacing": "1px", "textTransform": "uppercase"}),
            html.P("Select model(s) above to show and tune their hyperparameters.",
                   style={"color": SUB, "fontSize": "12px", "margin": "0 0 8px"}),

            html.Div([
                html.H5("Random Forest",
                        style={"color": C1, "margin": "0 0 10px", "fontSize": "12px"}),
                html.Div([
                    html.Div([
                        html.Label("Trees (n_estimators)", style=label_s()),
                        dcc.Slider(id="ml-rf-est", min=50, max=350, step=25, value=125,
                                   marks={50: "50", 150: "150", 250: "250", 350: "350"},
                                   tooltip={"placement": "bottom"}, className="ci-slider"),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Max Depth", style=label_s()),
                        dcc.Dropdown(id="ml-rf-depth",
                                     options=[
                                         {"label": "Auto", "value": "none"},
                                         {"label": "4", "value": "4"},
                                         {"label": "6", "value": "6"},
                                         {"label": "8", "value": "8"},
                                         {"label": "12", "value": "12"},
                                         {"label": "16", "value": "16"},
                                     ],
                                     value="none", clearable=False, className="ci-dropdown"),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Min Samples Split", style=label_s()),
                        dcc.Slider(id="ml-rf-minsplit", min=2, max=12, step=1, value=2,
                                   marks={2: "2", 6: "6", 12: "12"},
                                   tooltip={"placement": "bottom"}, className="ci-slider"),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "14px"}),
            ], id="ml-rf-params", style={"display": "none"}),

            html.Div([
                html.H5("Gradient Boosting",
                        style={"color": C4, "margin": "0 0 10px", "fontSize": "12px"}),
                html.Div([
                    html.Div([
                        html.Label("Trees (n_estimators)", style=label_s()),
                        dcc.Slider(id="ml-gb-est", min=50, max=350, step=25, value=125,
                                   marks={50: "50", 150: "150", 250: "250", 350: "350"},
                                   tooltip={"placement": "bottom"}, className="ci-slider"),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Learning Rate", style=label_s()),
                        dcc.Slider(id="ml-gb-lr", min=0.01, max=0.30, step=0.01, value=0.08,
                                   marks={0.01: "0.01", 0.1: "0.1", 0.2: "0.2", 0.3: "0.3"},
                                   tooltip={"placement": "bottom"}, className="ci-slider"),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Tree Depth", style=label_s()),
                        dcc.Dropdown(id="ml-gb-depth",
                                     options=[
                                         {"label": "1", "value": "1"},
                                         {"label": "2", "value": "2"},
                                         {"label": "3", "value": "3"},
                                         {"label": "4", "value": "4"},
                                         {"label": "5", "value": "5"},
                                     ],
                                     value="3", clearable=False, className="ci-dropdown"),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "14px"}),
            ], id="ml-gb-params", style={"display": "none"}),

            html.Div([
                html.H5("Logistic Regression",
                        style={"color": C2, "margin": "0 0 10px", "fontSize": "12px"}),
                html.Div([
                    html.Div([
                        html.Label("Regularization (C)", style=label_s()),
                        dcc.Slider(id="ml-lr-c", min=0.1, max=5.0, step=0.1, value=1.0,
                                   marks={0.1: "0.1", 1: "1", 3: "3", 5: "5"},
                                   tooltip={"placement": "bottom"}, className="ci-slider"),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Penalty", style=label_s()),
                        dcc.Dropdown(id="ml-lr-penalty",
                                     options=[
                                         {"label": "L2", "value": "l2"},
                                         {"label": "None", "value": "none"},
                                     ],
                                     value="l2", clearable=False, className="ci-dropdown"),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Max Iterations", style=label_s()),
                        dcc.Slider(id="ml-lr-iter", min=200, max=1200, step=100, value=700,
                                   marks={200: "200", 600: "600", 1000: "1000", 1200: "1200"},
                                   tooltip={"placement": "bottom"}, className="ci-slider"),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "14px"}),
            ], id="ml-lr-params", style={"display": "none"}),

            html.Div([
                html.H5("SVM",
                        style={"color": C5, "margin": "0 0 10px", "fontSize": "12px"}),
                html.Div([
                    html.Div([
                        html.Label("C", style=label_s()),
                        dcc.Slider(id="ml-svm-c", min=0.1, max=5.0, step=0.1, value=1.0,
                                   marks={0.1: "0.1", 1: "1", 3: "3", 5: "5"},
                                   tooltip={"placement": "bottom"}, className="ci-slider"),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Kernel", style=label_s()),
                        dcc.Dropdown(id="ml-svm-kernel",
                                     options=[
                                         {"label": "RBF", "value": "rbf"},
                                         {"label": "Linear", "value": "linear"},
                                         {"label": "Polynomial", "value": "poly"},
                                     ],
                                     value="rbf", clearable=False, className="ci-dropdown"),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Gamma", style=label_s()),
                        dcc.Dropdown(id="ml-svm-gamma",
                                     options=[
                                         {"label": "Scale", "value": "scale"},
                                         {"label": "Auto", "value": "auto"},
                                     ],
                                     value="scale", clearable=False, className="ci-dropdown"),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "14px"}),
            ], id="ml-svm-params", style={"display": "none"}),
        ], style={**card_s(), "marginBottom": "18px"}),

        html.Div(id="ml-kpis",
                 style={"display": "flex", "gap": "12px",
                        "flexWrap": "wrap", "marginBottom": "18px"}),

        html.Div([
            dcc.Graph(id="ml-roc", figure=blank_fig(), style={"flex": "1"}),
            dcc.Graph(id="ml-cv", figure=blank_fig(), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "14px"}),
    ])


@app.callback(
    Output("ml-rf-params", "style"),
    Output("ml-gb-params", "style"),
    Output("ml-lr-params", "style"),
    Output("ml-svm-params", "style"),
    Input("ml-models", "value"),
)
def toggle_ml_hyperparams(selected):
    selected = selected or []
    base = {
        "background": BG,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "14px",
        "marginTop": "10px",
    }

    def _style_for(model_key):
        return {**base, "display": "block"} if model_key in selected else {**base, "display": "none"}

    return _style_for("rf"), _style_for("gb"), _style_for("lr"), _style_for("svm")


@app.callback(
    Output("ml-kpis", "children"),
    Output("ml-roc",  "figure"),
    Output("ml-cv",   "figure"),
    Input("ml-models","value"),
    Input("ml-split", "value"),
    Input("ml-rf-est", "value"),
    Input("ml-rf-depth", "value"),
    Input("ml-rf-minsplit", "value"),
    Input("ml-gb-est", "value"),
    Input("ml-gb-lr", "value"),
    Input("ml-gb-depth", "value"),
    Input("ml-lr-c", "value"),
    Input("ml-lr-penalty", "value"),
    Input("ml-lr-iter", "value"),
    Input("ml-svm-c", "value"),
    Input("ml-svm-kernel", "value"),
    Input("ml-svm-gamma", "value"),
    Input("global-theme", "value"),
)
def upd_model(selected, split_pct,
              rf_est, rf_depth, rf_min_split,
              gb_est, gb_lr, gb_depth,
              lr_c, lr_penalty, lr_iter,
              svm_c, svm_kernel, svm_gamma,
              theme):
    model_colors = THEME_MODEL_COLORS.get(theme, THEME_MODEL_COLORS["br"])
    empty = go.Figure(); empty.update_layout(**PLOT)
    if not selected:
        return [], empty, empty, empty

    ts = split_pct / 100
    Xtr_proc, Xte_proc, ytr, yte = train_test_split(
        X_preprocessed, y, test_size=ts, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def build(k):
        if k == "rf":
            depth = None if rf_depth in (None, "none") else int(rf_depth)
            return RandomForestClassifier(
                n_estimators=int(rf_est or 125),
                max_depth=depth,
                min_samples_split=int(rf_min_split or 2),
                random_state=42,
                n_jobs=1,
            )
        if k == "gb":
            return GradientBoostingClassifier(
                n_estimators=int(gb_est or 125),
                learning_rate=float(gb_lr or 0.08),
                max_depth=int(gb_depth or 3),
                random_state=42,
            )
        if k == "lr":
            penalty = "none" if lr_penalty == "none" else "l2"
            return LogisticRegression(
                C=float(lr_c or 1.0),
                penalty=penalty,
                max_iter=int(lr_iter or 700),
                solver="lbfgs",
                random_state=42,
            )
        return SVC(
            C=float(svm_c or 1.0),
            kernel=svm_kernel or "rbf",
            gamma=svm_gamma or "scale",
        )

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color=BORDER),
                                  showlegend=False))
    kpis, primary_cm, cv_traces = [], None, []

    for k in selected:
        mdl = build(k)
        # Fast mode: lighter defaults + 3-fold CV for responsive UI
        mdl.fit(Xtr_proc, ytr)
        yp  = mdl.predict(Xte_proc)
        if hasattr(mdl, "predict_proba"):
            ypr = mdl.predict_proba(Xte_proc)[:, 1]
        else:
            ypr = mdl.decision_function(Xte_proc)

        acc = accuracy_score(yte, yp)
        f1w = f1_score(yte, yp, average="weighted")
        fpr, tpr, _ = roc_curve(yte, ypr)
        ra  = auc(fpr, tpr)
        cvs = cross_val_score(mdl, Xtr_proc, ytr, cv=cv, scoring="accuracy")

        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{ALGO_NAMES[k]} (AUC={ra:.2f})",
            line=dict(color=model_colors[k], width=2.5)))

        kpis.append(html.Div([
            html.P(ALGO_NAMES[k],
                 style={"color": model_colors[k], "margin": "0 0 6px",
                          "fontSize": "12px", "letterSpacing": "1px",
                          "textTransform": "uppercase"}),
            html.H3(f"{acc:.1%}",
                    style={"margin": "0", "fontSize": "32px",
                           "color": TEXT, "fontWeight": "700"}),
                 html.P(f"AUC {ra:.3f}  ·  F1 {f1w:.3f}",
                   style={"margin": "6px 0 0", "fontSize": "12px", "color": SUB}),
         ], style={**card_s(), "borderTop": f"3px solid {model_colors[k]}",
                  "minWidth": "155px"}))

        if primary_cm is None:
            primary_cm = (confusion_matrix(yte, yp), k)
        cv_traces.append((k, cvs))

    fig_roc.update_layout(**PLOT, title="ROC Curves",
                           xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate",
                           legend=dict(bgcolor="rgba(0,0,0,0)", x=0.5, y=0.05))

    fig_cv = go.Figure()
    for k, cvs in cv_traces:
        fig_cv.add_trace(go.Box(y=cvs, name=ALGO_NAMES[k],
                                 marker_color=model_colors[k], boxmean=True))
    fig_cv.update_layout(**PLOT, title="3-Fold CV Accuracy (Fast)",
                          yaxis_title="Accuracy", showlegend=False)

    return kpis, fig_roc, fig_cv


# ══════════════════════════════════════════════════════════════
# TAB 4 — PATIENT PREDICTION
# ══════════════════════════════════════════════════════════════
# Only base features are sliders (engineered features are computed)
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
    ("thal",     "Thalassemia",                    1,  3,   1,     1, {1:"Normal", 2:"Fixed defect", 3:"Reversible"}),
]
PRED_IDS = [r[0] for r in PRED_SLIDERS]

# Train prediction models on preprocessed data
_PRED_MODELS = {
    "rf": RandomForestClassifier(n_estimators=200, random_state=42).fit(X_preprocessed, y),
    "gb": GradientBoostingClassifier(n_estimators=200, random_state=42).fit(X_preprocessed, y),
    "lr": LogisticRegression(max_iter=1000, random_state=42).fit(X_preprocessed, y),
}

def pred_layout():
    def row(fid, lbl, mn, mx, step, val, marks):
        m = marks or {mn: str(mn), mx: str(mx)}
        return html.Div([
            html.Label(lbl, style={"color": SUB, "fontSize": "13px",
                                   "width": "210px", "flexShrink": "0"}),
            dcc.Slider(id=f"p-{fid}", min=mn, max=mx, step=step, value=val,
                       marks=m, tooltip={"placement":"top","always_visible":False},
                      className="ci-slider ci-slider-flex"),
            html.Span(id=f"p-{fid}-v", children=str(val),
                      style={"color": C1, "fontSize": "13px", "fontWeight": "700",
                             "width": "48px", "textAlign": "right", "flexShrink": "0"}),
        ], style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"9px"})

    return html.Div([
        html.Div([
            # Input panel
            html.Div([
                html.H3("🩺  Patient Parameters",
                        style={"color": C1, "margin": "0 0 20px", "fontSize": "16px",
                               "letterSpacing": "1px", "textTransform": "uppercase",
                               "borderBottom": f"1px solid {BORDER}",
                               "paddingBottom": "12px"}),
                *[row(*r) for r in PRED_SLIDERS],
                html.Div([
                    html.Label("Prediction Model", style=label_s()),
                    dcc.Dropdown(id="p-algo",
                        options=[{"label":"Random Forest",     "value":"rf"},
                                 {"label":"Gradient Boosting", "value":"gb"},
                                 {"label":"Logistic Regression","value":"lr"}],
                        value="rf", clearable=False, className="ci-dropdown"),
                ], style={"marginTop": "18px"}),
            ], style={**card_s(flex="1")}),

            # Output panel
            html.Div([
                html.H3("📈  Result",
                        style={"color": C1, "margin": "0 0 20px", "fontSize": "16px",
                               "letterSpacing": "1px", "textTransform": "uppercase",
                               "borderBottom": f"1px solid {BORDER}",
                               "paddingBottom": "12px"}),
                html.Div(id="p-banner"),
                dcc.Graph(id="p-gauge", figure=blank_fig(280), style={"height": "280px"}),
                dcc.Graph(id="p-contrib", figure=blank_fig(380), style={"height": "380px"}),
            ], style={**card_s(flex="1")}),
        ], style={"display": "flex", "gap": "20px"}),
        
        # Clinical Details Section (below all other content)
        html.Div([
            html.H3("📋 Clinical Analysis",
                    style={"color": C1, "margin": "0 0 20px", "fontSize": "20px",
                           "letterSpacing": "1px", "textTransform": "uppercase",
                           "borderBottom": f"1px solid {BORDER}",
                           "paddingBottom": "12px"}),
            
            # Clinical Profile Card
            html.Div([
                html.H4("📋 Clinical Profile", style={
                    "color": C1, "fontSize": "18px", "margin": "0 0 12px",
                    "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase"
                }),
                html.P(id="p-description", style={
                    "color": TEXT, "fontSize": "16px", "lineHeight": "1.95",
                    "margin": "0", "padding": "16px 18px", "background": f"{C1}14",
                    "borderRadius": "8px", "borderLeft": f"4px solid {C1}",
                }),
            ], style={"marginBottom": "14px"}),
            
            # Advice Card
            html.Div([
                html.H4(id="p-advice-title", style={
                    "fontSize": "18px", "margin": "0 0 12px",
                    "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase"
                }),
                html.P(id="p-advice", style={
                    "fontSize": "16px", "lineHeight": "2.0", "whiteSpace": "pre-wrap",
                    "margin": "0", "padding": "16px 18px",
                    "borderRadius": "8px",
                    "fontFamily": "JetBrains Mono, monospace"
                }),
            ]),
        ], style={**card_s(), "marginTop": "20px"}),
    ])


# Live slider display
for _fid, *_ in PRED_SLIDERS:
    @app.callback(Output(f"p-{_fid}-v", "children"), Input(f"p-{_fid}", "value"))
    def _disp(v, fid=_fid): return str(v)


def generate_clinical_summary(vals, prob, pred):
    """Generate clinical description and personalized advice based on patient values."""
    age = int(vals['age'])
    sex = "male" if vals['sex'] == 1 else "female"
    cp_type = int(vals['cp'])
    has_hypertension = vals['trestbps'] >= 140
    has_high_chol = vals['chol'] >= 240
    has_exang = vals['exang'] == 1
    oldpeak_val = vals['oldpeak']
    hr_reserve = vals['thalach'] / (220 - age)
    
    # ─── BUILD CLINICAL DESCRIPTION ───
    description = f"This {age}-year-old {sex} patient "
    
    # Describe symptom/presentation
    cp_descriptions = {
        0: "presents with asymptomatic status",
        1: "reports atypical angina symptoms",
        2: "experiences non-anginal chest pain",
        3: "has typical angina symptoms"
    }
    description += cp_descriptions.get(cp_type, "")
    
    # Add key risk factors
    risk_factors = []
    if has_hypertension:
        risk_factors.append(f"elevated blood pressure ({int(vals['trestbps'])} mmHg)")
    if has_high_chol:
        risk_factors.append(f"high cholesterol ({int(vals['chol'])} mg/dL)")
    if has_exang:
        risk_factors.append("exercise-induced angina")
    if oldpeak_val >= 1.0:
        risk_factors.append(f"significant ST depression ({oldpeak_val:.1f} mm)")
    
    if risk_factors:
        description += " with " + ", ".join(risk_factors) + "."
    else:
        description += " with no significant risk factors noted."
    
    description += f" The model indicates a {prob:.0%} probability of heart disease."
    
    # ─── BUILD PERSONALIZED ADVICE ───
    if pred:
        advice = "⚠️  HIGH RISK — ACTION RECOMMENDED\n\n"
        if has_hypertension and has_high_chol:
            advice += "• Urgent medical evaluation needed\n• Aggressive BP & cholesterol management with medication\n• Reduce salt intake (<2,300mg daily) and saturated fats\n• Exercise under medical supervision only\n• Monitor BP regularly at home"
        elif has_hypertension:
            advice += "• Consult cardiologist promptly\n• Focus on BP control through medication\n• Practice stress reduction techniques (yoga, meditation)\n• Limit sodium intake to <2,300mg daily\n• Regular aerobic exercise (with medical clearance)"
        elif has_high_chol:
            advice += "• Get comprehensive cardiac workup\n• Start or continue statin therapy\n• Increase soluble fiber intake\n• Reduce trans fats and processed foods\n• Consider lipid panel monitoring monthly"
        else:
            advice += "• Schedule cardiac evaluation soon\n• Consider stress testing (ECG/imaging)\n• Assess exercise capacity with physician\n• Lifestyle modifications to reduce future risk\n• Follow-up cardiology visit recommended"
    else:
        advice = "[OK] LOW RISK MAINTAIN HEALTHY HABITS\n\n"
        if age >= 50:
            advice += "• Continue regular health screening (every 2-3 years)\n• Annual cardiovascular checkup recommended\n• Maintain current healthy lifestyle\n• Consider periodic stress testing per physician\n• Monitor vital signs regularly"
        else:
            advice += "• Maintain heart-healthy lifestyle\n• Exercise 150 minutes/week (moderate intensity)\n• Eat balanced diet (Mediterranean style preferred)\n• Keep annual follow-up visits\n• Recheck status at age 40-50"
    
    return description, advice


@app.callback(
    Output("p-banner",  "children"),
    Output("p-gauge",   "figure"),
    Output("p-contrib", "figure"),
    Output("p-description", "children"),
    Output("p-advice-title", "children"),
    Output("p-advice-title", "style"),
    Output("p-advice",  "children"),
    Output("p-advice",  "style"),
    [Input(f"p-{fid}", "value") for fid in PRED_IDS],
    Input("p-algo", "value"),
    Input("global-theme", "value"),
)
def upd_pred(*args):
    vals = {fid: float(v) for fid, v in zip(PRED_IDS, args[:len(PRED_IDS)])}
    algo = args[-2]
    theme = args[-1]
    healthy_color, disease_color = THEME_MAP.get(theme, THEME_MAP["br"])
    
    # Build dataframe with user values and compute engineered features
    patient_data = pd.DataFrame([vals])
    patient_data['hypertension'] = (patient_data['trestbps'] >= 140).astype(int)
    patient_data['high_chol'] = (patient_data['chol'] >= 240).astype(int)
    patient_data['hr_reserve_pct'] = patient_data['thalach'] / (220 - patient_data['age'])
    
    # Extract in correct feature order and preprocess
    patient_features = patient_data[feature_cols]
    patient_preprocessed = preprocessor.transform(patient_features)
    
    # Get prediction
    mdl = _PRED_MODELS[algo]
    prob = mdl.predict_proba(patient_preprocessed)[0][1]
    pred = int(prob >= 0.5)

    color = disease_color if pred else healthy_color
    icon  = "[!]" if pred else "[OK]"
    label = "HIGH RISK — Disease Likely" if pred else "LOW RISK — No Disease Detected"
    
    # Generate clinical summary
    description, advice = generate_clinical_summary(vals, prob, pred)

    # Banner - only risk assessment
    banner = html.Div([
        html.Div(f"{icon}  {label}", style={
            "color": color, "fontSize": "17px", "fontWeight": "700",
            "padding": "16px 20px", "borderRadius": "8px",
            "border": f"2px solid {color}", "textAlign": "center",
            "background": f"{color}15", "marginBottom": "16px",
        }),
        html.P(f"Estimated probability of disease: {prob:.1%}",
               style={"color": SUB, "fontSize": "13px",
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
                {"range": [0,  35], "color": hex_to_rgba(healthy_color, 0.13)},
                {"range": [35, 65], "color": "rgba(255, 166, 87, 0.13)"},
                {"range": [65, 100],"color": hex_to_rgba(disease_color, 0.13)},
            ],
            "threshold": {"line": {"color": TEXT, "width": 3},
                          "thickness": 0.85, "value": 50},
        },
        title={"text": "Risk Score", "font": {"color": SUB, "size": 13}},
    ))
    fig_g.update_layout(**PLOT, height=260)

    # Perturbation-based feature contribution (using original features)
    X_means = df[PRED_IDS].mean().to_dict()
    contrib = {}
    for feat in PRED_IDS:
        alt_vals = vals.copy()
        alt_vals[feat] = X_means[feat]
        
        # Build altered patient data and preprocess
        alt_patient = pd.DataFrame([alt_vals])
        alt_patient['hypertension'] = (alt_patient['trestbps'] >= 140).astype(int)
        alt_patient['high_chol'] = (alt_patient['chol'] >= 240).astype(int)
        alt_patient['hr_reserve_pct'] = alt_patient['thalach'] / (220 - alt_patient['age'])
        
        alt_features = alt_patient[feature_cols]
        alt_preprocessed = preprocessor.transform(alt_features)
        
        alt_prob = mdl.predict_proba(alt_preprocessed)[0][1]
        contrib[feat] = prob - alt_prob

    cs   = pd.Series(contrib).sort_values()
    cs_vals = cs.values.astype(float)
    cs_pad = max(0.01, (float(cs_vals.max()) - float(cs_vals.min())) * 0.20)
    bcol = [disease_color if v > 0 else healthy_color for v in cs.values]
    fig_c = go.Figure(go.Bar(
        x=cs.values, y=[FEATURE_LABELS[f] for f in cs.index],
        orientation="h", marker_color=bcol,
        text=[f"{v:+.3f}" for v in cs.values],
        textposition="outside",
        textfont=dict(color=TEXT, size=16, family="JetBrains Mono, monospace"),
        cliponaxis=False,
    ))
    fig_c.update_layout(**PLOT,
        title="Feature Contribution to Risk",
        xaxis_title="Δ Probability", height=360)
    fig_c.update_layout(margin=dict(l=240, r=90, t=48, b=40))
    fig_c.update_xaxes(range=[float(cs_vals.min()) - cs_pad, float(cs_vals.max()) + cs_pad])
    
    # Advice title styling
    advice_title = "💡 Personalized Advice"
    advice_title_style = {
        "color": color, "fontSize": "18px", "margin": "0 0 12px",
        "fontWeight": "700", "letterSpacing": "0.5px", "textTransform": "uppercase"
    }
    
    # Advice styling
    advice_style = {
        "color": TEXT, "fontSize": "16px", "lineHeight": "2.0", "whiteSpace": "pre-wrap",
        "margin": "0", "padding": "16px 18px", "background": f"{color}14",
        "borderRadius": "8px", "borderLeft": f"4px solid {color}",
        "fontFamily": "JetBrains Mono, monospace"
    }

    return banner, fig_g, fig_c, description, advice_title, advice_title_style, advice, advice_style


# ──────────────────────────────────────────────────────────────
# 6.  ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n  ❤️  CardioInsight →  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
