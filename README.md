# CardioInsight - Heart Disease Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-2.17.1-008DE4?logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![Plotly](https://img.shields.io/badge/Plotly-5.22.0-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/python/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Dash-healthcare/cardioinsight-ml-dashboard)

Interactive healthcare analytics dashboard built with Dash and Plotly for exploring, modeling, and simulating heart disease risk using the UCI Cleveland dataset.

## 🌐 Live Demo

[View on Hugging Face Spaces](https://huggingface.co/spaces/Dash-healthcare/cardioinsight-ml-dashboard)

## 📊 What This Project Includes

- Interactive exploratory analysis with multiple chart modes
- Global palette theming for consistent visuals across all tabs
- Feature importance analysis across multiple ML algorithms
- Model comparison with ROC, confusion matrix, and cross-validation
- Patient-level risk prediction with an interactive clinical profile
- Docker-ready deployment for Hugging Face Spaces
- GitHub Actions workflow to sync repo updates to Hugging Face

## 🎯 Key Features

### Exploration Tab
- Feature selector for all clinical variables
- Chart mode switcher adapts by feature type (Histogram, Box Plot, KDE for numerical; Bar, Stacked Bar, Heatmap, Pie for categorical)
- Global palette selector applies across all tabs
- Correlation-with-target bar chart
- Static comparison row with 3 interactive charts

### Feature Importance Tab
- Algorithm selector (Random Forest, Gradient Boosting, Logistic Regression)
- Top-N feature slider
- Importance ranking bar chart
- Correlation heatmap for selected features
- Parallel coordinates plot for top 5 features

### Model Performance Tab
- Multi-model comparison checklist
- Hyperparameter tuning (optimized for immediate feedback)
- Adjustable train/test split (10% to 40% test)
- KPI cards (Accuracy, AUC, Weighted F1)
- ROC curve comparison
- Confusion matrix visualization
- 3-fold stratified cross-validation accuracy plots

### Predict Patient Tab
- 13 interactive patient input sliders
- Real-time risk probability prediction
- Risk status banner (Low/High risk)
- Risk gauge visualization
- **AI Clinical Summary** (Gemini LLM with automatic failover)
- Feature contribution analysis

## 💻 Tech Stack

- **Python 3.11**
- **Dash 2.17.1** - Interactive web framework
- **Plotly 5.22.0** - Data visualization
- **scikit-learn 1.5.0** - Machine learning models
- **pandas 2.2.2** - Data manipulation
- **numpy 1.26.4** - Numerical computing
- **gunicorn 22.0.0** - Production server
- **requests 2.32.3** - HTTP client
- **python-dotenv 1.0.1** - Environment management

## 🚀 Quick Start

### Local Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY
python app.py
```

### Local Setup (Linux/macOS)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
python app.py
```

Visit: http://127.0.0.1:7860

### Docker Deployment

```bash
docker build -t cardioinsight .
docker run --rm -p 7860:7860 cardioinsight
```

Visit: http://127.0.0.1:7860

## 🔧 Environment Variables

Create `.env` from `.env.example`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_TIMEOUT_SECONDS=20
```

For Hugging Face Spaces, set these as Space Secrets (do not commit `.env`).

## 📦 Dataset and Preprocessing

**Primary Source:**
- UCI Cleveland Heart Disease Dataset: [Archive](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)

**Fallback Source:**
- Local bundled file: `data/Heart_disease.csv`

**Preprocessing Pipeline:**
- Load bundled CSV
- Cast all columns to numeric float
- Binarize target: `0` = no disease, `1..4` = disease (mapped to `1`)

**Feature Engineering:**
- Hypertension flag (BP ≥ 140 mmHg)
- High cholesterol flag (Cholesterol ≥ 240 mg/dL)
- HR Reserve % (normalized heart rate capacity)

**Scaling Strategy:**
- **StandardScaler**: Age, max heart rate, HR reserve % (normally distributed)
- **RobustScaler**: Blood pressure, cholesterol, ST depression (outlier-prone)
- **OrdinalEncoder**: Chest pain, ECG, ST slope, vessels, thalassemia (categorical)

**Dataset Summary:**
- Total records: 303 patients
- Input features: 13 base + 3 engineered = 16 total
- Missing values: None
- Class balance: ~54% no disease / ~46% disease

## 🔄 CI/CD to Hugging Face

The GitHub Actions workflow (`.github/workflows/spaces_publish.yml`) syncs this repository to Hugging Face Spaces on every push to `main`.

**Behavior:**
- GitHub README stays clean (no Hugging Face metadata)
- During CI, metadata header is prepended before pushing to Hugging Face
- Force push ensures HF Space stays aligned

**Required Secret:**
- `HF_TOKEN` with write permission to `Dash-healthcare/cardioinsight-ml-dashboard`

## 📁 Project Structure

```
.
├── app.py                          # Main Dash application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
├── README.md                       # This file
├── .env.example                    # Environment template
├── .github/
│   └── workflows/
│       └── spaces_publish.yml      # CI/CD workflow
├── assets/
│   └── style.css                   # UI styling
├── data/
│   └── Heart_disease.csv           # Dataset
└── notebooks/
    └── heart_disease_eda.ipynb     # EDA & preprocessing
```

## 📔 Exploratory Data Analysis

The **`notebooks/heart_disease_eda.ipynb`** Jupyter notebook contains:
- Data quality assessment and preprocessing
- Distribution analysis of clinical features
- Correlation analysis with target variable
- Statistical summaries and outlier detection
- Feature engineering walkthrough
- Preliminary model evaluation

This notebook informs the interactive dashboard design.

## ⚖️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📋 Disclaimer

This project is for **educational analytics and machine learning exploration only**. It is **not** intended as a medical diagnosis tool or substitute for professional medical advice. Always consult qualified healthcare professionals for clinical decisions.

## 🛠️ Notes

- The dashboard retrains models in real-time, enabling instant feedback on hyperparameter adjustments
- Optimized for explainability and responsiveness:
  - 3-fold stratified cross-validation for faster evaluation
  - Weighted F1 score for efficient metric computation
  - SVM decision function optimization to avoid calibration overhead
- All preprocessing and feature engineering happens automatically
- Predictions are explained through clinical narrative and feature contribution analysis
