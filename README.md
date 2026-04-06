---
title: CardioInsight Heart Disease Dashboard
emoji: ❤️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: Interactive Heart Disease Analytics — UCI Cleveland Dataset
---

# ❤️ CardioInsight — Heart Disease Analytics Dashboard

CardioInsight is an interactive healthcare analytics app built with Dash + Plotly for exploring, modeling, and simulating heart disease risk using the UCI Cleveland dataset.

Live Space:
https://huggingface.co/spaces/Dash-healthcare/cardioinsight-ml-dashboard

## Quick Start (2 Minutes)

### Local Run

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:
http://127.0.0.1:7860

### Docker Run

```bash
docker build -t cardioinsight .
docker run --rm -p 7860:7860 cardioinsight
```

Open:
http://127.0.0.1:7860

## What This Project Includes

- Interactive exploratory analysis with multiple chart modes
- Global palette theming for consistent visuals across all tabs
- Feature importance analysis across multiple ML algorithms
- Model comparison with ROC, confusion matrix, and cross-validation
- Patient-level risk prediction with an interactive clinical profile
- Docker-ready deployment for Hugging Face Spaces
- GitHub Actions workflow to sync repo updates to Hugging Face

## Full Feature Breakdown

### 1. Exploration Tab

- Feature selector for all clinical variables
- Chart mode switcher adapts by feature type:
	- Numerical: Histogram, Box Plot, KDE
	- Categorical: Bar Chart, Stacked Bar, Heatmap, Pie
- Global palette selector (in header) applies across all tabs and all Exploration visuals
- Correlation-with-target bar chart
- Static comparison row with 3 charts:
	- Age vs Max Heart Rate bubble scatter (bubble size = cholesterol)
	- Diagnosis split donut chart
	- Age distribution by sex chart

### 2. Feature Importance Tab

- Algorithm selector:
	- Random Forest
	- Gradient Boosting
	- Logistic Regression
- Top-N feature slider
- Importance ranking bar chart
- Correlation heatmap for selected top features
- Parallel coordinates plot for top 5 features

### 3. Model Performance Tab

- Multi-model comparison checklist:
	- Random Forest
	- Gradient Boosting
	- Logistic Regression
	- SVM
- **Hyperparameter Tuning** (optimized for immediate feedback):
	- Random Forest: n_estimators, max_depth, min_samples_split
	- Gradient Boosting: n_estimators, learning_rate, max_depth
	- Logistic Regression: C (regularization), penalty type, max_iter
	- SVM: C, kernel type (RBF/Linear/Poly), gamma
	- Dynamic UI visibility (controls shown only when model selected)
- Adjustable train/test split (10% to 40% test)
- KPI cards per selected model:
	- Accuracy
	- AUC
	- Weighted F1 score
- ROC curve comparison
- Confusion matrix (first selected model)
- 3-fold stratified cross-validation accuracy box plots (optimized for speed)

### 4. Predict Patient Tab

- 13 interactive patient input sliders (all core clinical features)
- Live display for each slider value
- Prediction model selector:
	- Random Forest
	- Gradient Boosting
	- Logistic Regression
- Real-time risk probability prediction
- Risk status banner:
	- Low risk
	- High risk
- Risk gauge chart
- **Clinical Summary Analysis**:
	- Patient description: Demographics, key symptoms, identified risk factors (hypertension, high cholesterol, exercise-induced angina)
	- Personalized advice: Algorithm-specific and age-adjusted recommendations based on risk level
- Feature contribution chart using perturbation-based sensitivity analysis

## Dataset and Preprocessing

Primary source:
- UCI Cleveland heart disease data:
	https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

Fallback source:
- Local bundled file: data/Heart_disease.csv

Preprocessing pipeline:
- Load bundled CSV from `data/Heart_disease.csv`
- Drop rows with missing values (none in the bundled dataset)
- Cast all columns to numeric float
- Binarize target:
	- `0` -> no disease
	- `1..4` -> disease (mapped to `1`)
- **Automatic Feature Engineering**:
	- Hypertension flag: Binary indicator (BP ≥ 140 mmHg)
	- High Cholesterol flag: Binary indicator (Cholesterol ≥ 240 mg/dL)
	- HR Reserve %: Normalized heart rate capacity (max_heart_rate / (220 - age))

Preprocessing strategy by feature type:
- **StandardScaler**: Age, max heart rate, HR reserve % (normally distributed)
- **RobustScaler**: Blood pressure, cholesterol, ST depression (outlier-prone)
- **OrdinalEncoder**: Chest pain, ECG status, ST slope, vessels, thalassemia (categorical)
- **SimpleImputer**: Handles missing values per feature type (median for numeric, most frequent for categorical)

Current dataset summary after loading:
- Raw rows: 303
- Missing rows removed: none in bundled file
- Input features: 13 base + 3 engineered = 16 total

## Tech Stack

- Python 3.11
- Dash 2.17.1
- Plotly 5.22.0
- pandas 2.2.2
- numpy 1.26.4
- scikit-learn 1.5.0
- gunicorn 22.0.0

## Project Structure

```text
my-first-dash-app/
├── app.py
├── requirements.txt
├── Dockerfile
├── DEPLOY.md
├── README.md
├── .github/
│   └── workflows/
│       └── spaces_publish.yml
├── data/
│   └── Heart_disease.csv
└── notebooks/
	└── heart_disease_eda.ipynb
```

## Exploratory Data Analysis Notebook

The **`notebooks/heart_disease_eda.ipynb`** Jupyter notebook contains comprehensive exploratory analysis including:
- Data quality assessment and preprocessing
- Distribution analysis of clinical features
- Correlation analysis with target variable
- Statistical summaries and outlier detection
- Feature engineering insights
- Preliminary model evaluation

This notebook informs the design and feature selection in the interactive dashboard.

## Run Locally

### 1. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the app

```bash
python app.py
```

Open:
http://127.0.0.1:7860

## Run with Docker

Build:

```bash
docker build -t cardioinsight .
```

Run:

```bash
docker run --rm -p 7860:7860 cardioinsight
```

Open:
http://127.0.0.1:7860

## Deploy to Hugging Face Spaces

This repo is configured for a Docker-based Hugging Face Space.

Target Space:
https://huggingface.co/spaces/Dash-healthcare/cardioinsight-ml-dashboard

Quick Git workflow:

```bash
git clone https://huggingface.co/spaces/Dash-healthcare/cardioinsight-ml-dashboard
cd cardioinsight-ml-dashboard
# copy project files into this repo
git add .
git commit -m "Update CardioInsight"
git push
```

Detailed deployment steps are available in DEPLOY.md.

## CI/CD Sync Workflow

GitHub Actions workflow file:
- .github/workflows/spaces_publish.yml

What it does:
- On push to `main`, pushes code to the Hugging Face Space repo.

Required secret:
- `HF_TOKEN` with write permission to the target Space.

## Troubleshooting

- `You are not authorized to push to this repo`
	- Ensure the Hugging Face token has write access.
	- Ensure your account has write permission on `Dash-healthcare/cardioinsight-ml-dashboard`.
- App fails at startup on Space
	- Check Space logs for dependency or import errors.
- Dataset fetch fails from UCI
	- App automatically falls back to `data/Heart_disease.csv`.
- Local package mismatch errors
	- Use a clean virtual environment and reinstall from `requirements.txt`.

## Notes

- The dashboard retrains models and evaluates interactively in real-time, enabling instant feedback on hyperparameter adjustments
- The app is tuned for **explainability and responsiveness**: 
	- 3-fold stratified cross-validation for faster model evaluation
	- Weighted F1 score for faster metric computation
	- SVM decision function optimization to avoid probability calibration overhead
- The app is intended for educational analytics and model exploration, not as a medical diagnosis system
- All data preprocessing and feature engineering happens automatically; raw predictions are explained through clinical narrative and feature contribution analysis
