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

An interactive clinical analytics dashboard built with **Dash + Plotly** using the **real UCI Cleveland Heart Disease dataset**.

## Features

| Tab | What it does |
|-----|-------------|
| 📊 Exploration | Histogram / Box / Violin per feature, correlation chart, scatter, data table |
| 🔍 Feature Importance | RF / GBM / LR importances, correlation heatmap, parallel coordinates |
| 🤖 Model Performance | Compare 4 models — ROC curves, confusion matrix, cross-validation |
| 🩺 Predict Patient | Adjust 13 clinical sliders → live risk gauge + feature contribution chart |

## Dataset
UCI Heart Disease (Cleveland) — 303 patients, 13 clinical features, binary target (disease / no disease).  
Auto-fetched from the UCI ML Repository on startup.
