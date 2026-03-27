# 🚀 CardioInsight — Hugging Face Spaces Deployment Guide

## What's in this folder

```
hf_space/
├── app.py              ← Dash application (main entry point)
├── requirements.txt    ← Python dependencies
├── Dockerfile          ← Container config for HF Spaces
├── README.md           ← Space description (shown on HF page)
└── data/
    └── heart.csv       ← Bundled fallback dataset
```

---

## Step-by-Step Deployment

### 1.  Create a Hugging Face account
Go to https://huggingface.co and sign up (free).

---

### 2.  Create a new Space

1. Click your profile icon → **New Space**
2. Fill in:
   - **Space name**: `cardioinsight` (or anything you like)
   - **License**: MIT
   - **SDK**: **Docker**   ← important!
   - **Hardware**: CPU Basic (free)
3. Click **Create Space**

---

### 3.  Upload your files

**Option A — Web UI (easiest)**
1. Open your new Space → click **Files** tab
2. Click **+ Add file → Upload files**
3. Upload ALL files:
   - `app.py`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`
   - `data/heart.csv`  (drag into a `data/` subfolder)

**Option B — Git (recommended)**
```bash
# Install git-lfs if needed
git lfs install

# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/cardioinsight
cd cardioinsight

# Copy all project files here
cp /path/to/hf_space/* .
mkdir -p data && cp /path/to/hf_space/data/heart.csv data/

# Push
git add .
git commit -m "Initial deploy: CardioInsight Dashboard"
git push
```

---

### 4.  Watch it build

- Go to your Space page → click the **Logs** tab
- Build takes ~3–5 minutes (installing packages)
- When you see `Booting up application`, it's live ✅

---

### 5.  Open your live dashboard

Your dashboard will be at:
```
https://huggingface.co/spaces/YOUR_USERNAME/cardioinsight
```

---

## How the dataset loads

On startup `app.py` tries:
1. **Remote** — fetches `processed.cleveland.data` directly from the UCI ML Repository
2. **Fallback** — if the network call fails, loads `data/heart.csv` (bundled)

Either way you always get the real UCI dataset.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails on `pip install` | Check `requirements.txt` versions match above |
| "Application failed to start" | Check Logs tab — usually a missing import |
| Blank page | Wait 30s more; large model training on first load |
| Dataset loads 0 rows | HF Space has no outbound internet — fallback CSV is used automatically |

---

## Local testing before deploy

```bash
cd hf_space
pip install -r requirements.txt
python app.py
# → open http://127.0.0.1:7860
```
