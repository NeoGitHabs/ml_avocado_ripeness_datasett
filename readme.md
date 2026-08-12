# Avocado Ripeness Classification API

> Classifies the ripeness stage of avocados from physical sensor measurements — a prototype for real-time quality control on packing lines and in smart retail systems.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()
[![Dataset](https://img.shields.io/badge/Dataset-synthetic%20%7C%20250%20rows-yellow)]()

---

## ⚠️ Known Issue — Read Before Using in Production

This repo currently has **two mismatches between the notebook and the API** that should be fixed before this is presented as a finished deployment:

1. **Class schema mismatch.** The notebook trains on the real target column `ripeness`, which has **5 classes**: `hard`, `breaking`, `pre-conditioned`, `firm-ripe`, `ripe`. `main.py`'s `RIPENESS_LABELS` dict only knows 3 (`unripe`, `ripe`, `overripe`) — the API's label formatting predates the current dataset/notebook and needs to be regenerated from `df['ripeness'].unique()`.
2. **`color_category` mismatch.** The dataset contains **4** categories (`black`, `green`, `dark green`, `purple`). `main.py`'s `VALID_COLORS` / `COLOR_CATEGORIES` only allow 3 (`dark green`, `green`, `purple`) — any request with `"black"` is currently rejected by the API even though it's a valid category the model was trained on.

Both are one-line fixes (regenerate the label list and category list from the notebook's `df['ripeness'].unique()` and `df['color_category'].unique()`), but until that's done, **don't ship this as-is** — this README documents what the notebook actually produced, not a patched version of the API.

---

## Business Problem

Grocery retailers and food distributors lose a meaningful share of perishable inventory annually to incorrect ripeness staging. Manual inspection is inconsistent and doesn't scale on a fast sorting line. This project explores automating ripeness classification from measurable physical properties (firmness, color, weight, acoustic response) as a first step toward consistent, human-independent grading.

---

## Project Structure

```
ml_AvocadoRipenessDataset/
├── .gitignore
├── readme.md
├── requirements.txt
└── AvocadoRipenessDataset/
    ├── AvocadoRipenessDataset.ipynb          # EDA + model comparison
    ├── main.py                               # FastAPI inference service
    ├── model_rf_AvocadoRipenessDataset.pkl   # deployed model (Random Forest)
    ├── scaler_AvocadoRipenessDataset.pkl     # StandardScaler used at inference
    ├── dataset/
    │   ├── DSAvocado.docx
    │   └── avocado_ripeness_dataset.csv
    └── Test.txt
```

---

## Demo

**POST** `http://127.0.0.1:8000/predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "firmness": 71.7,
    "hue": 53,
    "saturation": 69,
    "brightness": 75,
    "color_category": "green",
    "sound_db": 69,
    "weight_g": 206,
    "size_cm3": 185
  }'
```

**Response** (actual shape returned by `main.py`):
```json
{
  "status": "ripe",
  "probabilities": {
    "breaking": 0.02,
    "firm-ripe": 0.05,
    "hard": 0.0,
    "pre-conditioned": 0.03,
    "ripe": 0.9
  }
}
```

> `color_category` currently accepted by the API: `"dark green"`, `"green"`, `"purple"` — see the mismatch note above; the dataset also contains `"black"`.

---

## Results

Three models were trained and compared: Logistic Regression, Decision Tree, and Random Forest — all with default (unconstrained) hyperparameters. **All three scored a perfect 1.00 across every metric, on both train and test:**

| Model | Train Accuracy | Test Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|---|---|---|---|---|---|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **Random Forest** ✅ deployed | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

*(test set: 50 rows across 5 classes — every class hit precision/recall/F1 = 1.00)*

### Why this is a caveat, not a headline number

A perfect score from three structurally different models at once — a linear model, a single tree, and an ensemble — almost never means "the model is great." It means **the classes are trivially separable given the features**, which is expected for a small (250-row) **synthetic** dataset where `ripeness` was very likely generated from threshold rules on `firmness` / `sound_db` / `hue` rather than sampled from noisy real-world sensor readings. The EDA boxplots in the notebook (firmness/hue/saturation/sound_db grouped by `ripeness`) show near-zero class overlap, which supports this.

**What this means for how the number should be presented:** 1.00 F1 on this dataset demonstrates the pipeline works end-to-end (features → scaling → OHE → classifier → API), not that the model is production-grade on real produce. The honest framing for an interview or resume is "built and validated a full ripeness-classification pipeline on a synthetic benchmark; next step is validating against real, noisier sensor data" — not "achieved 100% accuracy."

---

## Dataset

- **Source:** Avocado Ripeness Dataset (Kaggle / synthetic sensor data)
- **Size:** 250 records — small; treat results as a pipeline proof-of-concept, not a statistically robust benchmark
- **Target:** `ripeness` — 5 classes: `hard`, `breaking`, `pre-conditioned`, `firm-ripe`, `ripe`
- **Features:** 8 — numeric (`firmness`, `hue`, `saturation`, `brightness`, `sound_db`, `weight_g`, `size_cm3`) + categorical (`color_category`, 4 values: `black`, `green`, `dark green`, `purple`)
- **Split:** `stratify=y` used to preserve per-class proportions given the multi-class, moderately imbalanced target

---

## Approach

1. **EDA** — distribution of `ripeness` classes; boxplots of `firmness`, `hue`, `saturation`, `sound_db` grouped by class (revealed the near-perfect class separability discussed above)
2. **Feature engineering** — One-Hot Encoding for `color_category` via `pd.get_dummies(..., drop_first=True)`, target column excluded from `x` before encoding (`x = df.drop(columns='ripeness')`) — no target leakage into the feature matrix
3. **Preprocessing** — `StandardScaler` fit on the train set only
4. **Model training** — Logistic Regression, Decision Tree, Random Forest, all with default hyperparameters, compared side by side
5. **Evaluation** — Accuracy, weighted Precision/Recall/F1, full `classification_report`, plus explicit train-vs-test score check per model
6. **Model persistence** — all three models and the scaler saved via `joblib`; Random Forest is the one loaded by `main.py`
7. **Deployment** — FastAPI endpoint reconstructs the One-Hot color vector from a raw string, scales, and returns the predicted class with per-class probabilities

---

## Key Challenges & Solutions

**Confirming no target leakage despite perfect scores**
Three different model families all hitting 1.00 is exactly the pattern you'd see from a leaked target — before trusting the number, checked the feature-construction line directly: `x = df.drop(columns='ripeness')` confirms `ripeness` itself was excluded from `x`, so the perfect score is a property of the (synthetic) data being cleanly separable by the sensor features, not a leak.

**API category lists drifting from the trained dataset**
`main.py`'s `VALID_COLORS`/`COLOR_CATEGORIES` (3 values) and `RIPENESS_LABELS` (3 values) don't match what the notebook actually trained on (4 color categories, 5 ripeness classes) — documented explicitly above rather than silently "fixed" by rewriting the demo response to hide it, since the mismatch itself is useful evidence of keeping the API in sync with the training pipeline as the dataset evolves.

**Small dataset (250 rows) across 5 classes**
With ~50 rows per class on average and a 20% test split, per-class test support is as low as 7–13 samples — too small to draw strong conclusions about generalization → framed results as a pipeline proof-of-concept rather than a production accuracy claim.

---

## Tech Stack

| Category   | Tools                              |
|------------|-------------------------------------|
| Language   | Python 3.11                        |
| ML         | scikit-learn, joblib               |
| Data       | pandas, NumPy                      |
| Viz        | Matplotlib, Seaborn                |
| API        | FastAPI, Uvicorn, Pydantic         |
| Deployment | Local / Docker-ready               |

---

## Deployment

The trained Random Forest model is served via **FastAPI**. On startup (`lifespan`), the model and scaler are loaded once into `app.state`.

```
POST /predict
```

The endpoint accepts 8 physical sensor features, reconstructs the One-Hot Encoded color vector internally, scales the full feature vector, and returns the predicted ripeness class along with per-class probabilities.

**To run locally:**
```bash
python main.py
# API at http://127.0.0.1:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/avocado-ripeness-api
cd avocado-ripeness-api
pip install -r requirements.txt
```

```bash
jupyter notebook AvocadoRipenessDataset/AvocadoRipenessDataset.ipynb
```

```bash
python AvocadoRipenessDataset/main.py
```

---

## Next Steps (honest roadmap)

- [ ] Sync `main.py`'s `RIPENESS_LABELS` and `COLOR_CATEGORIES` with the notebook's actual 5-class / 4-category schema
- [ ] Validate against a larger, noisier, real-world sensor dataset to get a meaningful (non-1.00) accuracy number
- [ ] Add cross-validation given the small (250-row) sample size, rather than a single train/test split
- [ ] Add a data-drift / monitoring hook before treating this as more than a prototype

---

[//]: # (## Author)

[//]: # ()
[//]: # (**[Your Name]** — [LinkedIn]&#40;https://linkedin.com&#41; | [GitHub]&#40;https://github.com&#41; | [Kaggle]&#40;https://kaggle.com&#41;)