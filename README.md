# Credit Scoring — Home Credit Default Risk

[🇷🇺 Читать на русском](README_RU.md)

An educational Data Science project focused on binary classification using Logistic Regression. The goal is to build a credit scoring model that predicts the probability of a borrower defaulting on a loan, and to interpret the results in business terms.

This project follows the structure and philosophy of my previous project, [hot-jupiter-detector](https://github.com/DigitalOcean78/hot-jupiter-detector): a single, well-documented notebook, clean code, and a strong emphasis on model interpretation.

## Problem Statement

**Task:** Binary classification.

**Target variable:** `TARGET`
- `0` — The client repaid the loan (282,686 clients, 91.93%)
- `1` — The client had payment difficulties / default (24,825 clients, 8.07%)

**Goal:** Predict the probability of default `P(TARGET = 1 | X)` based on application data.

**Metrics:**
- **ROC-AUC** and **Gini** (= 2·AUC − 1) — primary metrics in credit scoring.
- **KS-statistic** — industry standard for score separation.
- **PR-AUC** — informative due to severe class imbalance.
- **Brier score** — measures the quality of probability calibration.

**Key Challenges:**
- Severe class imbalance (~11.4 : 1).
- Asymmetric cost of errors: a missed default (False Negative) is much more expensive than a false alarm (False Positive).
- High requirement for model interpretability (regulatory compliance).

## Data

**Source:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) competition on Kaggle.

**File used:** `application_train.csv` — loan applications (one row per client). Other competition files (bureau, previous_application, etc.) are intentionally excluded to focus on modeling and interpretation rather than complex joins.

**Size:** 307,511 rows × 122 features.

**Feature Types:** 65 `float64`, 41 `int64`, 16 `object` (categorical).

**Known Anomalies & Data Quirks:**
- `DAYS_EMPLOYED = 365243` — a marker for "unemployed/pensioner", not a real date. Requires replacement and a binary flag.
- `CODE_GENDER = 'XNA'` — a few records (4).
- Structural missing values: the housing block (~47 columns) is entirely missing for 47.4% of clients and fully populated for 25.6%.
- High cardinality: `ORGANIZATION_TYPE` (~58) and `OCCUPATION_TYPE` (~31) will produce many columns via One-Hot Encoding.

## Methodology

1. **EDA:**
   - Analyzed missing values (identified structural missingness in the housing block).
   - Visualized distributions of key numerical features vs. TARGET (`EXT_SOURCE_2/3` and `DAYS_BIRTH` showed clear separation).
   - Analyzed default rates across categorical features (`NAME_EDUCATION_TYPE` showed a massive spread from 2% to 11%).

2. **Preprocessing:**
   - Dropped 48 "garbage" columns (housing block with >30% missing + `OWN_CAR_AGE`).
   - Handled the `DAYS_EMPLOYED` anomaly (replaced with `NaN` and created `IS_UNEMPLOYED` flag).
   - Imputed missing values (median for numerical, `'Unknown'` for categorical). Replacing missing `OCCUPATION_TYPE` with `'Unknown'` proved highly effective.
   - Applied One-Hot Encoding to 12 categorical columns (resulting in 175 features).
   - Split data into Train/Test (80/20) with stratification to preserve the 8.07% default rate.
   - Scaled features using `StandardScaler` (fitted only on the training set to prevent data leakage).

3. **Modeling:**
   - Trained `LogisticRegression` with `class_weight='balanced'` to handle the class imbalance.

4. **Interpretation:**
   - Extracted coefficients and converted them to Odds Ratios.
   - Identified key risk drivers (`AMT_CREDIT`) and protective factors (`EXT_SOURCE_1/2/3`, `AMT_GOODS_PRICE`).

5. **Calibration:**
   - Applied Isotonic Regression (`CalibratedClassifierCV` with `FrozenEstimator`) to fix the overconfident probabilities caused by `class_weight='balanced'`.

## Results (Test Set)

| Metric | Value | Comment |
|--------|-------|---------|
| ROC-AUC | 0.7485 | Strong baseline for Logistic Regression |
| Gini | 0.4971 | >0.4 is considered strong in scoring |
| KS | 0.3689 | >0.3 is an industry standard for a "good" model |
| PR-AUC | 0.2288 | ~2.8x better than random baseline (0.0807) |
| Brier (calibrated) | 0.0685 | Probabilities are calibrated and business-ready |

## Key Insights

1. **Top predictors:** `EXT_SOURCE_1/2/3` (external scoring points), `AMT_CREDIT` (loan amount), and `AMT_GOODS_PRICE` (goods price). The model independently identified the same strong features observed during EDA.
2. **Unknown as signal:** Imputing `OCCUPATION_TYPE` with `'Unknown'` was the right call. The feature `OCCUPATION_TYPE_Unknown` emerged in the top-20 coefficients, proving that the absence of data correlates with risk.
3. **Calibration is critical:** Raw probabilities from the balanced model had a Brier score of 0.2026. After isotonic calibration, it dropped to 0.0685, making the probabilities suitable for expected loss calculations (PD × LGD × EAD).

## Limitations & Future Improvements

- **No feature engineering:** The baseline uses only `application_train.csv`. Joining with `bureau.csv` and `previous_application.csv` typically adds +0.02 to +0.04 AUC.
- **No WOE binning:** Weight of Evidence binning is the industry standard for credit scoring. It improves interpretability and sometimes model performance.
- **Threshold optimization:** The default 0.5 threshold is used. In a real business context, the threshold should be chosen based on the cost matrix (FN vs. FP).
- **No out-of-time validation:** There are no dates in the data to perform a true temporal validation.

## Project Structure

credit-scoring/
├── data/
│   └── application_train.csv
├── notebooks/
│   └── credit_scoring.ipynb
├── requirements.txt
├── README.md
└── README_RU.md

## How to Run

1. Clone the repository.
2. Download `application_train.csv` from the Kaggle competition page and place it in the `data/` directory.
3. Install dependencies: `pip install -r requirements.txt`
4. Open and run the notebook: `jupyter notebook notebooks/credit_scoring.ipynb`