# Credit Risk Probability Model for Alternative Data
**Author:** Solomon Tsega  
**Date:** 15/12/2025

---

## Executive summary
This project builds and deploys a credit risk probability model for Bati Bank’s Buy-Now-Pay-Later (BNPL) service using alternative eCommerce transaction data. We create a proxy target (is_high_risk) from Recency-Frequency-Monetary (RFM) clustering, engineer a reproducible feature pipeline, evaluate several predictive models (Logistic Regression, Random Forest, XGBoost), and deploy a REST API for inference with CI/CD and containerization. The recommended production approach: an interpretable Logistic Regression scorecard (regulatory-facing) with a Random Forest/GBM challenger for performance monitoring.

Key deliverables:
- EDA and feature engineering notebooks
- `credit_risk_model_v1.joblib` (model + preprocessor)
- `src/api/` (FastAPI service) with `/predict` and `/health`
- Dockerfile + docker-compose
- GitHub Actions CI (linting + tests)

---

## 1. Business objective & regulatory context
- **Business need:** Provide a risk probability that can determine BNPL eligibility and terms. The model will guide lending decisions and pricing.
- **Regulatory context (Basel II):** PD models used for credit risk estimation must be transparent, auditable, and validated. This constrains the primary production model toward interpretability and rigorous documentation.
- **Proxy label rationale:** Since the dataset lacks explicit default labels, we derive `is_high_risk` using RFM clustering; we document all assumptions and quantify proxy-related risk (
Type I/II errors, bias concerns).

---

## 2. Data summary & EDA highlights
**Data source:** Xente transaction dataset (processed into `final_customer_data_with_risk.csv`).  
**Key fields:** CustomerId, AccountId, TransactionStartTime, Amount, Value, ProductCategory, ChannelId, PricingStrategy, FraudResult, Risk_Label / is_high_risk.

Top insights (from EDA):
- Transaction amounts are strongly right-skewed; log-transform or robust scaling recommended.
- Channel usage (web/mobile/pay-later) shows varied risk profiles; pay-later channel has higher incidence of low-frequency customers.
- RFM metrics reveal separable clusters by engagement; lowest-engagement cluster aligns with proxy high-risk.

*Figures to include:* distribution plots for Amount, RFM scatterplot, missing-value heatmap. (See `notebooks/eda.ipynb`.)

---

## 3. Proxy target construction (RFM + clustering)
**Snapshot date:** set to the latest transaction date + 1 day to compute Recency.  
**RFM features:**
- Recency = days since last transaction
- Frequency = number of transactions
- Monetary = total spend

**Clustering:** StandardScaler → KMeans (k=3, random_state=42). Identify cluster with lowest Frequency & Monetary as `is_high_risk = 1`.

**Validation:** Compare cluster summaries and chosen cluster’s default proxy distribution. Sensitivity analysis: vary k and compare cluster stability.

*Figure to include:* RFM clusters labeled, cluster summaries table.

---

## 4. Feature engineering & preprocessing
**Approach:** sklearn Pipelines and ColumnTransformer for reproducibility.

**Features engineered:**
- Aggregates: total_amount, avg_amount, transaction_count, std_amount
- Temporal: last_transaction_hour, day_of_week, month
- Behavior: recency_bucket, frequency_bucket
- Categorical encodings: OneHot for low-cardinality, WoE encoding for features used in scorecards
- Missing values: median imputation for numerics; 'missing' for categoricals

**WoE & IV:** Implemented for candidate features to assess predictive power and support scorecard creation.

*Note:* Use `src/data_processing.py` for full pipeline implementation.

---

## 5. Models, training, and selection
**Models evaluated:**
- Logistic Regression (interpretable baseline)
- Random Forest (robust non-linear baseline)
- XGBoost (optional high-performance model)

**Tuning:** RandomizedSearchCV (scoring: `f1_weighted`), stratified k-fold CV.  
**Selection criteria:** Balanced between ROC-AUC, F1, calibration, and interpretability for regulatory suitability.

**Final decision:** Use Logistic Regression (WoE-transformed inputs) as the regulatory PD model; retain RandomForest/XGBoost as challenger models.

*Table — Model comparison (placeholder)*

| Model | Validation ROC-AUC | Validation F1 | Test ROC-AUC | Test F1 | Notes |
|---|---:|---:|---:|---:|---|
| Logistic Regression | **PLACEHOLDER** | **PLACEHOLDER** | **PLACEHOLDER** | **PLACEHOLDER** | Interpretable; scorecard-ready |
| Random Forest | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | Strong performance, less interpretable |
| XGBoost | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | Challenging but high performance |

*Action:* Replace placeholders after running final training (I can run it and update the file).

---

## 6. Model evaluation & calibration
**Metrics produced:** Confusion matrices, classification reports, ROC and PR curves, cross-validated F1 scores.  
**Calibration:** Use calibration plots and isotonic regression / Platt scaling if model probabilities are poorly calibrated.

*Figures to include:* ROC curves, confusion matrices for the selected model, calibration plots.

---

## 7. Deployment, API & QA
**API:** `src/api/main.py` — FastAPI app with:
- `GET /health` → returns `{ "status": "ok" }`
- `POST /predict` → accepts `{"features": {...}}` and returns `{"probability": p, "risk_label": 0/1}`

**Containerization:** `Dockerfile` and `docker-compose.yml` included; to run:
```bash
# build and run
docker-compose up --build -d api
# check logs
docker logs -f <container>
```
**CI:** `.github/workflows/ci.yml` runs flake8 and pytest on push/PR.  
**Tests:** `tests/` includes API, pipeline, and helper tests (all pass locally).

*Proof placeholders:* Insert screenshots of Docker Desktop showing the running container, `curl` outputs for `/health` and `/predict`, and GitHub Actions passing.

---

## 8. Limitations & future work
**Key limitations:**
- Proxy label (RFM) may not capture true default risk; needs validation with repayment/default data.
- Possible biases from behavioral proxies; must run fairness checks and protected-attribute analysis.
- Feature drift and data pipeline robustness require monitoring.

**Actionable next steps:**
1. Validate proxy with real loan outcomes when available; re-train the model.
2. Register model in MLflow registry, and configure API to load registered models.
3. Add production monitoring: PSI, drift detectors, and SHAP-based monitoring for feature-level alerts.
4. Add AB testing capability for acceptance policy changes.

---

## 9. Recommendations for business users
- Use an interpretable Logistic Regression scorecard for customer-facing PD decisions and compliance.  
- Employ complex models as challengers and use them for internal segmentation and pricing optimization.  
- Implement policy-level thresholds that trade off approval rates and expected loss (simulate business P&L at candidate thresholds).

---

## 10. Appendix
**Reproducibility checklist**
- Environment: `pip install -r requirements.txt` (see `requirements.txt`)  
- Run EDA: `notebooks/eda.ipynb`  
- Train & evaluate models: `notebooks/model_training.ipynb`  
- Run API locally: `uvicorn src.api.main:app --reload --port 8000`  
- Run tests: `pytest -q`

**Required visuals to attach** (in the final submission):
- EDA figures (distributions, RFM scatter)
- RFM cluster summary table
- Model comparison table and ROC curves
- Confusion matrices and calibration plots
- Screenshots: MLflow run list, Docker Desktop running the API, GitHub Actions passing CI

---

## How I can help next
I can run the final training, produce numeric metrics, generate all figures and screenshots, insert them into this report, and commit the file to the repo and open a PR. If you'd like, I can start training now and update the report with the exact numbers and images.

---

*End of report*