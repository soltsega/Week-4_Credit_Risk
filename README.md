# 💳 Credit Risk Scoring Challenge: Buy-Now-Pay-Later Implementation

## 🎯 Challenge Overview

This project is part of the 10 Academy AI Mastery Week 4 Challenge, focusing on building an end-to-end credit risk model for Bati Bank's new buy-now-pay-later service. The challenge requires transforming eCommerce transaction data into actionable credit risk insights.

## 🏦 Business Context

Bati Bank is partnering with a successful eCommerce platform to offer flexible payment options. Your task is to develop a credit scoring system that evaluates customer creditworthiness using alternative data sources, specifically transaction history from the eCommerce platform.

## 🎯 Core Challenge Components

### 1\. Proxy Variable Definition

**Challenge**: The dataset lacks direct credit default labels.
**Solution Required**:

  * Implement RFM (Recency, Frequency, Monetary) analysis
  * Use K-Means clustering to segment customers
  * Create a binary `is_high_risk` label (1 = high risk, 0 = low risk)
  * Justify your clustering approach and risk threshold

### 2\. Feature Engineering

**Challenge**: Transform raw transaction data into meaningful predictors.
**Required Features**:

  * Transaction patterns (time-based, amount distributions)
  * Customer behavior metrics
  * Product category preferences
  * Channel usage patterns
  * Implement Weight of Evidence (WoE) and Information Value (IV) transformations

### 3\. Model Development

**Challenge**: Build a model that accurately predicts credit risk.
**Requirements**:

  * Implement at least two different algorithms
  * Include hyperparameter tuning
  * Handle class imbalance if present
  * Document model selection rationale

### 4\. Risk Probability & Credit Scoring

**Challenge**: Convert model outputs into actionable scores.
**Deliverables**:

  * Risk probability (0-1)
  * Credit score (300-850 scale)
  * Optimal loan amount and duration recommendations

-----

## 🏛️ Credit Scoring Business Understanding (Task 1 Deliverable)

### Basel II Accord's Influence on Model Interpretability

**Description:** The Basel II Capital Accord mandates that banks using internal models to estimate risk parameters (specifically the Probability of Default - PD) must adhere to strict validation and supervisory oversight requirements. This necessitates that the model is not a 'black box,' but rather is transparent, auditable, and well-documented. An interpretable model is crucial for regulatory compliance, ensuring the bank can provide clear, non-discriminatory reasons for credit decisions to both regulators (Pillar 2: Supervisory Review) and customers.

### Necessity of a Proxy Variable and Business Risks

**Description:** A proxy variable, derived from customer behavioral data (RFM clustering), is essential because the bank lacks historical 'default' labels for this new BNPL product. The proxy ($is\_high\_risk$) serves as the target variable for supervised machine learning, allowing the model to connect eCommerce behavior (low frequency, low monetary value) to an assumed higher financial risk profile.
**Potential Business Risks:**

  * **Credit Loss:** The primary risk is a **False Negative (Type II Error)**, where a truly high-risk customer is labeled low-risk by the proxy, leading to loan approval and subsequent default.
  * **Lost Revenue:** A **False Positive (Type I Error)**, where a creditworthy customer is labeled high-risk, results in unnecessary loan rejection and loss of potential interest income.
  * **Bias and Fairness:** The RFM proxy may inadvertently capture and amplify biases in transactional behavior, leading to unfair or unequal treatment of different customer segments.

### Trade-offs: Simple (Logistic Regression with WoE) vs. Complex (Gradient Boosting) Models

**Description:** The choice of model involves a core trade-off between predictive performance and regulatory compliance/interpretability:

| Model Type | Primary Advantage | Primary Disadvantage | Financial Context Rationale |
| :--- | :--- | :--- | :--- |
| **Simple (LogReg + WoE)** | **High Interpretability & Auditability.** Easy to generate regulatory-compliant scorecards. | Lower predictive power if relationships are highly non-linear. | Favored for the **official PD model** due to transparency required by regulators. |
| **Complex (Gradient Boosting)** | **Superior Predictive Performance.** Highly effective at capturing complex feature interactions. | **Low Interpretability** (Black Box). Requires post-hoc explainability tools (SHAP/LIME), complicating audit and customer communication. | Used as a **Challenger Model** or for internal strategy optimization, but harder to deploy as the main regulatory tool. |

-----

## 📊 Evaluation Metrics

Models will be evaluated on:

  * ROC-AUC Score (Primary Metric)
  * Precision-Recall Trade-off
  * F1 Score
  * Business Impact Analysis

## 🚀 Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # CI/CD Pipeline
├── data/                      # Data storage
│   ├── raw/                   # Original dataset
│   └── processed/             # Processed datasets
├── notebooks/
│   └── eda.ipynb              # Exploratory analysis
├── src/
│   ├── data_processing.py     # Feature engineering
│   ├── train.py               # Model training
│   ├── predict.py             # Inference
│   └── api/                   # Deployment
└── tests/                     # Unit tests
```
## Basic requirements
# Core
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- jupyter>=1.0.0

# Visualization
- matplotlib>=3.4.0
- seaborn>=0.11.0
- plotly>=5.0.0

# ML
- xgboost>=1.5.0
- imbalanced-learn>=0.8.0

## 🛠 Technical Implementation

### Data Processing Pipeline

1.  **Data Loading & Cleaning**

      - Handle missing values
      - Detect and treat outliers
      - Feature extraction from timestamps

2.  **Feature Engineering**

    ```python
    # Example RFM Calculation
    recency = (snapshot_date - last_transaction_date).days
    frequency = transaction_count / customer_active_days
    monetary = total_spend / transaction_count
    ```

3.  **Model Training**

      - Implement cross-validation
      - Hyperparameter tuning
      - Feature importance analysis

### Model Deployment

  * REST API with FastAPI
  * Input validation with Pydantic
  * Containerization with Docker
  * CI/CD with GitHub Actions

## 📅 Timeline

  * **Interim Submission**: Dec 14, 2025 (8:00 PM UTC)
  * **Final Submission**: Dec 16, 2025 (8:00 PM UTC)

## 🎯 Success Criteria

1.  **Code Quality**
      - Clean, modular code
      - Comprehensive documentation
      - Unit test coverage
2.  **Model Performance**
      - High discriminative power (AUC \> 0.8)
      - Stable performance across segments
      - Reasonable feature importance
3.  **Deployment Readiness**
      - Containerized solution
      - API documentation
      - Environment reproducibility

## 📚 Resources

  * [Basel II Capital Accord](https://www.bis.org/publ/bcbs128.pdf)
  * [Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
  * [Xente Challenge Dataset](https://www.kaggle.com/datasets/atwine/xente-challenge)
## API Usage

Run the API locally:

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Predict example (curl):

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d \
'{"features": {"feature_0": 0.5, "feature_1": 1.2, "feature_2": -0.2}}'
```

Response:

```json
{"probability": 0.7, "risk_label": 1}
```

---
## 🛠 Getting Started

1.  Clone the repository
2.  Set up environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  Run EDA notebook:
    ```bash
    jupyter notebook notebooks/eda.ipynb
    ```

## 👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

*This challenge is part of the 10 Academy AI Mastery Program - Week 4 (Dec 10-16, 2025)*


