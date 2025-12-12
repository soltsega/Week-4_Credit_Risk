# Credit Risk Scoring Challenge: Buy-Now-Pay-Later Implementation

## 🎯 Challenge Overview
This project is part of the 10 Academy AI Mastery Week 4 Challenge, focusing on building an end-to-end credit risk model for Bati Bank's new buy-now-pay-later service. The challenge requires transforming eCommerce transaction data into actionable credit risk insights.

## 🏦 Business Context
Bati Bank is partnering with a successful eCommerce platform to offer flexible payment options. Your task is to develop a credit scoring system that evaluates customer creditworthiness using alternative data sources, specifically transaction history from the eCommerce platform.

## 🎯 Core Challenge Components

### 1. Proxy Variable Definition
**Challenge**: The dataset lacks direct credit default labels.  
**Solution Required**:
- Implement RFM (Recency, Frequency, Monetary) analysis
- Use K-Means clustering to segment customers
- Create a binary `is_high_risk` label (1 = high risk, 0 = low risk)
- Justify your clustering approach and risk threshold

### 2. Feature Engineering
**Challenge**: Transform raw transaction data into meaningful predictors.  
**Required Features**:
- Transaction patterns (time-based, amount distributions)
- Customer behavior metrics
- Product category preferences
- Channel usage patterns
- Implement Weight of Evidence (WoE) and Information Value (IV) transformations

### 3. Model Development
**Challenge**: Build a model that accurately predicts credit risk.  
**Requirements**:
- Implement at least two different algorithms
- Include hyperparameter tuning
- Handle class imbalance if present
- Document model selection rationale

### 4. Risk Probability & Credit Scoring
**Challenge**: Convert model outputs into actionable scores.  
**Deliverables**:
- Risk probability (0-1)
- Credit score (300-850 scale)
- Optimal loan amount and duration recommendations

## 📊 Evaluation Metrics
Models will be evaluated on:
- ROC-AUC Score (Primary Metric)
- Precision-Recall Trade-off
- F1 Score
- Business Impact Analysis

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

## 🛠 Technical Implementation

### Data Processing Pipeline
1. **Data Loading & Cleaning**
   - Handle missing values
   - Detect and treat outliers
   - Feature extraction from timestamps

2. **Feature Engineering**
   ```python
   # Example RFM Calculation
   recency = (snapshot_date - last_transaction_date).days
   frequency = transaction_count / customer_active_days
   monetary = total_spend / transaction_count
   ```

3. **Model Training**
   - Implement cross-validation
   - Hyperparameter tuning
   - Feature importance analysis

### Model Deployment
- REST API with FastAPI
- Input validation with Pydantic
- Containerization with Docker

## 📅 Timeline
- **Interim Submission**: Dec 14, 2025 (8:00 PM UTC)
- **Final Submission**: Dec 16, 2025 (8:00 PM UTC)

## 🎯 Success Criteria
1. **Code Quality**
   - Clean, modular code
   - Comprehensive documentation
   - Unit test coverage

2. **Model Performance**
   - High discriminative power (AUC > 0.8)
   - Stable performance across segments
   - Reasonable feature importance

3. **Deployment Readiness**
   - Containerized solution
   - API documentation
   - Environment reproducibility

## 📚 Resources
- [Basel II Capital Accord](https://www.bis.org/publ/bcbs128.pdf)
- [Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Xente Challenge Dataset](https://www.kaggle.com/datasets/atwine/xente-challenge)

## 🛠 Getting Started
1. Clone the repository
2. Set up environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run EDA notebook:
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

## Solomon Tsega

## 📝 License
- MIT License


*This challenge is part of the 10 Academy AI Mastery Program - Week 4 (Dec 10-16, 2025)*
