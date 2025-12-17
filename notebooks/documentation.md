## Overview 🔍
- **Goal**: Train and evaluate a classifier that predicts customer risk (`Risk_Label`) using processed customer data.
- **Notebook path**: model_training.ipynb
- **Main outputs**: trained model files (`best_model.joblib`, `credit_risk_model_v1.joblib`) and data splits saved in splits.


## Data & Target 🎯
- **Input dataset**: final_customer_data_with_risk.csv
- **Target column**: `Risk_Label` (0 / 1)
- **Note**: Notebook originally planned `is_high_risk`; the actual column used is `Risk_Label`.

---

## Train / Validation / Test Splits ✂️
- There are two splitting approaches present:
  - Early split: 80% train / 10% val / 10% test (via two step split).
  - The implemented `load_and_split_data()` saves splits with **70% train / 15% val / 15% test** into splits (these are ultimately loaded and used).
- **Stratified splitting** is used to preserve class distribution.
- **Random seed**: `RANDOM_STATE = 42` (reproducible).

> Recommendation: pick one consistent split scheme (preferably documented) and confirm saved splits match the notebook expectation.

---

## Preprocessing Pipeline 🔧
- **Dropped columns**: `TransactionId`, `BatchId`, `TransactionStartTime`.
- **Numeric columns** (example in notebook):
  - `['Amount', 'Value', 'PricingStrategy', 'FraudResult', 'Cluster', 'CountryCode']`
- **Categorical columns** (example in notebook):
  - `['AccountId', 'SubscriptionId', 'CustomerId', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']`
- **Numeric processing**:
  - `SimpleImputer(strategy='median')`
  - `StandardScaler()`
- **Categorical processing**:
  - `SimpleImputer(strategy='constant', fill_value='missing')`
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=True)`
- Implementation uses `ColumnTransformer` and `Pipeline`. Output is typically a sparse matrix; code converts to dense before stacking when needed.

---

## Models & Baselines 🧪
- **Baseline models trained**:
  - Logistic Regression (`sklearn.linear_model.LogisticRegression`, `max_iter=1000`)
  - Random Forest (`sklearn.ensemble.RandomForestClassifier`, `n_estimators=100`, `class_weight='balanced'`)
- **Evaluation approach**:
  - Predictions on validation set
  - Metrics: classification report (precision/recall/F1), confusion matrix
  - Cross-validation: `cross_val_score` with `scoring='f1_weighted'`
- **MLflow** is imported and available but not fully integrated across all training steps (could be added to log params/metrics/artifacts).

---

## Hyperparameter Tuning ⚙️
- **Method**: `RandomizedSearchCV` (example search)
- **Parameter distribution**:
  - `n_estimators`: randint(50, 150)
  - `max_depth`: [None, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `max_features`: ['sqrt', 'log2']
  - `class_weight`: ['balanced']
- **Search settings**: `n_iter=10`, `cv=3`, `scoring='f1_weighted'`, `n_jobs=-1`
- **Best model** obtained by random search (`best_rf`) and validated on validation set.

---

## Final Model Training & Saving 💾
- Combine training + validation (convert sparse -> dense when necessary) to form `X_full` and `y_full`.
- Final model used in the notebook:
  - `RandomForestClassifier(n_estimators=132, max_depth=None, min_samples_split=5, min_samples_leaf=4, max_features='sqrt', class_weight='balanced', random_state=42)`
- **Saved artifacts**:
  - `best_model.joblib` (earlier selection)
  - `credit_risk_model_v1.joblib` (final artifact) — contains:
    - `model` (final RandomForest)
    - `preprocessor` (ColumnTransformer pipeline)
    - `label_encoder` (`le`)
    - `class_names` (`le.classes_`)
    - `feature_names` (extracted from preprocessor where available)
- Example load snippet:
  ```python
  import joblib
  model_bundle = joblib.load('credit_risk_model_v1.joblib')
  model = model_bundle['model']
  preprocessor = model_bundle['preprocessor']
  ```

---

## Evaluation Metrics & Visuals 📊
- **Validation/Test metrics printed**:
  - `classification_report` (precision, recall, F1)
  - `accuracy_score` printed for test set
  - `confusion_matrix` plotted via Seaborn heatmap
- **Cross-validation**: `f1_weighted` scores reported during model evaluation
- (Note: `roc_auc_score` is imported but not used; consider adding ROC and AUC plots for class imbalance insight)

---

## Reproducibility & Environment 🧾
- **Random seed**: `42`
- **Primary packages used**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `mlflow`, `xgboost`, `joblib`, `imbalanced-learn` (see requirements.txt)
- Ensure requirements.txt includes all required packages and versions.
