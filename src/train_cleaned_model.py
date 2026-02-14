#!/usr/bin/env python3
"""
Cleaned Model Training - Credit Risk Project
============================================

This script trains models on the cleaned dataset without data leakage.
Focus on realistic performance and business metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc
)
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data():
    """Load the cleaned dataset without data leakage."""
    try:
        df = pd.read_csv('data/processed/final_customer_data_cleaned.csv')
        print(f"Loaded cleaned dataset: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: Cleaned dataset not found. Run data_leakage_analysis.py first.")
        return None

def prepare_features(df):
    """Prepare features for model training."""
    print("Preparing features...")
    
    # Select relevant features
    numeric_features = ['Amount', 'Value', 'PricingStrategy', 'FraudResult', 'CountryCode']
    categorical_features = ['ProviderId', 'ProductCategory', 'ChannelId']
    
    # Create feature matrix
    features = numeric_features + categorical_features
    X = df[features].copy()
    y = df['Risk_Label']
    
    # Handle categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Feature matrix shape: {X_encoded.shape}")
    print(f"Target distribution: {np.bincount(y_encoded)}")
    print(f"Class names: {le.classes_}")
    
    return X_encoded, y_encoded, le

def train_models(X, y):
    """Train multiple models with cross-validation."""
    print("\nTraining models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # For multi-class ROC-AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        except:
            roc_auc = 0.0
        
        # Cross-validation
        if name == 'Logistic Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'scaler': scaler if name == 'Logistic Regression' else None
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results, X_test, y_test

def analyze_class_performance(y_true, y_pred, class_names):
    """Analyze performance by class."""
    print("\nClass-wise Performance Analysis:")
    print("="*50)
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    for class_name in class_names:
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = report[class_name]['support']
        
        print(f"{class_name}:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  Support: {support}")
        print()
    
    # Identify worst performing class
    worst_class = min(class_names, key=lambda x: report[x]['f1-score'])
    worst_f1 = report[worst_class]['f1-score']
    
    print(f"Worst performing class: {worst_class} (F1: {worst_f1:.3f})")
    
    return report

def calculate_business_metrics(y_true, y_pred, class_names):
    """Calculate business-relevant metrics."""
    print("\nBusiness Metrics Analysis:")
    print("="*50)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Define costs (example values)
    # False Negative: Approving high-risk customer (high cost)
    # False Positive: Rejecting low-risk customer (opportunity cost)
    cost_fn = 1000  # Cost of default
    cost_fp = 100   # Opportunity cost
    
    total_cost = 0
    cost_breakdown = {}
    
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            count = cm[i, j]
            if i != j:  # Misclassification
                if true_class == 'High Risk' and pred_class != 'High Risk':
                    # False Negative: High risk predicted as lower risk
                    cost = count * cost_fn
                    cost_breakdown[f'FN_{true_class}'] = cost
                    total_cost += cost
                elif true_class != 'High Risk' and pred_class == 'High Risk':
                    # False Positive: Low/Medium risk predicted as high risk
                    cost = count * cost_fp
                    cost_breakdown[f'FP_{true_class}'] = cost
                    total_cost += cost
    
    print(f"Total Business Cost: ${total_cost:,.0f}")
    print("Cost Breakdown:")
    for cost_type, cost in cost_breakdown.items():
        print(f"  {cost_type}: ${cost:,.0f}")
    
    # Calculate potential savings with perfect model
    perfect_cost = 0
    for i, true_class in enumerate(class_names):
        if true_class == 'High Risk':
            perfect_cost += cm[i, i] * 0  # Perfect classification: no cost
    
    potential_savings = total_cost - perfect_cost
    print(f"Potential Savings with Perfect Model: ${potential_savings:,.0f}")
    
    return {
        'total_cost': total_cost,
        'cost_breakdown': cost_breakdown,
        'potential_savings': potential_savings
    }

def log_to_mlflow(results, X_test, y_test, class_names):
    """Log results to MLflow."""
    print("\nLogging to MLflow...")
    
    mlflow.set_experiment("credit_risk_cleaned_modeling")
    
    for model_name, result in results.items():
        with mlflow.start_run(run_name=f"{model_name}_cleaned"):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("dataset", "cleaned_no_leakage")
            mlflow.log_param("feature_count", X_test.shape[1])
            
            # Log metrics
            mlflow.log_metric("accuracy", result['accuracy'])
            mlflow.log_metric("roc_auc", result['roc_auc'])
            mlflow.log_metric("cv_mean", result['cv_mean'])
            mlflow.log_metric("cv_std", result['cv_std'])
            
            # Log class-wise metrics
            report = classification_report(y_test, result['predictions'], 
                                        target_names=class_names, output_dict=True)
            for class_name in class_names:
                mlflow.log_metric(f"{class_name}_precision", report[class_name]['precision'])
                mlflow.log_metric(f"{class_name}_recall", report[class_name]['recall'])
                mlflow.log_metric(f"{class_name}_f1", report[class_name]['f1-score'])
            
            # Log model
            model_bundle = {
                'model': result['model'],
                'scaler': result['scaler'],
                'feature_names': list(X_test.columns),
                'class_names': class_names
            }
            mlflow.sklearn.log_model(result['model'], "model")
            joblib.dump(model_bundle, f"{model_name.lower().replace(' ', '_')}_cleaned.joblib")
            mlflow.log_artifact(f"{model_name.lower().replace(' ', '_')}_cleaned.joblib")
            
            print(f"Logged {model_name} to MLflow")

def main():
    """Main training pipeline."""
    print("="*60)
    print("CLEANED MODEL TRAINING - CREDIT RISK PROJECT")
    print("="*60)
    
    # Load data
    df = load_cleaned_data()
    if df is None:
        return
    
    # Prepare features
    X, y, label_encoder = prepare_features(df)
    class_names = label_encoder.classes_
    
    # Train models
    results, X_test, y_test = train_models(X, y)
    
    # Analyze performance
    for model_name, result in results.items():
        print(f"\n{model_name} Detailed Analysis:")
        print("="*40)
        
        # Class-wise performance
        analyze_class_performance(y_test, result['predictions'], class_names)
        
        # Business metrics
        business_metrics = calculate_business_metrics(
            y_test, result['predictions'], class_names
        )
        
        # Store business metrics
        result['business_metrics'] = business_metrics
    
    # Log to MLflow
    log_to_mlflow(results, X_test, y_test, class_names)
    
    # Save label encoder
    joblib.dump(label_encoder, 'label_encoder_cleaned.joblib')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("Models saved:")
    for model_name in results.keys():
        print(f"  - {model_name.lower().replace(' ', '_')}_cleaned.joblib")
    print("\nNext steps:")
    print("1. Review MLflow experiment results")
    print("2. Implement class imbalance solutions")
    print("3. Add model explainability (SHAP)")
    print("4. Create interactive dashboard")

if __name__ == "__main__":
    main()
