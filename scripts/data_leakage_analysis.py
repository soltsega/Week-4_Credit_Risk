#!/usr/bin/env python3
"""
Data Leakage Analysis - Credit Risk Project
===========================================

This script analyzes and documents the data leakage issues in the credit risk model.
The main issue is that the 'Cluster' feature contains target information.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def analyze_data_leakage():
    """Analyze data leakage in the credit risk dataset."""
    
    print("=" * 60)
    print("DATA LEAKAGE ANALYSIS - CREDIT RISK PROJECT")
    print("=" * 60)
    
    # Load the data
    df = pd.read_csv('data/processed/final_customer_data_with_risk.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Analyze cluster vs risk label relationship
    print("\n" + "="*40)
    print("CLUSTER vs RISK_LABEL ANALYSIS")
    print("="*40)
    
    # Create contingency table
    contingency = pd.crosstab(df['Cluster'], df['Risk_Label'])
    print("Contingency Table (Cluster vs Risk_Label):")
    print(contingency)
    
    # Calculate correlation
    # Convert to numeric for correlation
    risk_label_map = {'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}
    df['Risk_Label_Numeric'] = df['Risk_Label'].map(risk_label_map)
    
    correlation = df[['Cluster', 'Risk_Label_Numeric']].corr().iloc[0, 1]
    print(f"\nCorrelation between Cluster and Risk_Label: {correlation:.4f}")
    
    # Test model performance with and without cluster
    print("\n" + "="*40)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*40)
    
    # Prepare features (simplified approach)
    # Select only numeric features for quick analysis
    numeric_features = ['Amount', 'Value', 'PricingStrategy', 'FraudResult', 'CountryCode', 'Cluster']
    
    # Create feature matrix with numeric features only
    X_with_cluster = df[numeric_features].copy()
    y = df['Risk_Label_Numeric']
    
    # Features WITHOUT cluster
    X_without_cluster = X_with_cluster.drop(['Cluster'], axis=1)
    
    # Split data
    X_train_wc, X_test_wc, y_train, y_test = train_test_split(
        X_with_cluster, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_nc, X_test_nc, _, _ = train_test_split(
        X_without_cluster, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model WITH cluster
    rf_wc = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_wc.fit(X_train_wc, y_train)
    y_pred_wc = rf_wc.predict(X_test_wc)
    accuracy_wc = accuracy_score(y_test, y_pred_wc)
    
    # Model WITHOUT cluster
    rf_nc = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_nc.fit(X_train_nc, y_train)
    y_pred_nc = rf_nc.predict(X_test_nc)
    accuracy_nc = accuracy_score(y_test, y_pred_nc)
    
    print(f"Accuracy WITH Cluster: {accuracy_wc:.4f}")
    print(f"Accuracy WITHOUT Cluster: {accuracy_nc:.4f}")
    print(f"Performance Drop: {accuracy_wc - accuracy_nc:.4f}")
    
    # Feature importance analysis
    print("\n" + "="*40)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*40)
    
    feature_importance = pd.DataFrame({
        'feature': X_with_cluster.columns,
        'importance': rf_wc.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features (WITH Cluster):")
    print(feature_importance.head(10))
    
    # Check if cluster is the most important
    cluster_importance = feature_importance[feature_importance['feature'] == 'Cluster']['importance'].iloc[0]
    print(f"\nCluster Feature Importance: {cluster_importance:.4f}")
    print(f"Cluster Importance Percentage: {cluster_importance * 100:.2f}%")
    
    # Data leakage assessment
    print("\n" + "="*40)
    print("DATA LEAKAGE ASSESSMENT")
    print("="*40)
    
    leakage_score = 0
    reasons = []
    
    # Check 1: High correlation
    if abs(correlation) > 0.7:
        leakage_score += 3
        reasons.append(f"High correlation ({correlation:.3f}) between Cluster and target")
    
    # Check 2: High feature importance
    if cluster_importance > 0.3:
        leakage_score += 3
        reasons.append(f"Cluster dominates feature importance ({cluster_importance:.3f})")
    
    # Check 3: Perfect/near-perfect separation
    if accuracy_wc > 0.95:
        leakage_score += 2
        reasons.append(f"Suspiciously high accuracy with cluster ({accuracy_wc:.3f})")
    
    # Check 4: Large performance drop
    performance_drop = accuracy_wc - accuracy_nc
    if performance_drop > 0.1:
        leakage_score += 2
        reasons.append(f"Large performance drop without cluster ({performance_drop:.3f})")
    
    print(f"Data Leakage Score: {leakage_score}/10")
    print("Evidence of Data Leakage:")
    for i, reason in enumerate(reasons, 1):
        print(f"  {i}. {reason}")
    
    if leakage_score >= 7:
        print("\n🚨 CONCLUSION: CRITICAL DATA LEAKAGE DETECTED")
        print("   The Cluster feature must be removed for realistic model performance.")
    elif leakage_score >= 4:
        print("\n⚠️  CONCLUSION: MODERATE DATA LEAKAGE SUSPECTED")
        print("   The Cluster feature should be investigated further.")
    else:
        print("\n✅ CONCLUSION: LOW DATA LEAKAGE RISK")
        print("   The Cluster feature appears acceptable.")
    
    return {
        'leakage_score': leakage_score,
        'correlation': correlation,
        'cluster_importance': cluster_importance,
        'accuracy_with_cluster': accuracy_wc,
        'accuracy_without_cluster': accuracy_nc,
        'performance_drop': performance_drop,
        'reasons': reasons
    }

def create_cleaned_dataset():
    """Create a cleaned dataset without the data leakage feature."""
    
    print("\n" + "="*60)
    print("CREATING CLEANED DATASET")
    print("="*60)
    
    # Load original data
    df = pd.read_csv('data/processed/final_customer_data_with_risk.csv')
    
    # Remove cluster feature
    df_clean = df.drop(['Cluster'], axis=1)
    
    # Save cleaned dataset
    df_clean.to_csv('data/processed/final_customer_data_cleaned.csv', index=False)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Removed feature: Cluster")
    print("Cleaned dataset saved as: data/processed/final_customer_data_cleaned.csv")
    
    return df_clean

if __name__ == "__main__":
    # Run data leakage analysis
    results = analyze_data_leakage()
    
    # Create cleaned dataset
    create_cleaned_dataset()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Next steps:")
    print("1. Use the cleaned dataset for model training")
    print("2. Expect more realistic performance (70-85% accuracy)")
    print("3. Focus on business metrics and model explainability")
