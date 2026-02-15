#!/usr/bin/env python3
"""
Generate Visualizations - Credit Risk Project
==========================================

Create all required visualizations for the project.
Focus on working plots that display correctly.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_model_and_data():
    """Load model and data for visualization."""
    try:
        # Load model
        model_bundle = joblib.load('random_forest_weighted_balanced.joblib')
        model = model_bundle['model']
        feature_names = model_bundle['feature_names']
        class_names = model_bundle['class_names']
        
        # Load data
        df = pd.read_csv('data/processed/final_customer_data_cleaned.csv')
        
        print(f"Loaded model: {type(model).__name__}")
        print(f"Features: {len(feature_names)}")
        print(f"Classes: {class_names}")
        print(f"Data shape: {df.shape}")
        
        return model, feature_names, class_names, df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def create_feature_importance_plot(model, feature_names):
    """Create feature importance visualization."""
    print("Creating feature importance plot...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance - Credit Risk Model')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Feature importance plot saved as 'feature_importance.png'")
    return feature_importance_df

def create_partial_dependence_plot(df, feature_names):
    """Create partial dependence style visualization."""
    print("Creating partial dependence plot...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Select numeric features to analyze
    numeric_features = ['Amount', 'Value', 'PricingStrategy', 'FraudResult']
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, feature in enumerate(numeric_features[:4]):
        if i >= len(axes):
            break
        
        if feature in df.columns:
            # Create feature distribution by risk level
            ax = axes[i]
            
            # Plot distribution for each risk level
            for risk_level, color in zip(df['Risk_Label'].unique(), colors):
                subset = df[df['Risk_Label'] == risk_level]
                ax.hist(subset[feature], bins=30, alpha=0.6, label=risk_level, color=color, density=True)
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'{feature} Distribution by Risk Level')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # If feature not found, create placeholder
            axes[i].text(0.5, 0.5, f'{feature}\nNot Available', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{feature} - Data Not Available')
    
    plt.tight_layout()
    plt.savefig('partial_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Partial dependence plot saved as 'partial_dependence.png'")

def create_permutation_importance_plot(model, X_test, y_test, feature_names):
    """Create permutation importance visualization."""
    print("Creating permutation importance plot...")
    
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, 
                                         n_repeats=5, random_state=42)
    
    # Create DataFrame
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    top_perm = perm_df.head(15)
    
    plt.barh(range(len(top_perm)), top_perm['importance'])
    plt.yticks(range(len(top_perm)), top_perm['feature'])
    plt.xlabel('Permutation Importance (Decrease in Accuracy)')
    plt.title('Top 15 Permutation Importance - Credit Risk Model')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('permutation_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Permutation importance plot saved as 'permutation_importance.png'")
    return perm_df

def create_risk_distribution_plot(df):
    """Create risk distribution visualization."""
    print("Creating risk distribution plot...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # 1. Risk distribution pie chart
    risk_counts = df['Risk_Label'].value_counts()
    colors = ['red', 'orange', 'green']
    
    axes[0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.0f%%', 
                colors=colors, startangle=90)
    axes[0].set_title('Risk Distribution')
    
    # 2. Transaction amount by risk
    sns.boxplot(data=df, x='Risk_Label', y='Amount', ax=axes[1])
    axes[1].set_title('Transaction Amount by Risk Level')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 3. Risk by Provider
    provider_risk = pd.crosstab(df['ProviderId'], df['Risk_Label'])
    provider_risk.plot(kind='bar', stacked=True, ax=axes[2])
    axes[2].set_title('Risk Distribution by Provider')
    axes[2].tick_params(axis='x', rotation=45)
    
    # 4. Risk by Product Category
    category_risk = pd.crosstab(df['ProductCategory'], df['Risk_Label'])
    category_risk.plot(kind='bar', stacked=True, ax=axes[3])
    axes[3].set_title('Risk Distribution by Product Category')
    axes[3].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Risk distribution plot saved as 'risk_distribution.png'")

def create_model_performance_plot():
    """Create model performance comparison plot."""
    print("Creating model performance plot...")
    
    # Model performance data
    models = ['Random Forest\n(Weighted)', 'Logistic Regression\n(Balanced)', 'Random Forest\n(Balanced)']
    accuracy = [0.247, 0.633, 0.580]
    roc_auc = [0.713, 0.631, 0.728]
    business_cost = [2067100, 3519700, 2676000]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Accuracy comparison
    axes[0].bar(models, accuracy, color=['green', 'blue', 'orange'])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)
    
    # ROC-AUC comparison
    axes[1].bar(models, roc_auc, color=['green', 'blue', 'orange'])
    axes[1].set_title('ROC-AUC Comparison')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Business cost comparison
    axes[2].bar(models, business_cost, color=['green', 'blue', 'orange'])
    axes[2].set_title('Business Cost Comparison')
    axes[2].set_ylabel('Business Cost ($)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Model performance plot saved as 'model_performance.png'")

def main():
    """Main visualization generation pipeline."""
    print("="*60)
    print("GENERATING VISUALIZATIONS - CREDIT RISK PROJECT")
    print("="*60)
    
    # Load model and data
    model, feature_names, class_names, df = load_model_and_data()
    if model is None:
        return
    
    # Create feature importance plot
    feature_importance_df = create_feature_importance_plot(model, feature_names)
    
    # Create partial dependence plot
    create_partial_dependence_plot(df, feature_names)
    
    # Create permutation importance plot
    # Prepare data for permutation importance
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Prepare features
    numeric_features = ['Amount', 'Value', 'PricingStrategy', 'FraudResult', 'CountryCode']
    categorical_features = ['ProviderId', 'ProductCategory', 'ChannelId']
    
    X = df[numeric_features + categorical_features].copy()
    y = df['Risk_Label']
    
    # Handle categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    # Ensure we have the same features as the model
    for feature in feature_names:
        if feature not in X_encoded.columns:
            X_encoded[feature] = 0
    
    X_encoded = X_encoded[feature_names]
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    _, X_test, _, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    perm_df = create_permutation_importance_plot(model, X_test, y_test, feature_names)
    
    # Create risk distribution plot
    create_risk_distribution_plot(df)
    
    # Create model performance plot
    create_model_performance_plot()
    
    print("\n" + "="*60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*60)
    print("Generated files:")
    print("  - feature_importance.png")
    print("  - partial_dependence.png")
    print("  - permutation_importance.png")
    print("  - risk_distribution.png")
    print("  - model_performance.png")
    print("\nAll visualizations are ready for use in documentation and dashboard.")

if __name__ == "__main__":
    main()
