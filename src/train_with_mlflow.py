import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")
experiment_name = "credit_risk_modeling"
mlflow.set_experiment(experiment_name)

def load_data():
    """Load preprocessed data from files"""
    try:
        # Update these paths according to your actual file structure
        data_dir = os.path.join("data", "processed")
        X_train = pd.read_csv(os.path.join(data_dir, "X_train_processed.csv"))
        X_val = pd.read_csv(os.path.join(data_dir, "X_val_processed.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train_encoded.csv"), header=None, squeeze=True)
        y_val = pd.read_csv(os.path.join(data_dir, "y_val_encoded.csv"), header=None, squeeze=True)
        
        print(f"Loaded data shapes - X_train: {X_train.shape}, X_val: {X_val.shape}")
        return X_train, X_val, y_train, y_val
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def train_and_log_model(X_train, X_val, y_train, y_val, model_params, run_name=None):
    """Train and log model with MLflow"""
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(model_params)
        
        # Train model
        model = RandomForestClassifier(**model_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run {run_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return model

if __name__ == "__main__":
    # Load preprocessed data
    print("Loading data...")
    X_train_processed, X_val_processed, y_train_encoded, y_val_encoded = load_data()
    
    # Define model parameters to test
    models_to_try = {
        "random_forest_100": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "class_weight": "balanced"
        },
        "random_forest_200": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "class_weight": "balanced"
        }
    }
    
    # Train and log each model
    for model_name, params in models_to_try.items():
        print(f"\nTraining {model_name}...")
        model = train_and_log_model(
            X_train_processed,
            X_val_processed,
            y_train_encoded,
            y_val_encoded,
            params,
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    print("\nTraining complete! View results with: mlflow ui")