import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.data_preprocessing import (
    load_data, clean_data, prepare_features_and_target,
    create_preprocessor, preprocess_pipeline, apply_smote
)

def evaluate_metrics(model, X_test, y_test):
    """
    Calculate and return performance metrics for a model.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }

def main():
    # Paths
    raw_data_path = r"c:\Users\Lenovo\Desktop\customer_churn_prediction\data\raw\Telco-Customer-Churn.csv"
    models_dir = r"c:\Users\Lenovo\Desktop\customer_churn_prediction\models"
    
    # 1. Load and clean
    df = load_data(raw_data_path)
    df_cleaned = clean_data(df)
    X, y = prepare_features_and_target(df_cleaned)
    
    # Define features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    # 2. Split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Preprocess
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    X_train_processed, X_test_processed = preprocess_pipeline(X_train, X_test, preprocessor)
    
    # 4. Handle Imbalance (SMOTE on training set only)
    X_train_resampled, y_train_resampled = apply_smote(X_train_processed, y_train)
    
    # 5. Baseline (Before Tuning)
    print("\nTraining baseline XGBoost model...")
    baseline_model = XGBClassifier(random_state=42, eval_metric='logloss')
    baseline_model.fit(X_train_resampled, y_train_resampled)
    baseline_metrics = evaluate_metrics(baseline_model, X_test_processed, y_test)
    
    # 6. Grid Search CV Setup
    print("\nStarting Grid Search CV (5-Fold)...")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    
    # Using ROC-AUC as the scoring metric for tuning
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print("\nBest Parameters found:")
    for param, val in best_params.items():
        print(f"  {param}: {val}")
        
    # 7. Evaluate Tuned Model (After Tuning)
    tuned_metrics = evaluate_metrics(best_model, X_test_processed, y_test)
    
    # 8. Compare Performance
    print("\n=== Performance Comparison (Before vs After Tuning) ===")
    comparison_df = pd.DataFrame({
        'Metric': list(baseline_metrics.keys()),
        'Before Tuning (Baseline)': [f"{val:.4f}" for val in baseline_metrics.values()],
        'After Tuning': [f"{val:.4f}" for val in tuned_metrics.values()]
    })
    print(comparison_df.to_string(index=False))
    
    # 9. Save Tuned Model as models/churn_model.pkl
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "churn_model.pkl")
    joblib.dump(best_model, model_save_path)
    print(f"\nFinal tuned model saved to: {model_save_path}")

if __name__ == "__main__":
    main()
