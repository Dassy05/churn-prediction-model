import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
)

from src.data_preprocessing import (
    load_data, clean_data, prepare_features_and_target,
    create_preprocessor, preprocess_pipeline, apply_smote
)

def evaluate_model(model, X_test, y_test):
    """
    Calculate and return performance metrics for a model.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    return metrics, y_pred_proba

def main():
    # Paths
    raw_data_path = r"c:\Users\Lenovo\Desktop\customer_churn_prediction\data\raw\Telco-Customer-Churn.csv"
    models_dir = r"c:\Users\Lenovo\Desktop\customer_churn_prediction\models"
    artifact_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\ac38cc11-fa03-40c3-8fda-412650506452"
    
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
    
    # Save the preprocessor
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    
    # 4. Handle Imbalance (SMOTE on training set only)
    X_train_resampled, y_train_resampled = apply_smote(X_train_processed, y_train)
    
    # 5. Initialize Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    }
    
    results = {}
    proba_results = {}
    
    # 6. Train and Evaluate
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        
        # Save model
        model_filename = f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, os.path.join(models_dir, model_filename))
        print(f"Saved {name} model to {model_filename}")
        
        # Evaluate
        metrics, y_pred_proba = evaluate_model(model, X_test_processed, y_test)
        results[name] = metrics
        proba_results[name] = y_pred_proba
        
        # Print results
        print(f"Results for {name}:")
        for metric_name, val in metrics.items():
            print(f"  {metric_name}: {val:.4f}")
            
    # 7. Plot ROC Curves
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess (AUC = 0.5000)')
    
    # Premium color map
    colors = {
        'Logistic Regression': '#4A90E2', # Blue
        'Random Forest': '#50E3C2',      # Turquoise/Green
        'XGBoost': '#E25B5B'            # Muted Coral/Red
    }
    
    for name in models.keys():
        y_pred_proba = proba_results[name]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[name], lw=2.5, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves Comparison', fontsize=14, weight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Save the ROC curve plot in the artifact directory
    plot_path = os.path.join(artifact_dir, "roc_curves.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nROC Curves comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
