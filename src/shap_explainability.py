import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    clean_data, prepare_features_and_target
)

def main():
    # Paths
    raw_data_path = r"c:\Users\Lenovo\Desktop\customer_churn_prediction\data\raw\Telco-Customer-Churn.csv"
    models_dir = r"c:\Users\Lenovo\Desktop\customer_churn_prediction\models"
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    model_path = os.path.join(models_dir, "churn_model.pkl")
    artifact_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\ac38cc11-fa03-40c3-8fda-412650506452"
    
    # 1. Load data and clean
    df = load_data_path = pd.read_csv(raw_data_path)
    df_cleaned = clean_data(df)
    X, y = prepare_features_and_target(df_cleaned)
    
    # 2. Split (80/20, stratified) to get the exact test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Load preprocessor and transform X_test
    print("Loading preprocessor and transforming test features...")
    preprocessor = joblib.load(preprocessor_path)
    X_test_arr = preprocessor.transform(X_test)
    
    # Retrieve feature names
    numeric_features = preprocessor.transformers[0][2]
    categorical_features = preprocessor.transformers[1][2]
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = list(numeric_features) + encoded_cat_names
    
    X_test_df = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)
    
    # 4. Load trained model
    print("Loading tuned XGBoost model...")
    model = joblib.load(model_path)
    
    # 5. SHAP Explainer
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_df)
    
    # Configure matplotlib style
    plt.rcParams.update({'font.size': 10, 'figure.titlesize': 14})
    
    # Plot 1: SHAP summary plot (top 10 features)
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, max_display=10, show=False)
    plt.title("SHAP Summary Plot (Top 10 Features)", weight='bold', pad=15)
    plt.tight_layout()
    summary_plot_path = os.path.join(artifact_dir, "shap_summary.png")
    plt.savefig(summary_plot_path, dpi=200)
    plt.close()
    
    # Plot 2: SHAP bar plot (mean feature importance)
    print("Generating SHAP bar plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title("SHAP Feature Importance (Bar Plot)", weight='bold', pad=15)
    plt.tight_layout()
    bar_plot_path = os.path.join(artifact_dir, "shap_bar.png")
    plt.savefig(bar_plot_path, dpi=200)
    plt.close()
    
    # Plot 3: SHAP waterfall plot for 1 sample prediction (using sample index 0)
    print("Generating SHAP waterfall plot...")
    plt.figure(figsize=(10, 6))
    # We find an interesting sample (e.g. index 0)
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.title("SHAP Waterfall Plot for Single Customer Prediction (Index 0)", weight='bold', pad=15)
    plt.tight_layout()
    waterfall_plot_path = os.path.join(artifact_dir, "shap_waterfall.png")
    plt.savefig(waterfall_plot_path, dpi=200)
    plt.close()
    
    print("\nSHAP explainability plots generated successfully!")
    print(f"Summary plot: {summary_plot_path}")
    print(f"Bar plot: {bar_plot_path}")
    print(f"Waterfall plot: {waterfall_plot_path}")

if __name__ == "__main__":
    main()
