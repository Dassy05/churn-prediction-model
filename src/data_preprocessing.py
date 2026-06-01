import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw churn data from a CSV file.
    """
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial cleaning, handle missing values, and convert types.
    """
    print("Cleaning data...")
    df_cleaned = df.copy()
    
    # TotalCharges in Telco Churn often has empty spaces; replace with NaN and fill with median
    if 'TotalCharges' in df_cleaned.columns:
        df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'].replace(' ', np.nan), errors='coerce')
        # Fill missing TotalCharges with median
        median_total_charges = df_cleaned['TotalCharges'].median()
        df_cleaned['TotalCharges'] = df_cleaned['TotalCharges'].fillna(median_total_charges)
        
    return df_cleaned

def prepare_features_and_target(df: pd.DataFrame):
    """
    Separate target variable (Churn) and features, dropping customerID.
    """
    print("Preparing features and target...")
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return X, y

def create_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for scaling and one-hot encoding.
    """
    print("Creating ColumnTransformer preprocessor...")
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='if_binary', sparse_output=False), categorical_features)
        ]
    )

def preprocess_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, preprocessor: ColumnTransformer):
    """
    Fit and transform the training data, and transform the test data.
    Returns processed DataFrames with proper feature names.
    """
    print("Fitting and transforming features...")
    # Fit & Transform
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)
    
    # Retrieve feature names
    numeric_features = preprocessor.transformers[0][2]
    categorical_features = preprocessor.transformers[1][2]
    cat_encoder = preprocessor.named_transformers_['cat']
    
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = list(numeric_features) + encoded_cat_names
    
    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)
    
    return X_train_df, X_test_df

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Apply SMOTE to balance the training set classes.
    """
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def save_artifact(obj, filepath: str):
    """
    Save the serialized python object (e.g. preprocessor).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Artifact saved successfully to {filepath}")

if __name__ == "__main__":
    # Test script run
    raw_data_path = r"c:\Users\Lenovo\Desktop\customer_churn_prediction\data\raw\Telco-Customer-Churn.csv"
    models_dir = r"c:\Users\Lenovo\Desktop\customer_churn_prediction\models"
    
    df = load_data(raw_data_path)
    df_cleaned = clean_data(df)
    X, y = prepare_features_and_target(df_cleaned)
    
    # Define numeric and categorical feature names
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Preprocess
    X_train_processed, X_test_processed = preprocess_pipeline(X_train, X_test, preprocessor)
    
    # Save the preprocessor for future inference
    save_artifact(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    
    # Show distribution before SMOTE
    print("\n=== Class Distribution BEFORE SMOTE ===")
    print("Counts:")
    print(y_train.value_counts())
    print("\nProportions (%):")
    print(y_train.value_counts(normalize=True) * 100)
    
    # SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train_processed, y_train)
    
    # Show distribution after SMOTE
    print("\n=== Class Distribution AFTER SMOTE ===")
    print("Counts:")
    print(y_train_resampled.value_counts())
    print("\nProportions (%):")
    print(y_train_resampled.value_counts(normalize=True) * 100)
    
    print(f"\nResampled Training Set Shape: {X_train_resampled.shape}")
