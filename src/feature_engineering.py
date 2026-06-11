import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering (creating domain-specific features).
    """
    print("Engineering features...")
    df_engineered = df.copy()
    
    # Example feature engineering rules:
    # 1. Total Charges per Month ratio
    if 'TotalCharges' in df_engineered.columns and 'MonthlyCharges' in df_engineered.columns:
        df_engineered['charges_ratio'] = df_engineered['TotalCharges'] / (df_engineered['MonthlyCharges'] + 1e-5)
    
    # 2. Tenure to Monthly Charges ratio
    if 'tenure' in df_engineered.columns and 'MonthlyCharges' in df_engineered.columns:
        df_engineered['tenure_charge_interaction'] = df_engineered['tenure'] * df_engineered['MonthlyCharges']
        
    return df_engineered

def get_preprocessor(categorical_cols, numerical_cols):
    """
    Returns a ColumnTransformer for preprocessing and scaling features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
        ]
    )
    return preprocessor

if __name__ == "__main__":
    print("Feature Engineering module loaded.")
