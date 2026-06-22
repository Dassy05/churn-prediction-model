# Customer Churn Prediction Project

This project implements an end-to-end Machine Learning pipeline to predict customer churn. It includes data preprocessing, feature engineering, model training, evaluation, and a web application for serving predictions.

## Project Structure

```text
customer_churn_prediction/
├── data/                  # Data directory
│   ├── raw/               # Original raw data
│   └── processed/         # Cleaned and engineered data for modeling
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── models/                # Saved serialized model artifacts (e.g., joblib/pickle)
├── app/                   # Web application (Streamlit or FastAPI)
│   ├── __init__.py
│   └── app.py
├── requirements.txt       # Project python dependencies
└── README.md              # Project documentation template
```

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

- **Exploratory Data Analysis:** Run Jupyter Lab or Notebook:
  ```bash
  jupyter notebook
  ```
- **Train Model:** Run the training script:
  ```bash
  python src/model_training.py
  ```
- **Run Web App:** Start the Streamlit application:
  ```bash
  streamlit run app/app.py
  ```

## Model Details
### Model Architecture
The final model is a tuned **XGBoost Classifier** optimized via 5-Fold Grid Search CV using ROC-AUC as the scoring metric. To address the class imbalance in the training data, Synthetic Minority Over-sampling Technique (**SMOTE**) was applied to the training split.

### Evaluation Metrics (on Test Set)
* **Accuracy:** 78.42%
* **Precision:** 59.21%
* **Recall (Sensitivity):** 60.16%
* **F1-Score:** 59.68%
* **ROC-AUC Score:** 82.59%

### Top Feature Importances (SHAP)
Explainability analysis using SHAP (SHapley Additive exPlanations) highlights the top features driving the model's predictions:
1. **Contract Type (Month-to-month):** Customers on month-to-month contracts have a significantly higher risk of churn.
2. **Tenure (months):** Newly onboarded customers (lower tenure) have a higher propensity to churn.
3. **Monthly Charges:** High monthly charges increase the likelihood of customer churn.
4. **Tech Support & Online Security Add-ons:** Absence of these support services increases churn risk.
5. **Payment Method (Electronic Check):** Customers paying via electronic check show higher rates of churn.
