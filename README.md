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
*Add details about the model architecture, feature importances, and evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) once trained.*
