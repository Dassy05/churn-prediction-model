import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def evaluate_pipeline(pipeline, X_test, y_test):
    """
    Generate predictions and evaluate performance metrics.
    """
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Classification Report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    # AUC Score
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    return report, auc, cm

def plot_confusion_matrix(cm, classes=['Non-Churn', 'Churn']):
    """
    Plot confusion matrix as a heatmap.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Evaluation module loaded.")
