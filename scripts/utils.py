import pandas as pd
import matplotlib.pyplot as plt

def save_plot(fig, filepath):
    fig.savefig(filepath)
    print(f"Plot salvato in {filepath}")

def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

def save_metrics(metrics, filepath):
    pd.DataFrame([metrics]).to_csv(filepath, index=False)
    print(f"Metrics salvate in {filepath}")
