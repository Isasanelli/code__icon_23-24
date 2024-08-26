import pandas as pd
import os
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    le = LabelEncoder()
    df['numeric_rating'] = le.fit_transform(df['rating'])
    df['most_watched'] = df['numeric_rating'] > df['numeric_rating'].median()
    return df

def load_embeddings(filepath):
    return np.load(filepath)

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title('Confusion Matrix')
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Confusion matrix saved in {plot_path}")

def plot_roc_curve(y_true, y_prob, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plot_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"ROC curve saved in {plot_path}")

def plot_prediction_distribution(y_true, y_pred, output_dir):
    plt.figure(figsize=(8, 6))
    sns.histplot(y_true, color='blue', alpha=0.5, label='True Labels', kde=True)
    sns.histplot(y_pred, color='red', alpha=0.5, label='Predicted Labels', kde=True)
    plt.title('Prediction Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.legend()

    plot_path = os.path.join(output_dir, 'prediction_distribution.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Prediction distribution plot saved in {plot_path}")


def cross_validate_models(baseDir):
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    df = load_processed_data(filepath)
    df = preprocess_data(df)
    
    columns_to_check = ['rating', 'release_year']
    columns_to_dropna = [col for col in columns_to_check if col in df.columns]
    
    if columns_to_dropna:
        df = df.dropna(subset=columns_to_dropna)
    
    embeddings_path = os.path.join(baseDir, '..', 'data', 'content_category_embeddings.npy')
    embeddings = load_embeddings(embeddings_path)
    
    if len(embeddings) != len(df):
        raise ValueError("Mismatch between embeddings and dataset rows.")
    
    features = np.hstack([embeddings, df[['numeric_rating', 'release_year']].values])
    y = df['most_watched']
    
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None, max_features='sqrt')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(rf_model, features, y, cv=skf)
    y_prob = cross_val_predict(rf_model, features, y, cv=skf, method='predict_proba')[:, 1]
    
    output_dir_visualizations = os.path.join(baseDir, '..', 'results', 'visualizations', 'cross_validation')
    os.makedirs(output_dir_visualizations, exist_ok=True)
    
    plot_confusion_matrix(y, y_pred, output_dir_visualizations)
    plot_roc_curve(y, y_prob, output_dir_visualizations)
    plot_prediction_distribution(y, y_pred, output_dir_visualizations)

