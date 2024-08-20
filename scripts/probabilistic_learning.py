import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Convertire 'rating' in valori numerici
    le = LabelEncoder()
    df['numeric_rating'] = le.fit_transform(df['rating'].fillna('Unrated'))
    
    # Creazione della colonna 'most_watched' basata su un criterio per film e serie TV
    df['most_watched'] = df.groupby('content_category')['numeric_rating'].transform(lambda x: x > x.median())
    
    return df

def train_model(X, y, baseDir):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calcolare la curva ROC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Salva la curva ROC
    output_dir_visualization = os.path.join(baseDir, 'results', 'visualizations', 'probabilistic_learning')
    os.makedirs(output_dir_visualization, exist_ok=True)
    
    roc_curve_path = os.path.join(output_dir_visualization, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close()
    
    print(f"ROC curve salvata in {roc_curve_path}")
    
    print("\nClassification Report:")
    print("="*50)
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Exclude accuracy and other summary metrics
            print(f"Class '{label}':")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1-score']:.4f}")
            print(f"    Support: {metrics['support']:.0f}")
            print("-"*50)
        else:
            print(f"{label.capitalize()}: {metrics:.4f}")
    print("="*50)
    
    return model, report

def save_results(report, baseDir):
    # Creazione della directory se non esiste
    output_dir_model = os.path.join(baseDir, 'results', 'models', 'probabilistic_learning')
    os.makedirs(output_dir_model, exist_ok=True)
    
    # Salvataggio del report in un file CSV
    report_path = os.path.join(output_dir_model, 'classification_report.csv')
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"Report salvato in {report_path}")

def probabilistic_learning(baseDir):
    """Funzione principale per eseguire l'apprendimento probabilistico."""
    filepath = os.path.join(baseDir, 'data', 'processed_data.csv')
    df = load_processed_data(filepath)
    
    df = preprocess_data(df)
    
    if 'most_watched' not in df.columns:
        raise ValueError("La colonna 'most_watched' non esiste nel dataframe.")
    
    embeddings_path = os.path.join(baseDir, 'data', 'content_category_embeddings.npy')
    embeddings = np.load(embeddings_path)
    
    # Includere il tipo di contenuto come feature
    type_encoded = pd.get_dummies(df['content_category'], drop_first=True)
    features = np.hstack([embeddings, df[['numeric_rating', 'release_year', 'title_length', 'release_month', 'release_season']].values, type_encoded.values])
    y = df['most_watched']
    # Addestra il modello e genera il report
    model, report = train_model(features, y, baseDir)
    
    # Salva i risultati
    save_results(report, baseDir)
    
    print("Modello probabilistico addestrato e valutato con successo.")

   
