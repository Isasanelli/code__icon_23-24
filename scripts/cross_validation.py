import pandas as pd
import os
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Converti 'rating' in valori numerici
    le = LabelEncoder()
    df['numeric_rating'] = le.fit_transform(df['rating'])
    
    # Creazione della colonna 'most_watched' basata sulla mediana del rating numerico
    df['most_watched'] = df['numeric_rating'] > df['numeric_rating'].median()
    return df

def load_embeddings(filepath):
    return np.load(filepath)

def evaluate_model(model, X, y, cv_folds=5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    metrics = {
        'accuracy': cross_val_score(model, X, y, cv=skf, scoring='accuracy').mean(),
        'precision': cross_val_score(model, X, y, cv=skf, scoring='precision_weighted').mean(),
        'recall': cross_val_score(model, X, y, cv=skf, scoring='recall_weighted').mean(),
        'f1_score': cross_val_score(model, X, y, cv=skf, scoring='f1_weighted').mean()
    }
    
    return metrics

if __name__ == "__main__":
    # Determina il percorso della directory corrente
    baseDir = os.path.dirname(os.path.abspath(__file__))

    # Carica il dataset processato
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    df = load_processed_data(filepath)
    
    # Preprocessa i dati per creare la colonna 'most_watched'
    df = preprocess_data(df)
    
    # Gestisci i dati mancanti
    df = df.dropna(subset=['rating', 'release_year', 'duration'])
    
    # Carica gli embeddings
    embeddings_path = os.path.join(baseDir, '..', 'data', 'description_embeddings.npy')
    embeddings = load_embeddings(embeddings_path)
    
    # Verifica che le dimensioni coincidano
    if len(embeddings) != len(df):
        raise ValueError("Mismatch between embeddings and dataset rows.")
    
    # Unisci gli embeddings alle altre caratteristiche (rating, release_year, duration)
    features = np.hstack([embeddings, df[['numeric_rating', 'release_year', 'duration']].values])
    y = df['most_watched']
    
    # Valutazione del modello supervisionato (Random Forest)
    rf_model = RandomForestClassifier(random_state=42)
    rf_metrics = evaluate_model(rf_model, features, y)
    
    print("Random Forest Metrics:")
    print(f"Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"Precision: {rf_metrics['precision']:.4f}")
    print(f"Recall: {rf_metrics['recall']:.4f}")
    print(f"F1 Score: {rf_metrics['f1_score']:.4f}")
    
    # Valutazione del modello probabilistico (Naive Bayes)
    nb_model = GaussianNB()
    nb_metrics = evaluate_model(nb_model, features, y)
    
    print("\nNaive Bayes Metrics:")
    print(f"Accuracy: {nb_metrics['accuracy']:.4f}")
    print(f"Precision: {nb_metrics['precision']:.4f}")
    print(f"Recall: {nb_metrics['recall']:.4f}")
    print(f"F1 Score: {nb_metrics['f1_score']:.4f}")
    
    # Assicurati che le directory esistano
    output_dir = os.path.join(baseDir, '..', 'results', 'models', 'cross_validation')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salvataggio dei risultati
    rf_output_path = os.path.join(output_dir, 'rf_cross_validation_metrics.csv')
    nb_output_path = os.path.join(output_dir, 'nb_cross_validation_metrics.csv')
    
    pd.DataFrame([rf_metrics]).to_csv(rf_output_path, index=False)
    pd.DataFrame([nb_metrics]).to_csv(nb_output_path, index=False)
    
    print(f"Metriche della Random Forest salvate in {rf_output_path}")
    print(f"Metriche del Naive Bayes salvate in {nb_output_path}")
