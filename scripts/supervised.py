import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Mappare i valori di rating su numeri
    rating_mapping = {
        'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 'NC-17': 5,
        'Unrated': np.nan, '13+': 3, '16+': 4, '18+': 5, '7+': 2, 'ALL': 1
    }
    
    # Applicare la mappatura
    df['numeric_rating'] = df['rating'].map(rating_mapping)
    
    # Eliminare righe con rating non mappabile o durata mancante
    df = df.dropna(subset=['numeric_rating', 'duration'])
    
    # Creazione della colonna 'most_watched' basata su un criterio per film e serie TV
    df['most_watched'] = df.groupby('type')['numeric_rating'].transform(lambda x: x > x.median())
    
    return df

def filter_embeddings(df, embeddings):
    # Filtra gli embeddings per corrispondere al DataFrame dopo la pulizia
    if len(embeddings) > len(df):
        embeddings = embeddings[df.index]
    return embeddings

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)
    
    return model, report

def save_results(report, output_dir):
    # Creazione della directory se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salvataggio del report in un file CSV
    report_path = os.path.join(output_dir, 'classification_report.csv')
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"Report salvato in {report_path}")

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))

    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    df = load_processed_data(filepath)
    
    df = preprocess_data(df)
    
    if 'most_watched' not in df.columns:
        raise ValueError("La colonna 'most_watched' non esiste nel dataframe.")
    
    embeddings_path = os.path.join(baseDir, '..', 'data', 'description_embeddings.npy')
    embeddings = np.load(embeddings_path)
    
    # Filtraggio degli embeddings per corrispondere al DataFrame pulito
    embeddings = filter_embeddings(df, embeddings)
    
    features = np.hstack([embeddings, df[['numeric_rating', 'release_year', 'duration']].values])
    y = df['most_watched']
    
    model, report = train_model(features, y)
    
    # Definisci il percorso di output per salvare i risultati
    output_dir = os.path.join(baseDir, '..', 'results', 'models', 'supervised')
    save_results(report, output_dir)
    
    print("Modello supervisionato addestrato e valutato")
