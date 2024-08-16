import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_dataset(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    # Rimozione dei duplicati
    df = df.drop_duplicates()
    
    # Gestione dei valori mancanti
    df = df.dropna(subset=['title', 'cast', 'director', 'rating', 'release_year'])
    
    # Riempimento dei valori mancanti per le colonne meno critiche
    df['cast'] = df['cast'].fillna('Unknown')
    df['director'] = df['director'].fillna('Unknown')
    
    # Pulizia della colonna Duration
    df['duration'] = df['duration'].str.replace(' min', '').astype(int)
    
    return df


def encode_categorical(df):
    # Codifica delle variabili categoriali
    label_encoders = {}
    for column in ['rating', 'director']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

def save_clean_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Determina il percorso della directory corrente
    baseDir = os.path.dirname(os.path.abspath(__file__))

    # Definisce il percorso assoluto del file CSV di input
    filepath = os.path.join(baseDir, '..', 'data', 'amazon_prime_titles.csv')

    # Definisce il percorso per il file di output
    output_path = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    
    # Esegui le funzioni di preprocessing
    df = load_dataset(filepath)
    df = clean_data(df)
    df, label_encoders = encode_categorical(df)
    save_clean_data(df, output_path)
    
    print(f"Dataset preprocessato e salvato in {output_path}")
