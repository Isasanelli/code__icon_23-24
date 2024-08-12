from datetime import datetime
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Paths dei dataset
PRIME_DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'source', 'amazon_prime_titles.csv')
PREPROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'source', 'preprocessed')
CLEAN_PRIME_PATH = os.path.join(PREPROCESSED_DIR, 'titles_selected.csv')

# Assicurati che la directory preprocessed esista
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def extract_prime_dataset() -> pd.DataFrame:
    """Estrae il dataset dal file CSV."""
    if not os.path.exists(PRIME_DATASET_PATH):
        raise FileNotFoundError(f"Il file {PRIME_DATASET_PATH} non esiste.")
    
    df = pd.read_csv(PRIME_DATASET_PATH)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Gestisce i valori mancanti nel dataset."""
    # Rimuovi righe con valori mancanti significativi
    df.dropna(subset=['Title', 'Director', 'release_year'], inplace=True)

    # Imputazione dei valori mancanti
    df['Country'] = df['Country'].fillna('Unknown')
    df['Date_Added'] = df['Date_Added'].fillna('2000-01-01')

    return df

def adjust_string_columns(df: pd.DataFrame, except_columns=None) -> pd.DataFrame:
    """Converte le stringhe delle colonne in minuscolo, tranne quelle specificate."""
    if except_columns is None:
        except_columns = []

    for col_name, col_data in df.items():
        if pd.api.types.is_string_dtype(col_data) and col_name not in except_columns:
            df[col_name] = col_data.str.lower().str.replace(r'\W+', '_', regex=True)
    return df

def encode_categorical_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Applica one-hot encoding alle variabili categoriali."""
    encoder = OneHotEncoder(sparse_output=False)  # Usa sparse_output invece di sparse
    for col in columns:
        encoded = encoder.fit_transform(df[[col]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
        df = df.drop(col, axis=1)
        df = pd.concat([df, encoded_df], axis=1)
    return df

def remove_sparse_columns(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Rimuove colonne che hanno una percentuale di valori 0 superiore alla soglia specificata."""
    threshold_value = len(df) * threshold
    df = df.loc[:, (df != 0).sum(axis=0) > threshold_value]  # Rimuove colonne sparse
    return df

def remove_zero_rows_and_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove colonne e righe che contengono solo valori 0."""
    df = df.loc[:, (df != 0).any(axis=0)]  # Rimuove colonne con solo 0
    df = df[(df.T != 0).any()]  # Rimuove righe con solo 0
    return df

def preprocess_prime_dataset(extracted_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessa il dataset rimuovendo colonne non necessarie, gestendo i valori mancanti, e codificando le variabili categoriali."""
    # Rimozione delle colonne non necessarie
    col_del = ['show_id', 'rating', 'description', 'type']
    col_ren = {'title': 'Title', 'director': 'Director', 'release_year': 'release_year', 'country': 'Country', 'date_added': 'Date_Added'}

    extracted_df.rename(columns=col_ren, inplace=True)
    extracted_df = extracted_df.drop(columns=col_del, axis=1)

    # Rimozione dei titoli duplicati
    extracted_df.drop_duplicates(subset=['Title', 'Director'], inplace=True)

    # Gestione dei valori mancanti
    extracted_df = handle_missing_values(extracted_df)

    # Trasformazione delle stringhe in minuscolo e rimozione dei caratteri speciali
    extracted_df = adjust_string_columns(extracted_df, except_columns=["Title", "Director"])

    # Codifica delle variabili categoriali (es. Country, Genre)
    extracted_df = encode_categorical_columns(extracted_df, columns=['Country'])

    # Rimozione di colonne sparse e righe con soli valori 0
    extracted_df = remove_sparse_columns(extracted_df, threshold=0.95)
    extracted_df = remove_zero_rows_and_columns(extracted_df)

    return extracted_df

if __name__ == "__main__":
    try:
        extracted_df = extract_prime_dataset()
        clean_df = preprocess_prime_dataset(extracted_df)
        
        # Salvataggio del dataset preprocessato in una nuova directory
        clean_df.to_csv(CLEAN_PRIME_PATH, index=False)
        print(f"Preprocessamento completato. Il dataset pulito è stato salvato in '{CLEAN_PRIME_PATH}'.")
    except Exception as e:
        print(f"Errore durante l'analisi: {e}")
