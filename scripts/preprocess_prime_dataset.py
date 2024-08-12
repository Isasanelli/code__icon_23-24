from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt

# Paths dei dataset
PRIME_DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'source', 'amazon_prime_titles.csv')
CLEAN_PRIME_PATH = os.path.join(os.path.dirname(__file__), '..', 'source', 'titles_selected.csv')
OUTPUT_CHARTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'charts', 'title_amazon_prime')

def extract_prime_dataset() -> pd.DataFrame:
    """Estrae il dataset dal file CSV."""
    if not os.path.exists(PRIME_DATASET_PATH):
        raise FileNotFoundError(f"Il file {PRIME_DATASET_PATH} non esiste.")
    
    df = pd.read_csv(PRIME_DATASET_PATH)
    return df

def preprocess_prime_dataset(extracted_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessa il dataset rimuovendo colonne non necessarie e duplicati."""
    # Rimozione delle colonne non necessarie
    col_del = ['show_id', 'rating', 'description', 'type']
    col_ren = {'title': 'Title', 'director': 'Director', 'release_year': 'Year', 'country': 'Country', 'date_added': 'Date_Added'}

    extracted_df.rename(columns=col_ren, inplace=True)
    extracted_df = extracted_df.drop(columns=col_del, axis=1)

    # Rimozione dei titoli duplicati
    extracted_df.drop_duplicates(subset=['Title', 'Director'], inplace=True)

    # Trasformazione delle stringhe in minuscolo e rimozione dei caratteri speciali
    extracted_df = adjust_string_columns(extracted_df, except_columns=["Title", "Director"])

    return extracted_df

def adjust_string_columns(df: pd.DataFrame, except_columns=None) -> pd.DataFrame:
    """Converte le stringhe delle colonne in minuscolo, tranne quelle specificate."""
    if except_columns is None:
        except_columns = []

    for col_name, col_data in df.items():
        if pd.api.types.is_string_dtype(col_data) and col_name not in except_columns:
            df[col_name] = col_data.str.lower().str.replace(r'\W+', '_', regex=True)
    return df

if __name__ == "__main__":
    try:
        extracted_df = extract_prime_dataset()
        clean_df = preprocess_prime_dataset(extracted_df)
        clean_df.to_csv(CLEAN_PRIME_PATH, index=False)
        print(f"Preprocessamento completato. Il dataset pulito è stato salvato in '{CLEAN_PRIME_PATH}'.")
    except Exception as e:
        print(f"Errore durante l'analisi: {e}")