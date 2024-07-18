from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt

# Paths dei dataset
PRIME_DATASET_PATH = "../source/amazon_prime_titles.csv"
CLEAN_PRIME_PATH = "../source/titles_selected.csv"
OUTPUT_CHARTS_PATH = "../charts"

# Estrae e pre-processa il dataset
def extract_prime_dataset() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(PRIME_DATASET_PATH)
    return df

def preprocess_prime_dataset(extracted_df: pd.DataFrame) -> pd.DataFrame:
    # Rimozione delle colonne non necessarie
    col_del = ['show_id', 'type', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']

    # Rinomina colonne
    col_ren = {'title': 'Title', 'director': 'Director'}

    extracted_df.rename(columns=col_ren, inplace=True)
    extracted_df = extracted_df.drop(col_del, axis=1)

    # Trasformazione delle stringhe in minuscolo
    extracted_df = adjust_string_columns(extracted_df, except_columns=["Title", "Director"])

    return extracted_df

def adjust_string_columns(df: pd.DataFrame, except_columns=None) -> pd.DataFrame:
    if except_columns is None:
        except_columns = []

    for col_name, col_data in df.items():
        if pd.api.types.is_string_dtype(col_data) and (col_name not in except_columns):
            df[col_name] = col_data.str.lower().apply(lambda x: ''.join(['_' if not c.isalnum() else c for c in x]))
    return df

def generate_charts(df: pd.DataFrame):
    os.makedirs(OUTPUT_CHARTS_PATH, exist_ok=True)

    # Grafico a barre del numero di titoli per regista
    top_directors = df['Director'].value_counts()
    top_directors = top_directors[top_directors.index != '1'].head(10)  # Escludi '1' dalla lista
    plt.figure(figsize=(12, 8))
    bars = plt.bar(top_directors.index, top_directors.values, edgecolor='black', color='skyblue')
    plt.title('Top 10 Registi con Più Titoli su Amazon Prime', fontsize=16)
    plt.xlabel('Regista', fontsize=14)
    plt.ylabel('Numero di Titoli', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.tight_layout()
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=12)
    plt.savefig(os.path.join(OUTPUT_CHARTS_PATH, 'top_directors.png'))
    plt.close()

    # Istogramma della lunghezza dei titoli
    title_lengths = df['Title'].apply(len)
    plt.figure(figsize=(12, 8))
    plt.hist(title_lengths, bins=30, edgecolor='black', color='skyblue')
    plt.title('Distribuzione della Lunghezza dei Titoli su Amazon Prime', fontsize=16)
    plt.xlabel('Lunghezza del Titolo (numero di caratteri)', fontsize=14)
    plt.ylabel('Numero di Titoli', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_CHARTS_PATH, 'title_lengths.png'))
    plt.close()

def main():
    extracted_df = extract_prime_dataset()
    clean_df: pd.DataFrame = preprocess_prime_dataset(extracted_df)
    clean_df.to_csv(CLEAN_PRIME_PATH, index=False)
    generate_charts(clean_df)
    print(f"Preprocessamento completato. Il dataset pulito è stato salvato in '{CLEAN_PRIME_PATH}'. Le visualizzazioni grafiche sono state salvate nella cartella '{OUTPUT_CHARTS_PATH}'.")

if __name__ == "__main__":
    main()
