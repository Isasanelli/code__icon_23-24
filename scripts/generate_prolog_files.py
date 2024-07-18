import os
import pandas as pd


def clean_data(df):
    # Rimuovi duplicati
    df = df.drop_duplicates()
    # Rimuovi righe con valori NaN
    df = df.dropna()
    # Sostituisci caratteri speciali nei campi di testo
    df = df.replace({"'": "\\'"}, regex=True)
    return df

def generate_facts_pl(dataset):
    os.makedirs('file_prolog', exist_ok=True)
    with open('file_prolog/prime_facts.pl', 'w', encoding='utf-8') as file:
        for index, row in dataset.iterrows():
            if pd.notna(row['title']) and pd.notna(row['release_year']) and pd.notna(row['listed_in']):
                fact = f"fact('{row['title']}', {row['release_year']}, '{row['listed_in']}')."
                file.write(fact + '\n')

def generate_locations_pl(dataset):
    os.makedirs('file_prolog', exist_ok=True)
    with open('file_prolog/prime_locations.pl', 'w', encoding='utf-8') as file:
        for index, row in dataset.iterrows():
            if pd.notna(row['title']) and pd.notna(row['country']) and pd.notna(row['date_added']):
                location = f"location('{row['title']}', '{row['country']}', '{row['date_added']}')."
                file.write(location + '\n')

# Percorso del file CSV di Amazon Prime
file_path = os.path.join(os.path.dirname(__file__), '..', 'source', 'amazon_prime_titles.csv')

# Leggi il dataset
dataset = pd.read_csv(file_path)

# Verifica i dati letti
print(f"Dataset caricato con {dataset.shape[0]} righe e {dataset.shape[1]} colonne.")

# Rimuovi i duplicati
dataset = dataset.drop_duplicates()

# Rimuovi le righe con valori NaN solo nelle colonne rilevanti
facts_dataset = dataset[['title', 'release_year', 'listed_in']].dropna()
locations_dataset = dataset[['title', 'country', 'date_added']].dropna()

# Verifica i dati dopo la pulizia
print(f"Dataset per facts: {facts_dataset.shape[0]} righe")
print(f"Dataset per locations: {locations_dataset.shape[0]} righe")

# Genera i file Prolog
generate_facts_pl(facts_dataset)
generate_locations_pl(locations_dataset)

print("Generazione file Prolog completata.")
