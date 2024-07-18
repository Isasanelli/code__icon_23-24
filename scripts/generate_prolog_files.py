import os
import pandas as pd

def clean_data(df):
    # Rimuovi duplicati
    df = df.drop_duplicates()
    # Rimuovi righe con valori NaN
    df = df.dropna(subset=['title', 'release_year', 'listed_in', 'country', 'date_added'])
    # Sostituisci caratteri speciali nei campi di testo
    df = df.replace({"'": "\\'"}, regex=True)
    return df

def generate_facts_pl(dataset, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index, row in dataset.iterrows():
            if pd.notna(row['title']) and pd.notna(row['release_year']) and pd.notna(row['listed_in']):
                genres = row['listed_in'].split(', ')
                for genre in genres:
                    fact = f"fact('{row['title']}', {row['release_year']}, '{genre}')."
                    file.write(fact + '\n')

def generate_locations_pl(dataset, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index, row in dataset.iterrows():
            if pd.notna(row['title']) and pd.notna(row['country']) and pd.notna(row['date_added']):
                countries = row['country'].split(', ')
                for country in countries:
                    location = f"location('{row['title']}', '{country}', '{row['date_added']}')."
                    file.write(location + '\n')

# Percorso del file CSV di Amazon Prime
file_path = os.path.join(os.path.dirname(__file__), '..', 'source', 'amazon_prime_titles.csv')

# Leggi il dataset
dataset = pd.read_csv(file_path)

# Verifica i dati letti
print(f"Dataset caricato con {dataset.shape[0]} righe e {dataset.shape[1]} colonne.")

# Pulisci il dataset
dataset = clean_data(dataset)

# Dividi i dataset per fatti e località
facts_dataset = dataset[['title', 'release_year', 'listed_in']]
locations_dataset = dataset[['title', 'country', 'date_added']]

# Genera i file Prolog
generate_facts_pl(facts_dataset, os.path.join(os.path.dirname(__file__), '..', 'file_prolog', 'prime_facts.pl'))
generate_locations_pl(locations_dataset, os.path.join(os.path.dirname(__file__), '..', 'file_prolog', 'prime_locations.pl'))

print("Generazione file Prolog completata.")
