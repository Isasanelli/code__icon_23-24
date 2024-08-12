import os
import pandas as pd

def clean_data(df):
    """
    Pulisce il dataset rimuovendo duplicati, righe con valori NaN e sostituendo caratteri speciali.
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=['title', 'release_year', 'listed_in', 'country', 'date_added'])
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.strftime('%Y-%m-%d')
    df = df.replace({"'": "\\'"}, regex=True)
    return df

def generate_facts_pl(dataset, file_path):
    """
    Genera un file Prolog contenente i fatti relativi ai titoli, agli anni di rilascio e ai generi.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for index, row in dataset.iterrows():
            if pd.notna(row['title']) and pd.notna(row['release_year']) and pd.notna(row['listed_in']):
                genres = row['listed_in'].split(', ')
                for genre in genres:
                    fact = f"fact('{row['title']}', {row['release_year']}, '{genre}')."
                    file.write(fact + '\n')

def generate_locations_pl(dataset, file_path):
    """
    Genera un file Prolog contenente le località associate ai titoli.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for index, row in dataset.iterrows():
            if pd.notna(row['title']) and pd.notna(row['country']) and pd.notna(row['date_added']):
                countries = row['country'].split(', ')
                for country in countries:
                    location = f"location('{row['title']}', '{country}', '{row['date_added']}')."
                    file.write(location + '\n')

def main():
    # Configurazione dei percorsi dei file
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '..', 'source')
    prolog_dir = os.path.join(base_dir, '..', 'file_prolog')
    
    file_path = os.path.join(data_dir, 'amazon_prime_titles.csv')
    
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
    generate_facts_pl(facts_dataset, os.path.join(prolog_dir, 'prolog' ,'prime_facts.pl'))
    generate_locations_pl(locations_dataset, os.path.join(prolog_dir, 'prolog' ,'prime_locations.pl'))
    
    print("Generazione file Prolog completata.")

if __name__ == "__main__":
    main()
