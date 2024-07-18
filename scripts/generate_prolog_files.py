import os
import pandas as pd

def generate_facts_pl(dataset):
    with open('source/prime_facts.pl', 'w', encoding='utf-8') as file:
        for index, row in dataset.iterrows():
            fact = f"fact('{row['title']}', {row['release_year']}, '{row['listed_in']}')."
            file.write(fact + '\n')

def generate_locations_pl(dataset):
    with open('source/prime_locations.pl', 'w', encoding='utf-8') as file:
        for index, row in dataset.iterrows():
            location = f"location('{row['title']}', '{row['country']}', '{row['date_added']}')."
            file.write(location + '\n')

# Percorso del file CSV di Amazon Prime
file_path = 'source/amazon_prime_titles.csv'

# Leggi il dataset
dataset = pd.read_csv(file_path)

# Genera i file Prolog
generate_facts_pl(dataset)
generate_locations_pl(dataset)
