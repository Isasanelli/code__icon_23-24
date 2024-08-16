import pandas as pd
import os
import re

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def clean_string(s):
    """Converte il valore in stringa e rimuove o sostituisce caratteri non sicuri per Prolog."""
    s = str(s)  # Converte il valore in stringa
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

def generate_prolog_facts(df, output_file):
    with open(output_file, 'w') as f:
        for index, row in df.iterrows():
            title = clean_string(row['title'])
            director = clean_string(row['director'])
            content_type = clean_string(row['type'])
            
            # Verifica se è un film o una serie TV e genera i fatti Prolog di conseguenza
            if content_type.lower() == 'movie':
                fact = f"movie('{title}', '{director}', {row['release_year']}, {row['rating']}).\n"
            elif content_type.lower() == 'tv_show':
                fact = f"tv_show('{title}', '{director}', {row['release_year']}, {row['rating']}).\n"
            else:
                continue  # Salta righe con tipi di contenuto sconosciuti

            f.write(fact)

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))

    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    
    output_path = os.path.join(baseDir, '..', 'data', 'content_facts.pl')
    
    df = load_processed_data(filepath)
    generate_prolog_facts(df, output_path)
    
    print(f"Fatti Prolog generati e salvati in {output_path}")
