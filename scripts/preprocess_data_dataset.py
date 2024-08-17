import pandas as pd
import os

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # Rimozione dei duplicati
    df = df.drop_duplicates()
    
    # Rimozione delle colonne non rilevanti
    columns_to_remove = ['description', 'duration']
    df = df.drop(columns=columns_to_remove, errors='ignore')

    # Riempimento dei valori mancanti
    df['title'] = df['title'].fillna('Unknown Title')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['director'] = df['director'].fillna('Unknown Director')
    df['rating'] = df['rating'].fillna('Unrated')
    df['release_year'] = df['release_year'].fillna(0)
    
    df['type'] = df['type'].fillna('Unknown Type')
    df['listed_in'] = df['listed_in'].fillna('Unknown Category')
    
    # Unione delle colonne 'type' e 'listed_in' in una nuova colonna 'content_category'
    df['content_category'] = df['type'] + ' - ' + df['listed_in']

    # Rimozione delle colonne originali 'type' e 'listed_in'
    df = df.drop(columns=['type', 'listed_in'], errors='ignore')

    return df

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))

    # Percorso corretto per il file di input di Netflix
    filepath = os.path.join(baseDir, '..', 'data', 'netflix_titles.csv')

    output_path = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    
    df = load_dataset(filepath)
    df = clean_data(df)
    
    df.to_csv(output_path, index=False)
    print(f"Dataset preprocessato e salvato in {output_path}")
