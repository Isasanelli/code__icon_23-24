import pandas as pd
import os

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    print("Distribuzione dei tipi di contenuto prima della pulizia:")
    print(df['type'].value_counts())

    
    return df

def clean_data(df):
    # Rimozione dei duplicati
    df = df.drop_duplicates()
    
    # Riempimento dei valori mancanti
    df['title'] = df['title'].fillna('Unknown Title')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['director'] = df['director'].fillna('Unknown Director')
    df['rating'] = df['rating'].fillna('Unrated')
    df['release_year'] = df['release_year'].fillna(0)
    
    df['type'] = df['type'].fillna('Unknown Type')
    df['duration'] = df['duration'].fillna('Unknown Duration')
    
    # Pulizia della colonna Duration
    df['duration'] = df.apply(lambda row: clean_duration(row['duration'], row['type']), axis=1)
    
    
    return df

def clean_duration(duration, content_type):
    if pd.isna(duration):
        return 0
    
    if content_type == 'Movie':
        try:
            return int(duration.replace(' min', ''))
        except ValueError:
            return 0
    
    elif content_type == 'TV Show':
        if 'Season' in duration or 'Seasons' in duration:
            return int(duration.split(' ')[0]) * 10
        elif 'Episode' in duration or 'Episodes' in duration:
            return int(duration.split(' ')[0])
        else:
            return 1
    
    else:
        return 0

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))

    filepath = os.path.join(baseDir, '..', 'data', 'amazon_prime_titles.csv')

    output_path = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    
    df = load_dataset(filepath)
    df = clean_data(df)
    print("Distribuzione dei tipi di contenuto dopo la pulizia:")
    print(df['type'].value_counts())
    
    df.to_csv(output_path, index=False)
    print(f"Dataset preprocessato e salvato in {output_path}")
