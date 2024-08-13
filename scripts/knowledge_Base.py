import time
import pandas as pd
import os

# Clausole definite per generi, location, anno e città
def same_genre(kb, title1, title2) -> int:
    return 1 if kb["genre"].get(title1) == kb["genre"].get(title2) else 0

def same_location(kb, title1, title2) -> int:
    return 1 if kb["location"].get(title1) == kb["location"].get(title2) else 0

def same_year(kb, title1, title2) -> int:
    return 1 if kb["year"].get(title1) == kb["year"].get(title2) else 0

def same_city(kb, title1, title2) -> int:
    return 1 if kb["city"].get(title1) == kb["city"].get(title2) else 0

# Funzione per creare il KB
def create_kb(facts_df) -> dict:
    kb = {
        "genre": {},
        "location": {},
        "year": {},
        "city": {}
    }

    for index, row in facts_df.iterrows():
        if pd.notnull(row['title']):
            title = str(row['title']).lower().strip()
            kb["genre"][title] = row['listed_in'].strip()
            kb["location"][title] = "N/A"  # Nessuna colonna "country" nel DataFrame originale
            kb["year"][title] = row['release_year']
            kb["city"][title] = "N/A"  # Nessuna colonna "city" nel DataFrame originale

    return kb

# Funzione per calcolare le feature
def calculate_features(kb, title_id) -> dict:
    features_dict = {}
    another_title = 'another title'  # Puoi cambiare questo con un altro titolo di riferimento

    features_dict["TITLE"] = title_id
    features_dict["SAME_GENRE"] = same_genre(kb, title_id, another_title)
    features_dict["SAME_LOCATION"] = same_location(kb, title_id, another_title)
    features_dict["SAME_YEAR"] = same_year(kb, title_id, another_title)
    features_dict["SAME_CITY"] = same_city(kb, title_id, another_title)

    # Estrazione dei valori di YEAR e CITY dal knowledge base
    features_dict["YEAR"] = kb["year"].get(title_id, "N/A")
    features_dict["CITY"] = kb["city"].get(title_id, "N/A")

    return features_dict

# Funzione per produrre il dataset finale
def produce_working_dataset(kb: dict, path: str):
    print(f"Producing dataset at {path}")
    start = time.time()
    
    titles_complete: pd.DataFrame = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'source', 'preprocessed', 'titles_selected.csv'))
    titles_complete.columns = titles_complete.columns.str.lower()  # Converte tutte le colonne in minuscolo

    extracted_values_df = pd.DataFrame()

    for title_id in titles_complete["title"]:  # Ora la colonna "title" è assicurata di essere minuscola
        if pd.notnull(title_id):
            title_id = str(title_id).lower().strip()  # Normalizza il titolo
            features_dict = calculate_features(kb, title_id)
            extracted_values_df = pd.concat([extracted_values_df, pd.DataFrame([features_dict])], ignore_index=True)

    extracted_values_df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'source', 'kb', path), index=False)
    end = time.time()
    print("Total time: ", end-start)

# Caricamento dei fatti dal dataset originale
facts_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'source', 'preprocessed', 'titles_selected.csv'))
facts_df.columns = facts_df.columns.str.lower()  # Normalizza i nomi delle colonne in minuscolo

# Creazione del knowledge base
knowledge_base = create_kb(facts_df)

# Produzione del dataset con le caratteristiche estratte
produce_working_dataset(knowledge_base, "working_dataset.csv")
