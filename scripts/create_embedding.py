import pandas as pd
import spacy
import numpy as np
import os

def load_processed_data(filepath):
    """Carica i dati preprocessati dal file CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Il file '{filepath}' non esiste.")
    return pd.read_csv(filepath)

def generate_embeddings(df, column, nlp):
    """Genera gli embeddings per una colonna specificata del DataFrame utilizzando SpaCy."""
    embeddings = []
    
    # Verifica se la colonna esiste
    if column not in df.columns:
        raise ValueError(f"La colonna '{column}' non esiste nel DataFrame.")
    
    # Verifica la presenza di valori mancanti nella colonna
    df[column] = df[column].fillna('')  # Sostituisce eventuali valori mancanti con stringhe vuote

    # Genera gli embeddings per ogni voce nella colonna specificata
    for doc in nlp.pipe(df[column].astype('unicode').values, batch_size=50):
        if doc.has_vector:
            embeddings.append(doc.vector)
        else:
            embeddings.append(np.zeros((nlp.vocab.vectors_length,)))
    
    return np.array(embeddings)

def save_embeddings(embeddings, output_path):
    """Salva gli embeddings in formato NumPy."""
    try:
        np.save(output_path, embeddings)
        print(f"Embeddings salvati in {output_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio degli embeddings: {e}")

def generate_and_save_embeddings(df, column, output_dir, nlp):
    """Genera e salva gli embeddings per una specifica colonna."""
    try:
        print(f"Generazione degli embeddings per la colonna '{column}'...")
        embeddings = generate_embeddings(df, column, nlp)
        
        output_path = os.path.join(output_dir, f'{column}_embeddings.npy')
        save_embeddings(embeddings, output_path)
    except Exception as e:
        print(f"Errore durante la generazione o il salvataggio degli embeddings per '{column}': {e}")

def create_embeddings_pipeline(baseDir):
    try:
        # Carica il modello SpaCy
        print("Caricamento del modello SpaCy...")
        nlp = spacy.load('en_core_web_sm')
        
        # Percorso dei dati e della directory di output
        filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
        output_dir = os.path.join(baseDir, '..', 'data')

        # Creazione della directory di output se non esiste
        os.makedirs(output_dir, exist_ok=True)
        
        # Caricamento dei dati
        df = load_processed_data(filepath)

        # Genera e salva gli embeddings per 'content_category'
        generate_and_save_embeddings(df, 'content_category', output_dir, nlp)
        
        # Genera e salva gli embeddings per 'title'
        generate_and_save_embeddings(df, 'title', output_dir, nlp)
    except Exception as e:
        print(f"Errore durante la pipeline di creazione degli embeddings: {e}")
