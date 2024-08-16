import pandas as pd
import spacy
import numpy as np
import os

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def generate_embeddings(df, column):
    nlp = spacy.load('en_core_web_sm')
    embeddings = []
    
    for doc in nlp.pipe(df[column].astype('unicode').values, batch_size=50):
        if doc.has_vector:
            embeddings.append(doc.vector)
        else:
            embeddings.append(np.zeros((nlp.vocab.vectors_length,)))
    
    return np.array(embeddings)

def save_embeddings(embeddings, output_path):
    np.save(output_path, embeddings)

if __name__ == "__main__":
    # Determina il percorso della directory corrente
    baseDir = os.path.dirname(os.path.abspath(__file__))

    # Definisce il percorso assoluto del file CSV di input
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    
    # Carica i dati
    df = load_processed_data(filepath)
    
    # Genera embeddings per la colonna 'description'
    embeddings = generate_embeddings(df, 'description')
    
    # Definisce il percorso per salvare gli embeddings
    output_path = os.path.join(baseDir, '..', 'data', 'description_embeddings.npy')
    
    # Salva gli embeddings
    save_embeddings(embeddings, output_path)
    
    print(f"Embeddings generati e salvati in {output_path}")
