import pandas as pd
import spacy
import numpy as np
import os

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def generate_embeddings(df, column):
    nlp = spacy.load('en_core_web_sm')  # Carica il modello SpaCy
    embeddings = []
    
    # Genera gli embeddings per ogni voce nella colonna specificata
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
    
    # Genera embeddings per la colonna 'content_category'
    category_embeddings = generate_embeddings(df, 'content_category')
    
    # Genera embeddings per la colonna 'title'
    title_embeddings = generate_embeddings(df, 'title')
    
    # Salva gli embeddings per 'content_category'
    category_output_path = os.path.join(baseDir, '..', 'data', 'content_category_embeddings.npy')
    save_embeddings(category_embeddings, category_output_path)
    
    # Salva gli embeddings per 'title'
    title_output_path = os.path.join(baseDir, '..', 'data', 'title_embeddings.npy')
    save_embeddings(title_embeddings, title_output_path)
    
    print(f"Embeddings per 'content_category' generati e salvati in {category_output_path}")
    print(f"Embeddings per 'title' generati e salvati in {title_output_path}")
