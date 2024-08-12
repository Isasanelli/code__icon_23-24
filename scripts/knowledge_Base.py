import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PRIME_DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'source', 'preprocessed', 'titles_selected.csv')
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'source', 'kb')
KNOWLEDGE_BASE_PATH = os.path.join(KNOWLEDGE_BASE_DIR, 'kb.csv')

def load_dataset() -> pd.DataFrame:
    """Carica il dataset preprocessato."""
    df: pd.DataFrame = pd.read_csv(PRIME_DATASET_PATH)
    return df

def create_knowledge_base(df: pd.DataFrame) -> pd.DataFrame:
    """Crea una Knowledge Base aggiungendo feature rilevanti."""
    # Gestione dei valori mancanti o non validi nella colonna 'Title' e 'Director'
    df['Title'] = df['Title'].fillna('').astype(str)
    df['Director'] = df['Director'].fillna('').astype(str)

    # Feature: se il regista è lo stesso per più titoli
    df['same_director'] = df.groupby('Director')['Director'].transform(lambda x: (x != "").sum() > 1).astype(int)

    # Feature: similarità tra titoli basata su TF-IDF
    tfidf = TfidfVectorizer().fit_transform(df['Title'])
    similarity_matrix = cosine_similarity(tfidf)
    df['similarity_score'] = similarity_matrix.diagonal()  # Similarità di ogni titolo con se stesso, che sarà sempre 1

    return df

def main():
    # Creazione della directory se non esiste
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

    df = load_dataset()
    knowledge_df = create_knowledge_base(df)
    knowledge_df.to_csv(KNOWLEDGE_BASE_PATH, index=False)
    print(f"Knowledge base generata e salvata in '{KNOWLEDGE_BASE_PATH}'.")

if __name__ == "__main__":
    main()
