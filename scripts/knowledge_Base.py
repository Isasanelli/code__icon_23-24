import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PRIME_DATASET_PATH = "../source/titles_selected.csv"
KNOWLEDGE_BASE_PATH = "../source/knowledge_base.csv"

def load_dataset() -> pd.DataFrame:
    """Carica il dataset preprocessato."""
    df: pd.DataFrame = pd.read_csv(PRIME_DATASET_PATH)
    return df

def create_knowledge_base(df: pd.DataFrame) -> pd.DataFrame:
    """Crea una Knowledge Base aggiungendo feature rilevanti."""
    # Feature: se il regista è lo stesso per più titoli
    df['same_director'] = df['Director'].duplicated(keep=False).astype(int)

    # Feature: similarità tra titoli basata su TF-IDF
    tfidf = TfidfVectorizer().fit_transform(df['Title'])
    df['similarity_score'] = cosine_similarity(tfidf, tfidf).diagonal()

    return df

def main():
    df = load_dataset()
    knowledge_df = create_knowledge_base(df)
    knowledge_df.to_csv(KNOWLEDGE_BASE_PATH, index=False)
    print(f"Knowledge base generata e salvata in '{KNOWLEDGE_BASE_PATH}'.")

if __name__ == "__main__":
    main()