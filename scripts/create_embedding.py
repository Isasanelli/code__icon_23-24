import os
import pandas as pd
from embedding import Embedding

# Percorso del file CSV con i titoli di Amazon Prime
file_path = 'source/amazon_prime_titles.csv'

# Leggi il dataset
dataset = pd.read_csv(file_path)

# Estrarre i titoli dei film
documents = dataset['title'].astype(str).tolist()

# Crea embeddings
embedding = Embedding()
embeddings = embedding.build_embedding(documents)

# Salva embeddings in un file CSV con la colonna 'title'
output_file = 'source/embeddings.csv'
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['title'] = documents
embeddings_df.to_csv(output_file, index=False)

print(f"Embeddings salvati in {output_file}")
