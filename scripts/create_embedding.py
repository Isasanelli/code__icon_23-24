import os
import pandas as pd
from embedding import Embedding
from multiprocessing import Pool
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Percorso del file CSV con i titoli di Amazon Prime
file_path = 'source/amazon_prime_titles.csv'

# Leggi il dataset
dataset = pd.read_csv(file_path)

# Estrarre i titoli dei film
documents = dataset['Title'].astype(str).tolist()

# Rimuovere titoli duplicati
documents = list(set(documents))

# Normalizzare i titoli: tutto minuscolo e rimozione di caratteri speciali
documents = [doc.lower().strip() for doc in documents if pd.notnull(doc)]

# Crea embeddings
embedding = Embedding()

# Funzione per generare embeddings in parallelo
def generate_embeddings(documents):
    with Pool() as pool:
        embeddings = pool.map(embedding.build_embedding, [doc for doc in documents if doc])
    return embeddings

# Genera gli embeddings
embeddings = generate_embeddings(documents)

# Rinominare le colonne degli embeddings
column_names = [f'embedding_{i}' for i in range(len(embeddings[0]))]
embeddings_df = pd.DataFrame(embeddings, columns=column_names)

# Aggiungi i titoli come colonna
embeddings_df['Title'] = [doc for doc in documents if doc]

# Salva embeddings in un file CSV
output_file = 'path_to_your_dataset/embeddings.csv'
embeddings_df.to_csv(output_file, index=False)

print(f"Embeddings salvati in {output_file}")

# Visualizzazione degli embeddings con PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
plt.title('PCA of Embeddings')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()
