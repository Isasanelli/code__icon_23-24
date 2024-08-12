import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Percorso del file CSV con i titoli di Amazon Prime
PRIME_DATASET_PATH = 'source/amazon_prime_titles.csv'
EMBEDDINGS_PATH = 'source/embeddings/embeddings_word2vec.csv'
PCA_IMAGE_PATH = 'charts/embeddings/embedding_pca.png'

# Assicurati che le directory esistano
os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PCA_IMAGE_PATH), exist_ok=True)

# Carica il dataset
dataset = pd.read_csv(PRIME_DATASET_PATH)

# Estrai i titoli dei film
documents = dataset['title'].astype(str).tolist()

# Rimuovi titoli duplicati e normalizza i testi
documents = [doc.lower().strip() for doc in set(documents)]

# Tokenizzazione dei titoli per Word2Vec
tokenized_docs = [doc.split() for doc in documents]

# Creazione del modello Word2Vec
model = Word2Vec(sentences=tokenized_docs, vector_size=50, window=5, min_count=1, workers=4)

# Costruzione degli embeddings per ciascun titolo
embeddings = []
for doc in tokenized_docs:
    vector = np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
    embeddings.append(vector)

# Converti gli embeddings in un DataFrame
embedding_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(model.vector_size)])
embedding_df['title'] = documents

# Salva gli embeddings in un file CSV
embedding_df.to_csv(EMBEDDINGS_PATH, index=False)
print(f"Embeddings salvati in {EMBEDDINGS_PATH}")

# Riduzione della dimensionalità con PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Visualizzazione degli embeddings ridotti
plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
plt.title('PCA of Embeddings')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)

# Salva il grafico PCA
plt.savefig(PCA_IMAGE_PATH)
plt.show()

print(f"Grafico PCA salvato in {PCA_IMAGE_PATH}")
