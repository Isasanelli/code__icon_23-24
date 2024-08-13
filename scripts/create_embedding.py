import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from wordcloud import WordCloud, STOPWORDS
import gensim.downloader as api

# Percorsi dei file
PRIME_DATASET_PATH = 'source/amazon_prime_titles.csv'
EMBEDDINGS_PATH = 'source/embeddings/embeddings_word2vec.csv'
PCA_IMAGE_PATH = 'charts/embeddings/embedding_pca.png'
WORDCLOUD_PATH = 'charts/embeddings/wordcloud.png'

# Assicurati che le directory esistano
os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PCA_IMAGE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(WORDCLOUD_PATH), exist_ok=True)

class Embedding:
    @staticmethod
    def build_embedding(original_sentences):
        MAX_SEQUENCE_LENGTH = 80

        tokenizer = Tokenizer(char_level=False)
        tokenizer.fit_on_texts(original_sentences)

        sequences = tokenizer.texts_to_sequences(original_sentences)
        word_index = tokenizer.word_index

       # Padding dei titoli
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

        word2vec_model = api.load('glove-wiki-gigaword-100')

        news = []
        for sentence in padded_sequences:
            sum_emb = [0 for _ in range(100)]  # Our size will be 100
            num_tok = 0
            for tok in sentence:
                if tok != 0:
                    word = list(word_index.keys())[list(word_index.values()).index(tok)]
                    try:
                        word_embedding = word2vec_model[word]
                        sum_emb = [x + y for x, y in zip(sum_emb, word_embedding)]
                        num_tok += 1
                    except KeyError:  # skip if the word is not in the model
                        pass
            if num_tok != 0:
                news.append([x / num_tok for x in sum_emb])
            else:
                news.append(sum_emb)

        return np.array(news)

def create_embedding():
    df = pd.read_csv(PRIME_DATASET_PATH)

    # Preprocessamento e tokenizzazione dei titoli
    documents = [title.lower().strip() for title in df['title'].dropna()]

    # Genera embedding utilizzando la classe Embedding
    embeddings = Embedding.build_embedding(documents)

    # Converti gli embeddings in un DataFrame
    embedding_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(100)])
    embedding_df['title'] = documents

    # Filtra le righe con tutti valori 0
    embedding_df = embedding_df[(embedding_df.iloc[:, :-1] != 0).any(axis=1)]

    # Salva gli embeddings in un file CSV
    embedding_df.to_csv(EMBEDDINGS_PATH, index=False)
    print(f"Embeddings salvati in {EMBEDDINGS_PATH}")

def visualize_embeddings_with_pca():
    # Carica gli embeddings dal file CSV
    embedding_df = pd.read_csv(EMBEDDINGS_PATH)
    
    # Isola solo le colonne degli embedding (escludendo la colonna dei titoli)
    embeddings = embedding_df.iloc[:, :-1].values
    
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

def create_wordcloud():
    df = pd.read_csv(EMBEDDINGS_PATH)
    
    # Combina tutti i titoli in una singola stringa
    text = " ".join(df['title'].tolist())

    # Genera la word cloud
    wc = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=600).generate(text)

    # Mostra la word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    
    # Salva la word cloud
    plt.savefig(WORDCLOUD_PATH)
    plt.show()

    print(f"Word cloud salvato in {WORDCLOUD_PATH}")

# Esecuzione delle funzioni
create_embedding()
visualize_embeddings_with_pca()
create_wordcloud()
