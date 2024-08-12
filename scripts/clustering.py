import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy import stats
import nltk
import wordcloud

nltk.download('punkt')

# Definire la directory dello script per i percorsi relativi
script_dir = os.path.dirname(__file__)

def make_clusters(embeddings_path, elbow_plot_path, normalized_embeddings_path):
    """
    Funzione per creare cluster usando il metodo del gomito (Elbow Method).
    """
    df = pd.read_csv(embeddings_path)

    # Normalize embeddings
    titles = df["title"]
    df = pd.DataFrame(normalize(df.drop(["title"], axis=1), axis=1))
    df = df.assign(title=titles)

    # Save normalized embeddings
    df.to_csv(normalized_embeddings_path, index=False)

    wcss = []
    my_embedding = df.drop(["title"], axis=1).to_numpy()

    # Determinare il miglior numero di cluster
    for i in range(1, 21):
        clustering = KMeans(n_clusters=i, init='k-means++', random_state=100, n_init=10)
        clustering.fit(my_embedding)
        wcss.append(clustering.inertia_)

    ks = list(range(1, 21))
    plt.xticks(ks)
    plt.plot(ks, wcss)
    plt.xlabel('Numero di cluster (K)')
    plt.ylabel('Somma delle distanze quadrate')
    plt.title('Metodo Elbow per K ottimale')
    plt.savefig(elbow_plot_path)
    plt.show()

def final_cluster(embeddings_path, clustered_dataset_path, charts_dir, n_clusters=4):
    """
    Funzione per eseguire il clustering finale con il numero di cluster specificato.
    """
    df = pd.read_csv(embeddings_path)
    my_embedding = df.drop(["title"], axis=1).to_numpy()

    # Eseguire il clustering
    clustering = KMeans(n_clusters=n_clusters, init='k-means++', random_state=100, n_init=10)
    clustering.fit(my_embedding)

    df["Cluster"] = clustering.labels_
    df.to_csv(clustered_dataset_path, index=False)

    # Analizzare e visualizzare i risultati
    analyze_clusters(df, charts_dir, n_clusters)

def analyze_clusters(df, charts_dir, n_clusters):
    """
    Analizza i cluster e genera visualizzazioni delle correlazioni.
    """
    existing_columns = df.columns.tolist()
    columns = ["release_year", "country", "date_added", "Cluster"]
    columns = [col for col in columns if col in existing_columns]

    df = df[columns].dropna()

    for i in range(n_clusters):
        cluster_df = df[df["Cluster"] == i].copy()
        statistics_path = os.path.join(charts_dir, f"Statistiche_Cluster_{i}.csv")
        cluster_df.describe().to_csv(statistics_path)

        numeric_df = cluster_df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            correlation = numeric_df.corr()
            heatmap_path = os.path.join(charts_dir, f"Heatmap_Cluster_{i}.png")
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
            plt.title(f"Heatmap della correlazione per Cluster {i}")
            plt.savefig(heatmap_path)
            plt.close()

        f_test_columns(numeric_df, cluster_df)

def f_test_columns(numeric_df, cluster_df):
    """
    Esegue il test ANOVA per determinare se ci sono differenze significative tra i cluster.
    """
    for col in numeric_df.columns:
        f, p = stats.f_oneway(*(cluster_df[cluster_df["Cluster"] == i][col].dropna().values for i in range(numeric_df["Cluster"].nunique())))
        if p < 0.05:
            print(f"Dati diversi per colonna: {col}, valori: F={f}, p={p}")
        else:
            print(f"Dati non molto diversi per colonna: {col}, valori: F={f}, p={p}")

def create_wordcloud(clustered_dataset_path, charts_dir, n_clusters=4):
    """
    Genera word cloud per ciascun cluster.
    """
    df = pd.read_csv(clustered_dataset_path)
    news = [''] * n_clusters

    for index, row in df.iterrows():
        news[int(row["Cluster"])] += row["title"] + " "

    for i in range(n_clusters):
        wc = wordcloud.WordCloud(stopwords=set(wordcloud.STOPWORDS)).generate(news[i])
        plt.imshow(wc)
        plt.axis('off')
        plt.title(f"Word Cloud per Cluster {i}")
        wordcloud_path = os.path.join(charts_dir, f"Wordcloud_Cluster_{i}.png")
        plt.savefig(wordcloud_path)
        plt.show()

if __name__ == "__main__":
    # Configurazione dei percorsi
    base_dir = os.path.dirname(__file__)
    source_dir = os.path.join(base_dir, '../source')
    charts_dir = os.path.join(base_dir, '../charts/clustering')

    embeddings_path = os.path.join(source_dir, 'embeddings.csv')
    normalized_embeddings_path = os.path.join(source_dir, 'normalized_embeddings.csv')
    elbow_plot_path = os.path.join(charts_dir, 'grafico_elbow.png')
    clustered_dataset_path = os.path.join(source_dir, 'dataset_clustered.csv')

    # Esegui il clustering e le analisi
    make_clusters(embeddings_path, elbow_plot_path, normalized_embeddings_path)
    final_cluster(embeddings_path, clustered_dataset_path, charts_dir)
    create_wordcloud(clustered_dataset_path, charts_dir)
