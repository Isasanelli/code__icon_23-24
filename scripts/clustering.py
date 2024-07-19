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

def make_clusters():
    embeddings_path = os.path.join(script_dir, '../source/embeddings.csv')
    df = pd.read_csv(embeddings_path)

    # Normalize embeddings
    titles = df["title"]
    df = pd.DataFrame(normalize(df.drop(["title"], axis=1), axis=1))
    df = df.assign(title=titles)
    # Save normalized embeddings
    normalized_embeddings_path = os.path.join(script_dir, '../source/normalized_embeddings.csv')
    df.to_csv(normalized_embeddings_path, index=False)

    wcss = []
    my_embedding = df.drop(["title"], axis=1).to_numpy()

    # Trying to get best number of clusters
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
    elbow_plot_path = os.path.join(script_dir, '../charts/clustering/grafico_elbow.png')
    plt.savefig(elbow_plot_path)
    plt.show()

make_clusters()

def final_cluster():
    embeddings_path = os.path.join(script_dir, '../source/embeddings.csv')
    df = pd.read_csv(embeddings_path)

    my_embedding = df.drop(["title"], axis=1).to_numpy()
    n = 4
    clustering = KMeans(n_clusters=n, init='k-means++', random_state=100, n_init=10)
    clustering.fit(my_embedding)

    df["Cluster"] = clustering.labels_

    complete_df = df
    clustered_dataset_path = os.path.join(script_dir, '../source/dataset_clustered.csv')
    complete_df.to_csv(clustered_dataset_path)

    # Verificare quali colonne esistono effettivamente nel dataframe
    existing_columns = complete_df.columns.tolist()
    columns = ["release_year", "country", "date_added", "Cluster"]
    columns = [col for col in columns if col in existing_columns]

    complete_df = complete_df[columns]
    complete_df.dropna()

    df_statistics = []
    for i in range(0, n):
        df_statistics.append(complete_df.loc[complete_df["Cluster"] == i].copy())
        statistics_path = os.path.join(script_dir, f"../source/Statistiche_Cluster_{i}.csv")
        df_statistics[i].describe().to_csv(statistics_path)
        
        # Rimuovere le colonne non numeriche per la heatmap
        numeric_df = df_statistics[i].select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            correlation = numeric_df.corr()
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
            heatmap_path = os.path.join(script_dir, f"../charts/clustering/Heatmap_Cluster_{i}.png")
            plt.title(f"Heatmap della correlazione per Cluster {i}")
            plt.savefig(heatmap_path)
            plt.close()

    for df_stat in df_statistics:
        df_stat.drop(["Cluster"], axis=1, inplace=True)

    columns.remove("Cluster")

    for col in columns:
        f, p = stats.f_oneway(*(df_stat[col].dropna().values for df_stat in df_statistics))
        if p < 0.05:  # hp
            print(f"Dati diversi per colonna: {col}, valori: F={f}, p={p}")
        else:
            print(f"Dati non molto diversi per colonna: {col}, valori: F={f}, p={p}")

final_cluster()

def create_wordcloud():
    clustered_dataset_path = os.path.join(script_dir, '../source/dataset_clustered.csv')
    df = pd.read_csv(clustered_dataset_path)
    news = [''] * 4

    for index, row in df.iterrows():
        news[int(row["Cluster"])] += row["title"] + " "

    for i in range(0, 4):
        wc = wordcloud.WordCloud(stopwords=set(wordcloud.STOPWORDS)).generate(news[i])
        plt.imshow(wc)
        plt.axis('off')
        plt.title(f"Word Cloud per Cluster {i}")
        wordcloud_path = os.path.join(script_dir, f"../charts/clustering/Wordcloud_Cluster_{i}.png")
        plt.savefig(wordcloud_path)
        plt.show()

create_wordcloud()
