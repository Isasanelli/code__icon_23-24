import pandas as pd
import nltk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def make_clusters():
    nltk.download('punkt')
    df = pd.read_csv("source/embeddings.csv")
    if 'title' not in df.columns:
        print("Errore: La colonna 'title' non è presente nel file embeddings.csv")
        return

    titles = df["title"]
    df = pd.DataFrame(normalize(df.drop(["title"], axis=1), axis=1))
    df = df.assign(title=titles)
    df.to_csv("source/normalized_embeddings.csv", index=False)

    wcss = []
    my_embedding = df.drop(["title"], axis=1).to_numpy()
    for i in range(1, 21):
        clustering = KMeans(n_clusters=i, init='k-means++', random_state=100, n_init=10)
        clustering.fit(my_embedding)
        wcss.append(clustering.inertia_)

    ks = list(range(1, 21))
    plt.xticks(ks)
    plt.plot(ks, wcss)
    plt.show()

def final_cluster():
    df = pd.read_csv("source/normalized_embeddings.csv")
    my_embedding = df.drop(["title"], axis=1).to_numpy()
    n = 4
    clustering = KMeans(n_clusters=n, init='k-means++', random_state=100, n_init=10)
    clustering.fit(my_embedding)
    df["Cluster"] = clustering.labels_
    df.to_csv("source/dataset_clustered.csv", index=False)
    for i in range(n):
        df.loc[df["Cluster"] == i].describe().to_csv(f"source/Statistic{i}.csv")

if __name__ == "__main__":
    make_clusters()
    final_cluster()
