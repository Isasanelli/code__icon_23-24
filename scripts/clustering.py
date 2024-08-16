import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def load_embeddings(filepath):
    return np.load(filepath)

def apply_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans

def visualize_clusters(X, clusters, output_dir):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=clusters, palette='viridis', s=100, alpha=0.7)
    plt.title('Clusters Visualization (PCA)', fontsize=18)
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)
    plt.legend(title='Cluster', fontsize=12, loc='upper right')
    
    # Salva il grafico nella directory specificata
    output_path = os.path.join(output_dir, 'clusters_visualization_pca.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Determina il percorso della directory corrente
    baseDir = os.path.dirname(os.path.abspath(__file__))

    # Carica il dataset processato
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    df = load_processed_data(filepath)
    
    # Carica gli embeddings
    embeddings_path = os.path.join(baseDir, '..', 'data', 'description_embeddings.npy')
    embeddings = load_embeddings(embeddings_path)
    
    # Unisci gli embeddings alle altre caratteristiche
    features = np.hstack([embeddings, df[['rating', 'release_year', 'duration']].values])
    
    # Applica il clustering
    clusters, kmeans_model = apply_clustering(features, n_clusters=5)
    
    # Visualizza e salva i cluster
    output_dir = os.path.join(baseDir, '..', 'results', 'visualizations', 'clustering')
    visualize_clusters(features, clusters, output_dir)
    
    print(f"Clustering completato e grafico salvato in {output_dir}")
