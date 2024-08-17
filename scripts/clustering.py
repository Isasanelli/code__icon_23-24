import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def load_embeddings(filepath):
    return np.load(filepath)

def clean_and_encode_data(df):
    df['rating'] = df['rating'].replace('Unrated', np.nan)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Riempie i valori mancanti con la media per 'rating'
    df['rating'] = df['rating'].fillna(df['rating'].mean())
    
    # Riempie i valori mancanti nelle colonne categoriali con 'Unknown'
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    
    return df

def verify_data_after_cleaning(df):
    print("Numero di campioni dopo la pulizia:")
    print(df['content_category'].value_counts())
    print("\nEsempi di dati dopo la pulizia:")
    print(df.head())

def apply_clustering(X, n_clusters):
    # Imputation dei NaN nelle caratteristiche
    if np.any(np.isnan(X)):
        print("Riempimento dei NaN con la media delle colonne...")
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    # Rimuove eventuali colonne che non contengono valori osservati
    non_nan_columns = ~np.isnan(X).all(axis=0)
    X = X[:, non_nan_columns]

    # Verifica se ci sono ancora NaN residui e rimuovi le righe con NaN
    if np.any(np.isnan(X)):
        print("Rimozione delle righe con valori NaN residui...")
        X = X[~np.isnan(X).any(axis=1)]
    
    if len(X) < n_clusters:
        n_clusters = len(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans

def visualize_clusters(X, clusters, output_dir, filter_type):
    n_samples, n_features = X.shape
    
    n_components = min(n_samples, n_features, 2)
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=clusters, palette='viridis', s=100, alpha=0.7)
    plt.title(f'Clusters Visualization (PCA) - {filter_type}', fontsize=18)
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)
    plt.legend(title='Cluster', fontsize=12, loc='upper right')
    
    output_path = os.path.join(output_dir, f'clusters_visualization_pca_{filter_type}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    df = load_processed_data(filepath)
    
    # Verifica le colonne presenti nel DataFrame
    print("Colonne disponibili nel DataFrame:")
    print(df.columns)
    
    # Verifica se esiste una colonna che può distinguere tra Movie e TV Show
    if 'content_category' not in df.columns:
        raise KeyError("La colonna 'content_category' non è presente nel DataFrame.")

    df = clean_and_encode_data(df)
    verify_data_after_cleaning(df)  # Verifica di nuovo i dati dopo la pulizia
    
    embeddings_path = os.path.join(baseDir, '..', 'data', 'content_category_embeddings.npy')
    embeddings = load_embeddings(embeddings_path)
    
    output_dir = os.path.join(baseDir, '..', 'results', 'visualizations', 'clustering')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    at_least_one_clustered = False

    for content_type in ['Movie', 'TV Show']:
        print(f"Eseguendo il clustering per: {content_type}")
        
        df_filtered = df[df['content_category'].str.contains(content_type, case=False)].reset_index(drop=True)
        print(f"Numero di campioni per {content_type}: {len(df_filtered)}")
        
        if len(df_filtered) == 0:
            print(f"Nessun campione trovato per {content_type}. Salto il clustering.")
            continue
        
        embeddings_filtered = embeddings[df_filtered.index.values]
        features_filtered = np.hstack([embeddings_filtered, df_filtered[['rating', 'release_year']].values])
        
        print(f"Dimensione delle caratteristiche per {content_type}: {features_filtered.shape}")
        
        if len(features_filtered) < 2 or features_filtered.shape[1] < 2:
            print(f"Numero insufficiente di campioni o caratteristiche per il clustering di {content_type}.")
            continue
        
        clusters, kmeans_model = apply_clustering(features_filtered, n_clusters=5)
        visualize_clusters(features_filtered, clusters, output_dir, content_type)
        at_least_one_clustered = True
    
    if at_least_one_clustered:
        print(f"Clustering completato e grafici salvati in {output_dir}")
    else:
        print("Nessun clustering eseguito per nessuna categoria.")
