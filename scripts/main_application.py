from preprocess_prime_dataset import load_dataset, clean_data, encode_categorical, save_clean_data
from analyze_data import load_processed_data, plot_distribution, plot_correlation
from clustering import apply_clustering, visualize_clusters
from supervised import train_model
from probabilistic_learning import train_probabilistic_model
from generate_prolog_files import generate_prolog_facts
from knowledge_base import create_ontology
from cross_validation import evaluate_model
import os

if __name__ == "__main__":
    # 1. Preprocessing dei dati
       # Determina il percorso della directory corrente
    baseDir = os.path.dirname(os.path.abspath(__file__))

    # Definisce il percorso assoluto del file CSV di input
    filepath = os.path.join(baseDir, '..', 'data', 'amazon_prime_titles.csv')
    
    output_path = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    
    df = load_dataset(filepath)
    df = clean_data(df)
    df, label_encoders = encode_categorical(df)
    save_clean_data(df, output_path)
    print("1. Preprocessing dei dati completato.")
    
    # 2. Analisi dei dati
    df = load_processed_data(output_path)
    plot_distribution(df, 'release_year', 'Distribution of Titles by Release Year')
    plot_distribution(df, 'rating', 'Distribution of Titles by Rating')
    plot_correlation(df)
    print("2. Analisi dei dati completata.")
    
    # 3. Clustering
    df, kmeans_model = apply_clustering(df, n_clusters=5)
    visualize_clusters(df)
    print("3. Clustering completato.")
    
    # 4. Apprendimento Supervisionato
    supervised_model = train_model(df)
    print("4. Apprendimento supervisionato completato.")
    
    # 5. Apprendimento Probabilistico
    probabilistic_model = train_probabilistic_model(df)
    print("5. Apprendimento probabilistico completato.")
    
    # 6. Cross-Validation e valutazione dei modelli
    features = ['rating', 'release_year', 'duration']
    
    rf_metrics = evaluate_model(supervised_model, df[features], df['most_watched'])
    nb_metrics = evaluate_model(probabilistic_model, df[features], df['most_watched'])

    print("Random Forest Metrics:")
    print(f"Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"Precision: {rf_metrics['precision']:.4f}")
    print(f"Recall: {rf_metrics['recall']:.4f}")
    print(f"F1 Score: {rf_metrics['f1_score']:.4f}")

    print("\nNaive Bayes Metrics:")
    print(f"Accuracy: {nb_metrics['accuracy']:.4f}")
    print(f"Precision: {nb_metrics['precision']:.4f}")
    print(f"Recall: {nb_metrics['recall']:.4f}")
    print(f"F1 Score: {nb_metrics['f1_score']:.4f}")
    print("6. Cross-Validation e valutazione dei modelli completata.")
    
    # 7. Generazione dei fatti Prolog
    prolog_output_file = '../results/movies_facts.pl'
    generate_prolog_facts(df, prolog_output_file)
    print(f"7. Fatti Prolog generati e salvati in {prolog_output_file}.")
    
    # 8. Creazione dell'ontologia e della Knowledge Base
    ontology = create_ontology()
    print("8. Ontologia creata e salvata.")
    
    print("Processo completo. Tutte le fasi sono state eseguite con successo.")
