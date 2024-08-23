import pandas as pd
import os

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def generate_prolog_facts(df, column, fact_name, output_file):
    with open(output_file, 'a') as f:
        for _, row in df.iterrows():
            value = row[column].replace("'", "").replace(" ", "_")
            f.write(f"{fact_name}('{value}').\n")

def generate_prolog_rules(df, output_file):
    with open(output_file, 'a') as f:
        f.write("\n% Regole per raccomandare i contenuti in base alla categoria\n")
        for category in df['content_category'].unique():
            f.write(f"recommend(X) :- content_category(X, '{category}'), user_likes_category('{category}').\n")

def generate_prolog_files(baseDir):
    """Funzione principale per generare i file Prolog."""
    filepath = os.path.join(baseDir,'..' ,'data', 'processed_data.csv')
    df = load_processed_data(filepath)

    # Percorso di output per i file Prolog
    prolog_output_dir = os.path.join(baseDir, '..' ,'results', 'prolog')
    os.makedirs(prolog_output_dir, exist_ok=True)

    # File di output Prolog
    prolog_file_path = os.path.join(prolog_output_dir, 'knowledge_base_fact.pl')

    # Inizializzazione del file
    with open(prolog_file_path, 'w') as f:
        f.write("% Fatti generati automaticamente dai dati\n\n")

    # Generazione dei fatti per Prolog
    generate_prolog_facts(df, 'title', 'title', prolog_file_path)
    generate_prolog_facts(df, 'content_category', 'content_category', prolog_file_path)
    
    # Se il CSV contiene colonne per rating, type, director e preference
    if 'rating' in df.columns:
        generate_prolog_facts(df, 'rating', 'rating', prolog_file_path)
    if 'type' in df.columns:
        generate_prolog_facts(df, 'type', 'type', prolog_file_path)
    if 'preference' in df.columns:
        generate_prolog_facts(df, 'preference', 'preference_for', prolog_file_path)
    
    # Generazione delle regole per Prolog
    generate_prolog_rules(df, prolog_file_path)

    print(f"File Prolog generati e salvati in {prolog_file_path}")
