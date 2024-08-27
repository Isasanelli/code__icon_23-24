import os
import pandas as pd

def load_processed_data(filepath):
    return pd.read_csv(filepath, encoding='utf-8')

def append_new_content(df, title, category, year, preference):
    new_row = {'title': title, 'content_category': category, 'release_year': year, 'preferences': preference}
    df = df.append(new_row, ignore_index=True)
    return df

def update_prolog_facts(df, prolog_facts_path):
    with open(prolog_facts_path, 'w', encoding='utf-8') as f:
        f.write("% Fatti aggiornati sui contenuti\n\n")
        for _, row in df.iterrows():
            title = row['title'].replace("'", "\\'").replace(" ", "_")
            category = row['content_category'].replace("'", "\\'").replace(" ", "_")
            year = row['release_year']
            preference = row['preferences']
            f.write(f"content('{title}', '{category}', {year}, {preference}).\n")

def generate_prolog_facts(df, prolog_facts_path):
    with open(prolog_facts_path, 'w', encoding='utf-8') as f:
        f.write("% Fatti sui contenuti\n\n")
        for _, row in df.iterrows():
            title = row['title'].replace("'", "\\'").replace(" ", "_")
            category = row['content_category'].replace("'", "\\'").replace(" ", "_")
            year = row['release_year']
            preference = row['preferences']
            f.write(f"content('{title}', '{category}', {year}, {preference}).\n")

def generate_prolog_rules(prolog_rules_path):
    with open(prolog_rules_path, 'w', encoding='utf-8') as f:
        f.write("% Regole per la raccomandazione dei contenuti\n\n")
        
        # Regola per raccomandare contenuti con alta preferenza
        f.write("recommend(Content) :-\n")
        f.write("    content(Content, _, _, Preference),\n")
        f.write("    Preference >= 80.\n\n")
        
        # Regola per raccomandare contenuti simili in base alla categoria
        f.write("similar_content(Title1, Title2) :-\n")
        f.write("    content(Title1, Category, _, _),\n")
        f.write("    content(Title2, Category, _, _),\n")
        f.write("    Title1 \\= Title2.\n\n")

        # Raccomandazione di contenuti simili con preferenza alta
        f.write("recommend_similar(Title, RecommendedTitle) :-\n")
        f.write("    similar_content(Title, RecommendedTitle),\n")
        f.write("    content(RecommendedTitle, _, _, Preference),\n")
        f.write("    Preference >= 60.\n\n")

        # Raccomandare contenuti basati su un genere specifico (predeterminato o dato dall'utente)
        f.write("recommend_by_genre(Genre, Content) :-\n")
        f.write("    content(Content, Genre, _, Preference),\n")
        f.write("    Preference >= 50.\n\n")
        
        # Regola per raccomandare contenuti basati sull'anno di rilascio
        f.write("recommend_by_year(Year, Content) :-\n")
        f.write("    content(Content, _, Year, Preference),\n")
        f.write("    Preference >= 50.\n\n")
        
        # Regola per raccomandare contenuti basati su una combinazione di genere e anno
        f.write("recommend_by_genre_year(Genre, Year, Content) :-\n")
        f.write("    content(Content, Genre, Year, Preference),\n")
        f.write("    Preference >= 50.\n\n")

def generate_prolog_files(baseDir):
    if baseDir is None:
        raise ValueError("Il percorso baseDir è None. Assicurati di passare un percorso valido.")

    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Il file {filepath} non esiste. Verifica il percorso e riprova.")
    
    df = load_processed_data(filepath)

    prolog_output_dir = os.path.join(baseDir, '..', 'results', 'prolog')
    os.makedirs(prolog_output_dir, exist_ok=True)

    prolog_facts_path = os.path.join(prolog_output_dir, 'facts.pl')
    prolog_rules_path = os.path.join(prolog_output_dir, 'rules.pl')

    generate_prolog_facts(df, prolog_facts_path)
    generate_prolog_rules(prolog_rules_path)

    print(f"Fatti Prolog generati e salvati in {prolog_facts_path}")
    print(f"Regole Prolog generate e salvate in {prolog_rules_path}")
