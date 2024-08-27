import re
import pandas as pd
import os

def load_processed_data(filepath):
    return pd.read_csv(filepath, encoding='utf-8')

def generate_prolog_facts(df, output_file):
    content_categories = []
    preferences = []
    popular_genres = {}
    
    for _, row in df.iterrows():
        title = re.sub(r'[^\x00-\x7F]+', '', row['title'].replace("'", "").replace(" ", "_"))
        category = re.sub(r'[^\x00-\x7F]+', '', row['content_category'].replace("'", "").replace(" ", "_"))
        preference = re.sub(r'[^\x00-\x7F]+', '', str(row['preferences']).replace("'", "").replace(" ", "_"))

        content_categories.append(f"content_category('{title}', '{category}').\n")
        preferences.append(f"preference_for('{title}', '{preference}').\n")

        # Traccia i generi popolari
        if category not in popular_genres:
            popular_genres[category] = 1
        else:
            popular_genres[category] += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% Fatti generati automaticamente dai dati\n\n")
        f.writelines(content_categories)
        f.writelines(preferences)

        # Aggiungi i fatti per popular_genre/2
        f.write("\n% Fatti generati automaticamente per i generi popolari\n")
        for genre, count in popular_genres.items():
            f.write(f"popular_genre('{genre}', {count}).\n")

def generate_prolog_rules(df, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% Regole per interagire con l'utente e fornire suggerimenti\n\n")

        # Regola dinamica basata sull'input dell'utente
        f.write("recommend(Title) :-\n")
        f.write("    ask_user('Qual è il tuo genere preferito? ', Genere),\n")
        f.write("    find_content(Title, Genere).\n\n")

        # Raccomandazione basata su categoria e preferenze
        f.write("% Raccomandazione basata su categoria e preferenze\n")
        for _, row in df.iterrows():
            title = re.sub(r'[^\x00-\x7F]+', '', row['title'].replace("'", "").replace(" ", "_"))
            category = re.sub(r'[^\x00-\x7F]+', '', row['content_category'].replace("'", "").replace(" ", "_"))
            preference = re.sub(r'[^\x00-\x7F]+', '', str(row['preferences']).replace("'", "").replace(" ", "_"))
            
            f.write(f"recommend('{title}') :- content_category('{title}', '{category}'), preference_for('{title}', '{preference}').\n")
        
        # Regola find_content
        f.write("\nfind_content(Title, Genere) :-\n")
        f.write("    content_category(Title, Genere).\n\n")
        
        # Regola ask_user
        f.write("ask_user(Prompt, Response) :-\n")
        f.write("    write(Prompt),\n")
        f.write("    read(Response).\n\n")

        # Raccomandazione basata sui generi popolari
        f.write("% Raccomandazione basata sui generi popolari\n")
        for _, row in df.iterrows():
            title = re.sub(r'[^\x00-\x7F]+', '', row['title'].replace("'", "").replace(" ", "_"))
            category = re.sub(r'[^\x00-\x7F]+', '', row['content_category'].replace("'", "").replace(" ", "_"))
            
            f.write(f"recommend_genre('{title}') :- content_category('{title}', '{category}'), popular_genre('{category}', _).\n")

def generate_prolog_files(baseDir):
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    df = load_processed_data(filepath)

    prolog_output_dir = os.path.join(baseDir, '..', 'results', 'prolog')
    os.makedirs(prolog_output_dir, exist_ok=True)

    # Genera il file dei fatti
    prolog_facts_path = os.path.join(prolog_output_dir, 'facts.pl')
    generate_prolog_facts(df, prolog_facts_path)

    # Genera il file delle regole
    prolog_rules_path = os.path.join(baseDir, '..', 'results', 'prolog', 'rules.pl')
    generate_prolog_rules(df, prolog_rules_path)

    print(f"Fatti Prolog generati e salvati in {prolog_facts_path}")
    print(f"Regole Prolog generate e salvate in {prolog_rules_path}")
