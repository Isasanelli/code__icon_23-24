import re
import pandas as pd
import os

def load_processed_data(filepath):
    return pd.read_csv(filepath, encoding='utf-8')

def generate_prolog_facts(df, output_file):
    content_categories = []
    preferences = []
    
    for _, row in df.iterrows():
        title = re.sub(r'[^\x00-\x7F]+', '', row['title'].replace("'", "").replace(" ", "_"))
        category = re.sub(r'[^\x00-\x7F]+', '', row['content_category'].replace("'", "").replace(" ", "_"))
        preference = re.sub(r'[^\x00-\x7F]+', '', str(row['preferences']).replace("'", "").replace(" ", "_"))
        
        content_categories.append(f"content_category('{title}', '{category}').\n")
        preferences.append(f"preference_for('{title}', '{preference}').\n")
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("% Fatti generati automaticamente dai dati\n\n")
        f.writelines(content_categories)
        f.writelines(preferences)

def generate_prolog_rules(df, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% Regole per raccomandare i contenuti in base alla categoria, preferenze e generi popolari\n")

        # Raccomandazione basata su categoria e preferenze
        for _, row in df.iterrows():
            title = re.sub(r'[^\x00-\x7F]+', '', row['title'].replace("'", "").replace(" ", "_"))
            category = re.sub(r'[^\x00-\x7F]+', '', row['content_category'].replace("'", "").replace(" ", "_"))
            preference = re.sub(r'[^\x00-\x7F]+', '', str(row['preferences']).replace("'", "").replace(" ", "_"))
            
            f.write(f"recommend('{title}') :- content_category('{title}', '{category}'), preference_for('{title}', '{preference}').\n")
        
        # Raccomandazione basata sui generi popolari
        for _, row in df.iterrows():
            title = re.sub(r'[^\x00-\x7F]+', '', row['title'].replace("'", "").replace(" ", "_"))
            category = re.sub(r'[^\x00-\x7F]+', '', row['content_category'].replace("'", "").replace(" ", "_"))
            
            f.write(f"recommend_genre('{title}') :- content_category('{title}', '{category}'), popular_genre('{category}', _).\n")
        
        # Regola dinamica basata sull'input dell'utente
        f.write("\n% Regole dinamiche basate su input utente\n")
        f.write("recommend(Title) :-\n")
        f.write("    ask_user('Qual è il tuo genere preferito? ', Genere),\n")
        f.write("    find_content(Title, Genere).\n\n")
        
        f.write("find_content(Title, Genere) :-\n")
        f.write("    content_category(Title, Genere).\n\n")
        
        f.write("ask_user(Prompt, Response) :-\n")
        f.write("    write(Prompt),\n")
        f.write("    read(Response).\n")

def generate_prolog_files(baseDir):
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    df = load_processed_data(filepath)

    prolog_output_dir = os.path.join(baseDir, '..', 'results', 'prolog')
    os.makedirs(prolog_output_dir, exist_ok=True)

    prolog_facts_path = os.path.join(prolog_output_dir, 'knowledge_base_fact.pl')
    with open(prolog_facts_path, 'w', encoding='utf-8') as f:
        f.write("% Fatti generati automaticamente dai dati\n\n")
    
    generate_prolog_facts(df, prolog_facts_path)
    
    prolog_rules_path = os.path.join(prolog_output_dir, 'knowledge_base_rules.pl')
    generate_prolog_rules(df, prolog_rules_path)

    print(f"Fatti Prolog generati e salvati in {prolog_facts_path}")
    print(f"Regole Prolog generate e salvate in {prolog_rules_path}")
