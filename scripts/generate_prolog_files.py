from pyswip import Prolog
import pandas as pd
import os

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def generate_prolog_facts(df, output_path):
    facts = {
        'title': [],
        'director_of': [],
        'classify': [],
        'rating': [],
        'preference_for': []
    }

    for _, row in df.iterrows():
        title = row['title'].replace("'", "\\'")
        facts['title'].append(f"title('{title}').\n")

        if 'director' in row:
            director = row['director'].replace("'", "\\'")
            facts['director_of'].append(f"director_of('{director}', '{title}').\n")
        
        if 'content_category' in row:
            genre = row['content_category'].replace("'", "\\'")
            facts['classify'].append(f"classify('{title}', '{genre}').\n")
        
        if 'rating' in row:
            rating = row['rating'].replace("'", "\\'")
            facts['rating'].append(f"rating('{title}', '{rating}').\n")
        
        # Aggiungi altre clausole se necessario

    with open(output_path, 'w', encoding='utf-8') as f:
        for fact_type, fact_list in facts.items():
            f.writelines(fact_list)

def write_rules(prolog_file):
    with open(prolog_file, 'a', encoding='utf-8') as f:
        rules = [
            # Inserisci qui le regole di raccomandazione
            "recommend_for_kids(Title) :- rating(Title, 'G'), write('Raccomandato per bambini: '), write(Title), nl.\n",
            "recommend_for_teens(Title) :- rating(Title, 'PG-13'), write('Raccomandato per adolescenti: '), write(Title), nl.\n",
            "recommend_for_adults(Title) :- rating(Title, 'R'), write('Raccomandato per adulti: '), write(Title), nl.\n",
            "recommend_for_family(Title) :- rating(Title, 'PG'), preference_for('Family Features', Title), write('Raccomandato per famiglie: '), write(Title), nl.\n",
            "recommend_for_action_lovers(Title) :- preference_for('Action & Adventure', Title), write('Raccomandato per gli amanti dell\\'azione: '), write(Title), nl.\n",
            # Aggiungi qui altre regole come necessarie
        ]
        f.writelines(rules)

def initialize_prolog(prolog_file):
    prolog = Prolog()

    # Converte il percorso del file in un formato compatibile con Prolog
    prolog_file_prolog_format = prolog_file.replace('\\', '/')

    # Usa il percorso convertito per consultare il file Prolog
    prolog.consult(prolog_file_prolog_format)
    return prolog

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    output_path = os.path.join(baseDir, '..', 'results', 'knowledge_base', 'knowledge_base.pl')

    df = load_processed_data(data_file)

    if 'content_category' not in df.columns:
        raise KeyError("La colonna 'content_category' non è presente nel DataFrame.")
    
    # Genera i fatti Prolog
    generate_prolog_facts(df, output_path)

    # Scrivi le regole Prolog
    write_rules(output_path)
    
    # Inizializza Prolog
    prolog = initialize_prolog(output_path)
    print(f"Knowledge base Prolog generata in {output_path}")
