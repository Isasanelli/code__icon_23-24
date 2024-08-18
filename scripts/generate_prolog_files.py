import os
import pandas as pd

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def generate_prolog_facts(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Genera i fatti per i titoli
    with open(os.path.join(output_dir, 'titles.pl'), 'w', encoding='utf-8') as f:
        for title in df['title'].unique():
            title_safe = title.replace("'", "\\'")
            f.write(f"title('{title_safe}').\n")
    
    # Genera i fatti per i director
    with open(os.path.join(output_dir, 'directors.pl'), 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            if pd.notna(row['director']):
                director_safe = row['director'].replace("'", "\\'")
                title_safe = row['title'].replace("'", "\\'")
                f.write(f"director_of('{director_safe}', '{title_safe}').\n")
    
    # Genera i fatti per il rating
    with open(os.path.join(output_dir, 'ratings.pl'), 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            if pd.notna(row['rating']):
                rating_safe = row['rating'].replace("'", "\\'")
                title_safe = row['title'].replace("'", "\\'")
                f.write(f"rating('{title_safe}', '{rating_safe}').\n")
    
    # Genera i fatti per le categorie di contenuto
    with open(os.path.join(output_dir, 'categories.pl'), 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            if pd.notna(row['content_category']):
                category_safe = row['content_category'].replace("'", "\\'")
                title_safe = row['title'].replace("'", "\\'")
                f.write(f"classify('{title_safe}', '{category_safe}').\n")

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

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    output_dir = os.path.join(baseDir, '..', 'results', 'knowledge_base')

    df = load_processed_data(data_file)

    if 'content_category' not in df.columns:
        raise KeyError("La colonna 'content_category' non è presente nel DataFrame.")
    
    # Genera i fatti Prolog
    generate_prolog_facts(df, output_dir)

    # Scrivi le regole Prolog in un file separato
    rules_file = os.path.join(output_dir, 'rules.pl')
    write_rules(rules_file)
    
    print(f"Knowledge base Prolog generata in {output_dir}")
