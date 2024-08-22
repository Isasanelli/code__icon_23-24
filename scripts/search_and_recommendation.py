import os
import pandas as pd
import matplotlib.pyplot as plt
from difflib import get_close_matches

def load_processed_data(filepath):
    """Carica i dati preprocessati dal file CSV."""
    return pd.read_csv(filepath)

def map_user_input_to_category(user_input):
    """Mappa l'input dell'utente a categorie riconosciute nel dataset."""
    category_mapping = {
        'drammatico': 'Dramas',
        'drama': 'Dramas',
        'dramma': 'Dramas',
        'commedia': 'Comedies',
        'comedy': 'Comedies',
        'comico': 'Comedies',
        'documentario': 'Documentaries',
        'documentari': 'Documentaries',
        'docuserie': 'Docuseries',
        'crime': 'Crime TV Shows',
        'crimine': 'Crime TV Shows',
        'thriller': 'Thrillers',
        'azione': 'Action & Adventure',
        'reality': 'Reality TV',
        'reality show': 'Reality TV',
        'reality tv': 'Reality TV',
        'horror': 'Horror Movies',
        'romantico': 'Romantic Movies',
        'avventura': 'Action & Adventure',
        'anime serie': 'Anime Series',
        'anime': 'Anime Series',
        'bambini': 'Children & Family Movies',
        'kids': 'Children & Family Movies',
        'familiare': 'Children & Family Movies',
        'famiglia': 'Children & Family Movies',
    }

    user_input = user_input.lower().strip()
    
    for key in category_mapping:
        if key in user_input:
            return category_mapping[key]
    
    return user_input

def search_and_recommend(baseDir, df):
    """
    Funzione principale per gestire la ricerca e le raccomandazioni.
    """
    while True:
        user_input = input("Inserisci un titolo o una categoria (es. 'Grey's Anatomy' o 'Comedy'): ")
        mapped_category = map_user_input_to_category(user_input)
        
        # Cerca prima per titolo
        title_info = search_title_with_suggestions(df, user_input)
        
        if 'exact_match' in title_info:
            show_title_info(title_info['exact_match'].iloc[0])
            rating = get_user_rating(title_info['exact_match'].iloc[0]['title'])
            if rating <= 2:
                preferred_genre = input("Sembra che questo titolo non ti sia piaciuto. Che genere preferisci? ")
                recommended_titles = recommend_top_titles(df, content_category=map_user_input_to_category(preferred_genre))
                if not recommended_titles.empty:
                    print(f"\nEcco i titoli consigliati nella categoria '{preferred_genre}':")
                    for idx, row in recommended_titles.iterrows():
                        print(f"- {row['title']}")
                else:
                    print("Nessun titolo trovato per la categoria specificata.")
            else:
                print(f"\nPerché ti è piaciuto '{title_info['exact_match'].iloc[0]['title']}', ti raccomandiamo titoli simili:")
                recommend_popular(df, title_info['exact_match'].iloc[0]['content_category'])
            show_statistics_menu(df, baseDir)
        elif 'suggestions' in title_info:
            print("\nTitolo non trovato, ma abbiamo trovato questi suggerimenti:")
            for idx, row in title_info['suggestions'].iterrows():
                print(f"- {row['title']}")
            continue
        else:
            # Se non è un titolo, cerca per categoria
            category_recommendations = recommend_top_titles(df, content_category=mapped_category)
            if not category_recommendations.empty:
                print(f"\nTitoli nella categoria '{mapped_category}':")
                for idx, row in category_recommendations.iterrows():
                    print(f"- {row['title']} (tipo: {row['content_type']})")
                continue
            else:
                print(f"Nessun risultato trovato per la categoria o titolo: {mapped_category}.")

def show_title_info(title_info):
    """Mostra le informazioni di un titolo."""
    print(f"\nTitolo: {title_info['title']}")
    print(f"Regista: {title_info['director']}")
    print(f"Cast: {title_info['cast']}")
    print(f"Anno di rilascio: {title_info['release_year']}")
    print(f"Categoria: {title_info['content_category']}")
    print(f"Preferenze: {title_info['preferences']}")

def get_user_rating(title):
    """Chiede all'utente di valutare un titolo."""
    while True:
        try:
            rating = int(input(f"Come valuti il titolo '{title}' su una scala da 1 a 5? "))
            if 1 <= rating <= 5:
                return rating
            else:
                print("Per favore, inserisci un numero da 1 a 5.")
        except ValueError:
            print("Input non valido. Inserisci un numero da 1 a 5.")

def recommend_popular(df, category):
    """Raccomanda titoli popolari basati sulla categoria."""
    popular_titles = df[df['content_category'] == category].nlargest(5, 'preferences')
    for idx, row in popular_titles.iterrows():
        print(f"- {row['title']} (Preferenze: {row['preferences']})")

def show_statistics_menu(df, baseDir):
    """Mostra il menu delle statistiche."""
    while True:
        print("\nMenu:")
        print("1) Visualizza i 5 titoli di film più popolari")
        print("2) Visualizza i 5 titoli di serie TV più popolari")
        print("3) Visualizza i generi più popolari di Netflix")
        print("4) Nuova ricerca")
        print("5) Torna al menu principale")
        
        choice = input("Seleziona un'opzione (1-5): ")
        
        if choice == '1':
            show_top_movies(df)
        elif choice == '2':
            show_top_tv_shows(df)
        elif choice == '3':
            show_most_popular_genres(df, baseDir)
        elif choice == '4':
            return  # Torna alla funzione di ricerca
        elif choice == '5':
            break  # Esce dal menu delle statistiche e torna al menu principale
        else:
            print("Selezione non valida. Riprova.")


def show_top_movies(dataframe):
    """Mostra i 5 titoli di film più popolari."""
    top_movies = dataframe[dataframe['content_category'] == 'Movie'].nlargest(5, 'preferences')
    if not top_movies.empty:
        print("\nTop 5 Film più popolari:")
        for idx, row in top_movies.iterrows():
            print(f"{idx + 1}. {row['title']} (Preferenze: {row['preferences']})")
    else:
        print("Nessun film trovato.")

def show_top_tv_shows(dataframe):
    """Mostra i 5 titoli di serie TV più popolari."""
    top_tv_shows = dataframe[dataframe['content_category'] == 'TV Show'].nlargest(5, 'preferences')
    if not top_tv_shows.empty:
        print("\nTop 5 Serie TV più popolari:")
        for idx, row in top_tv_shows.iterrows():
            print(f"{idx + 1}. {row['title']} (Preferenze: {row['preferences']})")
    else:
        print("Nessuna serie TV trovata.")


def show_most_popular_genres(dataframe, baseDir):
    """Mostra i generi più popolari su Netflix e salva il grafico."""
    save_dir = os.path.join(baseDir, '..', 'results', 'visualizations', 'statistic_recommander')
    os.makedirs(save_dir, exist_ok=True)
    
    genre_counts = dataframe['content_category'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    genre_counts.sort_values().plot(kind='bar', color=plt.cm.viridis(range(len(genre_counts))))
    plt.title('Most Popular Genre on Netflix', fontsize=16, color='blue')
    plt.xlabel('Genres', fontsize=12)
    plt.ylabel('Number of Contents', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'most_popular_genre.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Grafico salvato in: {save_path}")

def search_title_with_suggestions(dataframe, search_query, top_n=5):
    """Cerca un titolo nel dataset e suggerisce titoli simili se non trovato."""
    search_query = search_query.lower().strip()
    exact_match = dataframe[dataframe['title'].str.lower() == search_query]
    
    if not exact_match.empty:
        return {"exact_match": exact_match}
    
    similar_titles = dataframe[dataframe['title'].str.lower().str.contains(search_query)]
    
    if not similar_titles.empty:
        return {"suggestions": similar_titles}
    
    all_titles = dataframe['title'].str.lower().tolist()
    close_matches = get_close_matches(search_query, all_titles, n=top_n, cutoff=0.5)
    
    if close_matches:
        suggestions = dataframe[dataframe['title'].str.lower().isin(close_matches)]
        return {"suggestions": suggestions}
    
    return {"message": "No matching titles found."}

def recommend_top_titles(dataframe, content_category=None, top_n=10):
    """Raccomanda i titoli più popolari in base al campo 'preferences'."""
    if content_category:
        recommendations = dataframe[dataframe['content_category'].str.contains(content_category, na=False)]
    else:
        recommendations = dataframe

    if recommendations.empty:
        return pd.DataFrame()
    
    top_recommendations = recommendations.sort_values(by='preferences', ascending=False).head(top_n)
    return top_recommendations
