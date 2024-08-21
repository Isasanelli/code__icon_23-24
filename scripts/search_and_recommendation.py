import os
import pandas as pd
import matplotlib.pyplot as plt
from difflib import get_close_matches

def load_processed_data(filepath):
    """Carica i dati preprocessati dal file CSV."""
    return pd.read_csv(filepath)

def map_user_input_to_category(user_input):
    """
    Mappa l'input dell'utente a categorie riconosciute nel dataset.
    """
    category_mapping = {
        'drammatico': 'Dramas',
        'drama': 'Dramas',
        'dramma': 'Dramas',
        'commedia': 'Comedies',
        'comedy': 'Comedies',
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
        'romantico': 'Romantic Movies'
    }

    user_input = user_input.lower().strip()
    
    for key in category_mapping:
        if key in user_input:
            return category_mapping[key]
    
    return user_input

def search_and_recommend(baseDir, df, user_input=None, top_n=5):
    """
    Funzione principale per gestire la ricerca e le raccomandazioni.
    """
    try:
        if user_input:
            mapped_category = map_user_input_to_category(user_input)
            category_recommendations = recommend_top_titles(df, content_category=mapped_category)

            # Attempt to find an exact match or suggestions
            title_search_result = search_title_with_suggestions(df, user_input)

            if isinstance(category_recommendations, pd.DataFrame) and not category_recommendations.empty:
                print(f"\nTitoli nella categoria '{mapped_category}':")
                for idx, row in category_recommendations.iterrows():
                    print(f"- {row['title']}")
                
            if 'exact_match' in title_search_result:
                title_info = title_search_result['exact_match'].iloc[0]
                print(f"\nHai scelto: {title_info['title']}")
                print(f"Regista: {title_info['director']}")
                print(f"Cast: {title_info['cast']}")
                print(f"Anno di rilascio: {title_info['release_year']}")
                print(f"Categoria: {title_info['content_category']}")
                print(f"Preferenze: {title_info['preferences']}")

                rating = int(input(f"Come valuti il titolo '{title_info['title']}' su una scala da 1 a 5? "))

                if rating <= 2:
                    print(f"Sembra che questo titolo non sia di tuo gradimento. Che genere ti piace?")
                    preferred_genre = input("Inserisci un genere (es. 'Commedia', 'Drammatico'): ")
                    mapped_genre = map_user_input_to_category(preferred_genre)
                    genre_recommendations = recommend_top_titles(df, content_category=mapped_genre)
                    if isinstance(genre_recommendations, pd.DataFrame) and not genre_recommendations.empty:
                        print(f"\nTitoli raccomandati nella categoria '{mapped_genre}':")
                        for idx, row in genre_recommendations.iterrows():
                            print(f"- {row['title']}")
                        # Richiedi una nuova valutazione
                        selected_title = input("\nSeleziona un titolo dalla lista per valutare: ")
                        selected_title_info = search_title_with_suggestions(df, selected_title)
                        
                        if 'exact_match' in selected_title_info and not selected_title_info['exact_match'].empty:
                            title_info = selected_title_info['exact_match'].iloc[0]
                            rating = int(input(f"Come valuti il titolo '{title_info['title']}' su una scala da 1 a 5? "))
                            print(f"Grazie! Hai valutato '{title_info['title']}' con un {rating}/5.")
                            # Mostra le raccomandazioni se la valutazione è positiva
                            if rating >= 3:
                                print(f"\nPerché ti è piaciuto '{title_info['title']}', ti raccomandiamo i titoli simili a '{title_info['title']}':")
                                recommend_popular(df, title_info['content_category'])
                    else:
                        print(f"Nessun risultato trovato per il genere: {mapped_genre}.")
                else:
                    print(f"Grazie! Hai valutato '{title_info['title']}' con un {rating}/5.")
                    print(f"\nPerché ti è piaciuto '{title_info['title']}', ti raccomandiamo i titoli simili a '{title_info['title']}':")
                    recommend_popular(df, title_info['content_category'])
            elif 'suggestions' in title_search_result:
                print("\nTitolo non trovato, ma abbiamo trovato questi suggerimenti:")
                for idx, row in title_search_result['suggestions'].iterrows():
                    print(f"- {row['title']}")

                selected_title = input("\nSeleziona un titolo dai suggerimenti o riprova: ")
                search_and_recommend(baseDir, df, user_input=selected_title)
            else:
                print("Il titolo scelto non è stato trovato.")
                retry = input("Vuoi riprovare? (s/n): ")
                if retry.lower() == 's':
                    user_input = input("Inserisci un nuovo titolo o categoria: ")
                    search_and_recommend(baseDir, df, user_input=user_input)
                return None

    except Exception as e:
        print(f"Errore durante la ricerca e raccomandazione: {e}")
        return None

def search_and_recommend_titoli(baseDir):
    """
    Funzione che gestisce la raccomandazione e poi visualizza il menu.
    """
    df = load_processed_data(os.path.join(baseDir, '..','data', 'processed_data.csv'))
    while True:
        user_input = input("Inserisci un titolo o una categoria (es. 'Grey's Anatomy' o 'Comedy'): ")
        
        # Effettua la raccomandazione
        search_and_recommend(baseDir, df, user_input=user_input)
        
        # Dopo la raccomandazione, visualizza il menu
        while True:
            print("\nMenu:")
            print("1) Visualizza i 5 titoli di film più popolari")
            print("2) Visualizza i 5 titoli di serie TV più popolari")
            print("3) Visualizza i generi più popolari su Netflix")
            print("4) Nuova ricerca")
            print("5) Esci")
            
            choice = input("Seleziona un'opzione (1-6): ")
            
            if choice == '1':
                show_top_movies(df, baseDir)
            elif choice == '2':
                show_top_tv_shows(df, baseDir)
            elif choice == '3':
                show_most_popular_genres(df, baseDir)
            elif choice == '4':
                break  # Break the menu loop to allow a new search
            elif choice == '5':
                print("Uscita dal programma.")
                return
            else:
                print("Selezione non valida. Riprova.")


def show_top_movies(dataframe, baseDir):
    """
    Mostra i 5 titoli di film più popolari.
    """
    save_dir = os.path.join(baseDir,'..', 'results', 'visualizations', 'statistic_recommander')
    os.makedirs(save_dir, exist_ok=True)
    
    top_movies = dataframe[dataframe['content_category'] == 'Movie'].nlargest(5, 'preferences')
    if not top_movies.empty:
        data = top_movies.set_index('title')['preferences']
        generate_chart(data, 'Top 5 Movies', 'Movie Titles', 'Preferences', os.path.join(save_dir, 'top_5_movies.png'))
    else:
        print("Nessun film trovato.")

def show_top_tv_shows(dataframe, baseDir):
    """
    Mostra i 5 titoli di serie TV più popolari.
    """
    save_dir = os.path.join(baseDir, '..','results', 'visualizations', 'statistic_recommander')
    os.makedirs(save_dir, exist_ok=True)
    
    top_tv_shows = dataframe[dataframe['content_category'] == 'TV Show'].nlargest(5, 'preferences')
    if not top_tv_shows.empty:
        data = top_tv_shows.set_index('title')['preferences']
        generate_chart(data, 'Top 5 TV Shows', 'TV Show Titles', 'Preferences', os.path.join(save_dir, 'top_5_tv_shows.png'))
    else:
        print("Nessuna serie TV trovata.")


def show_most_popular_movies(dataframe, baseDir):
    """
    Mostra i film più popolari su Netflix e salva il grafico.
    """
    save_dir = os.path.join(baseDir,'..' ,'results', 'visualizations', 'statistic_recommander')
    os.makedirs(save_dir, exist_ok=True)
    
    movie_counts = dataframe[dataframe['content_category'] == 'Movie']['title'].value_counts().head(10)
    
    # Set a smaller figure size and use a color palette
    plt.figure(figsize=(12, 6))
    movie_counts.sort_values().plot(kind='bar', color=plt.cm.viridis(range(len(movie_counts))))
    plt.title('Most Popular Movies on Netflix', fontsize=16, color='blue')
    plt.xlabel('Movie Titles', fontsize=12)
    plt.ylabel('Number of Contents', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'most_popular_movies.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Grafico salvato in: {save_path}")

def show_most_popular_tv_shows(dataframe, baseDir):
    """
    Mostra le serie TV più popolari su Netflix e salva il grafico.
    """
    save_dir = os.path.join(baseDir, '..' ,'results', 'visualizations', 'statistic_recommander')
    os.makedirs(save_dir, exist_ok=True)
    
    tv_show_counts = dataframe[dataframe['content_category'] == 'TV Show']['title'].value_counts().head(10)
    
    # Set a smaller figure size and use a color palette
    plt.figure(figsize=(12, 6))
    tv_show_counts.sort_values().plot(kind='bar', color=plt.cm.viridis(range(len(tv_show_counts))))
    plt.title('Most Popular TV Shows on Netflix', fontsize=16, color='blue')
    plt.xlabel('TV Show Titles', fontsize=12)
    plt.ylabel('Number of Contents', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'most_popular_tv_shows.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Grafico salvato in: {save_path}")

def show_most_popular_genres(dataframe, baseDir):
    """
    Mostra i generi più popolari su Netflix e salva il grafico.
    """
    save_dir = os.path.join(baseDir,'..' , 'results', 'visualizations', 'statistic_recommander')
    os.makedirs(save_dir, exist_ok=True)
    
    genre_counts = dataframe['content_category'].value_counts().head(10)
    
    # Set a smaller figure size and use a color palette
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



def generate_chart(data, title, xlabel, ylabel, save_path):
    """
    Genera un grafico a barre e lo salva nel percorso specificato.
    """
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def recommend_top_titles(dataframe, content_category=None, top_n=10):
    """
    Raccomanda i titoli più popolari in base al campo 'preferences'.
    Se non viene fornita una categoria specifica, restituisce i titoli globalmente più popolari.
    """
    if content_category:
        recommendations = dataframe[dataframe['content_category'].str.contains(content_category, na=False)]
    else:
        recommendations = dataframe

    if recommendations.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no recommendations are available
    
    top_recommendations = recommendations.sort_values(by='preferences', ascending=False).head(top_n)
    return top_recommendations

def search_title_with_suggestions(dataframe, search_query, top_n=5):
    """
    Cerca un titolo nel dataset. Se il titolo è trovato, restituisce le informazioni.
    Se non trovato, suggerisce titoli simili basati sulla similarità testuale.
    """
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

def recommend_popular(dataframe, content_category=None):
    """
    Raccomanda i titoli più popolari per film e serie TV in base a 'preferences'.
    """
    print("\nFilm più popolari:")
    popular_movies = recommend_top_titles(dataframe, content_category="Movie", top_n=5)
    if not popular_movies.empty:
        for title in popular_movies['title']:
            print(f"- {title}")
    else:
        print("Nessun film popolare trovato.")

    print("\nSerie TV più popolari:")
    popular_tv_shows = recommend_top_titles(dataframe, content_category="TV Show", top_n=5)
    if not popular_tv_shows.empty:
        for title in popular_tv_shows['title']:
            print(f"- {title}")
    else:
        print("Nessuna serie TV popolare trovata.")
