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
        'dramas': 'Dramas',
        'commedia': 'Comedies',
        'comedy': 'Comedies',
        'comico': 'Comedies',
        'documentario': 'Documentaries',
        'documentari': 'Documentaries',
        'docuserie': 'Docuseries',
        'crime': 'Crime TV Shows',
        'crimine': 'Crime TV Shows',
        'thriller': 'Thrillers',
        'reality': 'Reality TV',
        'reality show': 'Reality TV',
        'reality tv': 'Reality TV',
        'horror': 'Horror Movies',
        'romantico': 'Romantic Movies',
        'avventura': 'Action & Adventure',
        'adventure': 'Action & Adventure',
        'azione': 'Action & Adventure',
        'action': 'Action & Adventure',
        'fantasy': 'Anime Series',
        'anime serie': 'Anime Series',
        'anime': 'Anime Series',
        'bambini': 'Children & Family Movies',
        'kids': 'Children & Family Movies',
        'familiare': 'Children & Family Movies',
        'famiglia': 'Children & Family Movies',
        'scientifico': 'Sci-Fi & Fantasy',
        'sci-fi': 'Sci-Fi & Fantasy',
        'fantastico': 'Sci-Fi & Fantasy',
        'mistero': 'Mystery',
    }

    user_input = user_input.lower().strip()
    
    for key in category_mapping:
        if key in user_input:
            return category_mapping[key]
    
    return user_input
def search_by_title(df):
    while True:
        print("\n" + "="*60)
        print(" Seleziona un'opzione:")
        print("="*60)
        print(" 1. Scrivere un titolo (es. 'Grey's Anatomy', 'Breaking Bad')")
        print(" 2. Scrivere 'ricerca' per tornare alla selezione di ricerca")
        print(" 3. Scrivere 'menu' per tornare al menu principale")
        print("="*60)
        user_input = input(" >> ").strip().lower()
        
        if user_input == 'menu' or user_input == '3':
            return 'menu'
        elif user_input == 'ricerca' or user_input == '2':
            return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper
        elif user_input == '1':
            title_input = input(" Inserisci il titolo: ").strip().lower()
            title_info = search_title_with_suggestions(df, title_input)

            if 'exact_match' in title_info:
                show_title_info(title_info['exact_match'].iloc[0])
                rating = get_user_rating(title_info['exact_match'].iloc[0]['title'])
                if rating <= 2:
                    preferred_genre = input("\nSembra che questo titolo non ti sia piaciuto. Che genere preferisci? ")
                    result = search_by_category(df, map_user_input_to_category(preferred_genre))
                    if result == 'menu':
                        return 'menu'
                    elif result == 'ricerca':
                        return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper
                else:
                    recommend_popular(df, title_info['exact_match'].iloc[0]['content_category'], exclude_title=title_info['exact_match'].iloc[0]['title'])
                    result = post_rating_menu(df, os.path.dirname(os.path.abspath(__file__)))
                    if result == 'menu':
                        return 'menu'
                    elif result == 'ricerca':
                        return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper
            elif 'suggestions' in title_info:
                print("\nTitolo non trovato, ma abbiamo trovato questi suggerimenti:")
                for idx, row in title_info['suggestions'].iterrows():
                    print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
                while True:
                    print("\n" + "="*60)
                    print(" Menu di Selezione:")
                    print("="*60)
                    print(" 1. Seleziona un titolo dalla lista")
                    print(" 2. Fare una nuova ricerca")
                    print(" 3. Tornare al menu principale")
                    print("="*60)
                    title_choice = input(" Seleziona un'opzione (1-3): ").strip()

                    if title_choice == '1':
                        selected_title = input("\nInserisci solo il titolo esatto dalla lista, non includendo la categoria e la preferenza: ").strip().lower()
                        exact_match = df[df['title'].str.lower() == selected_title]
                        if not exact_match.empty:
                            show_title_info(exact_match.iloc[0])
                            rating = get_user_rating(exact_match.iloc[0]['title'])
                            if rating <= 2:
                                preferred_genre = input("\nSembra che questo titolo non ti sia piaciuto. Che genere preferisci? ")
                                result = search_by_category(df, map_user_input_to_category(preferred_genre))
                                if result == 'menu':
                                    return 'menu'
                                elif result == 'ricerca':
                                    return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper
                            else:
                                recommend_popular(df, exact_match.iloc[0]['content_category'], exclude_title=exact_match.iloc[0]['title'])
                                result = post_rating_menu(df, os.path.dirname(os.path.abspath(__file__)))
                                if result == 'menu':
                                    return 'menu'
                                elif result == 'ricerca':
                                    return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper
                        else:
                            print("\nTitolo non trovato nella lista suggerita. Riprova.")
                    elif title_choice == '2':
                        return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper
                    elif title_choice == '3':
                        return 'menu'
                    else:
                        print("\nOpzione non valida. Per favore, seleziona 1, 2 o 3.")
            else:
                print(f"\nNessun risultato trovato per: {title_input}.")
                continue
        else:
            print("\nOpzione non valida. Per favore, seleziona 1, 2 o 3.")


def search_by_category(df, category=None):
    while True:
        if category is None:
            print("\n" + "="*60)
            print(" Seleziona una categoria:")
            print("="*60)
            print(" 1. Inserisci una categoria (es. 'Comedy', 'Drama', 'Horror')")
            print(" 2. Scrivi 'ricerca' per tornare alla selezione di ricerca")
            print(" 3. Scrivi 'menu' per tornare al menu principale")
            print("="*60)
            user_input = input(" >> ").strip().lower()
            
            if user_input == 'menu' or user_input == '3':
                return 'menu'
            elif user_input == 'ricerca' or user_input == '2':
                return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper
            elif user_input == '1':
                category_input = input(" Inserisci la categoria: ").strip().lower()
                category = map_user_input_to_category(category_input)
            else:
                print("\nOpzione non valida. Per favore, seleziona 1, 2 o 3.")
                continue
        
        category_recommendations = recommend_top_titles(df, content_category=category)
        
        if not category_recommendations.empty:
            print(f"\nTitoli nella categoria '{category}':")
            for idx, row in category_recommendations.iterrows():
                print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")

            while True:
                print("\n" + "="*60)
                print(" Menu di Selezione:")
                print("="*60)
                print(" 1. Seleziona un titolo dalla lista")
                print(" 2. Fare una nuova ricerca")
                print(" 3. Tornare al menu principale")
                print("="*60)
                title_choice = input(" Seleziona un'opzione (1-3): ").strip()

                if title_choice == '1':
                    selected_title = input("\nInserisci solo il titolo esatto dalla lista, non includendo la categoria e la preferenza: ").strip().lower()
                    exact_match = df[df['title'].str.lower() == selected_title]
                    if not exact_match.empty:
                        show_title_info(exact_match.iloc[0])
                        rating = get_user_rating(exact_match.iloc[0]['title'])
                        if rating <= 2:
                            preferred_genre = input("\nSembra che questo titolo non ti sia piaciuto. Che genere preferisci? ")
                            result = search_by_category(df, map_user_input_to_category(preferred_genre))
                            if result == 'menu':
                                return 'menu'
                            elif result == 'ricerca':
                                return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper
                        else:
                            result = post_rating_menu(df, os.path.dirname(os.path.abspath(__file__)))
                            if result == 'menu':
                                return 'menu'
                            elif result == 'ricerca':
                                return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper
                    else:
                        print("\nTitolo non trovato nella lista suggerita. Riprova.")
                elif title_choice == '2':
                    return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper
                elif title_choice == '3':
                    return 'menu'
                else:
                    print("\nOpzione non valida. Per favore, seleziona 1, 2 o 3.")
        else:
            print(f"\nNessun risultato trovato per la categoria: {category}.")
            return 'ricerca'  # Torna a search_and_recommend_titoli_wrapper



def show_title_info(title_info):
    """Mostra le informazioni di un titolo in un formato più compatto."""
    print("\n" + "="*50)
    print(f"Titolo       : {title_info['title']}")
    print(f"Regista      : {title_info['director']}")
    print(f"Cast         : {title_info['cast']}")
    print(f"Anno         : {title_info['release_year']}")
    print(f"Categoria    : {title_info['content_category']}")
    print(f"Preferenze   : {title_info['preferences']}")
    print("="*50)

def get_user_rating(title):
    """Chiede all'utente di valutare un titolo."""
    while True:
        try:
            rating = int(input(f"Come valuti il titolo '{title}' su una scala da 1 a 5? "))
            if 1 <= rating <= 5:
                return rating
            else:
                print("\nPer favore, inserisci un numero da 1 a 5.")
        except ValueError:
            print("\nInput non valido. Inserisci un numero da 1 a 5.")

def recommend_popular(df, category, exclude_title=None):
    """Raccomanda titoli popolari basati sulla categoria, escludendo il titolo corrente."""
    print(f"\n Dato che ti è piaciuto '{exclude_title}',\n ti raccomandiamo titoli nella categoria '{category}': \n")
    
    # Filtro per categoria, escludendo il titolo corrente
    filtered_df = df[df['content_category'].str.contains(category, case=False, na=False)]
    
    if exclude_title:
        filtered_df = filtered_df[filtered_df['title'].str.lower() != exclude_title.lower()]
    
    if filtered_df.empty:
        print(f"Nessun titolo trovato nella categoria '{category}'.")
        return
    
    # Ottenere i 5 titoli più popolari
    popular_titles = filtered_df.nlargest(5, 'preferences')
    
    if popular_titles.empty:
        print(f"Nessun titolo popolare trovato nella categoria '{category}'.")
    else:
        for idx, row in popular_titles.iterrows():
            print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")

            
def post_rating_menu(df, baseDir):
    """Mostra il menu post valutazione con varie opzioni."""
    while True:
        print("\n" + "="*60)
        print(" Menu Post Valutazione")
        print("="*60)
        print(" 1) Visualizza i 5 film più popolari di Netflix")
        print(" 2) Visualizza i 5 titoli di serie TV più popolari")
        print(" 3) Visualizza le categorie più popolari su Netflix")
        print(" 4) Nuova ricerca")
        print(" 5) Torna al menu principale")
        print("="*60)
        
        choice = input(" Seleziona un'opzione (1-5): ").strip()
        
        if choice == '1':
            show_top_movies(df)
        elif choice == '2':
            show_top_tv_shows(df)
        elif choice == '3':
            show_most_popular_genres(df, baseDir)
        elif choice == '4':
            from main import search_and_recommend_titoli_wrapper  # Import locale per evitare circolare
            search_and_recommend_titoli_wrapper(baseDir)  
            return 'ricerca'  
        elif choice == '5':
            from main import display_menu  # Import locale per evitare circolare
            display_menu()
            return 'menu'  
        else:
            print(" Selezione non valida. Riprova.")



def show_top_movies(dataframe):
    """Mostra i 5 titoli di film più popolari."""
    top_movies = dataframe[dataframe['content_category'].str.contains('Movie', na=False)].nlargest(5, 'preferences')
    if not top_movies.empty:
        print("\nTop 5 Film più popolari:")
        for idx, row in top_movies.iterrows():
            print(f"{idx + 1}. {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
    else:
        print("Nessun film trovato.")

def show_top_tv_shows(dataframe):
    """Mostra i 5 titoli di serie TV più popolari."""
    top_tv_shows = dataframe[dataframe['content_category'].str.contains('TV Show', na=False)].nlargest(5, 'preferences')
    if not top_tv_shows.empty:
        print("\nTop 5 Serie TV più popolari:")
        for idx, row in top_tv_shows.iterrows():
            print(f"{idx + 1}. {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
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

    # Restituisce solo i dati senza stampare
    top_recommendations = recommendations.sort_values(by='preferences', ascending=False).head(top_n)
    return top_recommendations
