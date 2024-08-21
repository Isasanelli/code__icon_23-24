import os
import pandas as pd
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
        'commedia': 'Comedies',
        'comedy': 'Comedies',
        'documentario': 'Documentaries',
        'documentari': 'Documentaries',
        'docuserie': 'Docuseries',
        'serie tv': 'TV Show',
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

def search_and_recommend(baseDir, user_input=None, top_n=5):
    """
    Funzione principale per gestire la ricerca e le raccomandazioni.
    """
    try:
        filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Il file processed_data.csv non è stato trovato nel percorso: {filepath}")
        
        df = load_processed_data(filepath)

        if user_input:
            mapped_category = map_user_input_to_category(user_input)
            
            category_recommendations = recommend_top_titles(df, content_category=mapped_category)
            
            if isinstance(category_recommendations, pd.DataFrame) and not category_recommendations.empty:
                print(f"\nTitoli nella categoria '{mapped_category}':")
                for idx, row in category_recommendations.iterrows():
                    print(f"- {row['title']}")
                
                selected_title = input("\nSeleziona un titolo dalla lista per ricevere una raccomandazione o inserisci un altro titolo: ")
                selected_title_info = search_title_with_suggestions(df, selected_title)
                
                if 'exact_match' in selected_title_info:
                    title_info = selected_title_info['exact_match'].iloc[0]
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
                            
                            if 'exact_match' in selected_title_info:
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
                    return selected_title_info['exact_match']
                else:
                    print("Il titolo scelto non è stato trovato.")
                    return None
            else:
                print(f"Nessun risultato trovato per la categoria: {mapped_category}.")
                return None

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"Errore durante la ricerca e la raccomandazione: {e}")
        return None

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
        return {"message": f"No recommendations available for the selected category: {content_category}."}
    
    top_recommendations = recommendations.sort_values(by='preferences', ascending=False).head(top_n)
    return top_recommendations

def recommend_popular(dataframe, content_category=None):
    """
    Raccomanda i titoli più popolari per film e serie TV in base a 'preferences'.
    """
    print("\nFilm più popolari:")
    popular_movies = recommend_top_titles(dataframe, content_category="Movie", top_n=5)
    for title in popular_movies['title']:
        print(f"- {title}")
    
    print("\nSerie TV più popolari:")
    popular_tv_shows = recommend_top_titles(dataframe, content_category="TV Show", top_n=5)
    for title in popular_tv_shows['title']:
        print(f"- {title}")
