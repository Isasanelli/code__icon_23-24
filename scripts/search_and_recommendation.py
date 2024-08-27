import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import matplotlib.pyplot as plt


def load_processed_data(filepath):
    """Carica i dati preprocessati dal file CSV."""
    return pd.read_csv(filepath)

def load_embeddings(filepath):
    """Carica gli embeddings dal file npy."""
    return np.load(filepath)

def content_based_filtering(selected_titles, title_embeddings, df, n_recommendations=5):
    """Genera raccomandazioni utilizzando il content-based filtering."""
    selected_indices = df[df['title'].isin(selected_titles)].index
    
    if len(selected_indices) == 0:
        print("Nessun titolo selezionato è stato trovato nel dataset.")
        return []
    
    user_profile = title_embeddings[selected_indices].mean(axis=0)
    similarities = cosine_similarity([user_profile], title_embeddings).flatten()
    
    recommendations = pd.Series(similarities, index=df['title'])
    
    recommendations = recommendations[~recommendations.index.isin(selected_titles)]
    recommendations = recommendations.sort_values(ascending=False).head(n_recommendations)
    
    return recommendations.index.tolist()

def user_rating_flow_with_decision(df, recommendations):
    """Permette all'utente di valutare le raccomandazioni e adatta il flusso di raccomandazione."""
    for title in recommendations:
        print(f"\nValuta il titolo '{title}':")
        entertainment_rating = get_user_rating(title, "Intrattenimento")

        # Se il rating è basso, suggerisci un genere diverso
        if entertainment_rating <= 2:
            print(f"Sembra che '{title}' non ti abbia intrattenuto molto. Vuoi esplorare un genere diverso?")
            change_genre = input("Scrivi 'sì' per cambiare genere, altrimenti premi invio: ").strip().lower()
            if change_genre == 'sì':
                new_genre = input("Inserisci il nuovo genere che vuoi esplorare: ").strip().lower()
                new_genre = map_user_input_to_category(new_genre)  # Mappa l'input dell'utente a categorie riconosciute
                new_recommendations = recommend_top_titles(df, content_category=new_genre)
                print(f"Ecco alcune raccomandazioni nel genere '{new_genre}':")
                print(new_recommendations['title'].tolist())
                return new_recommendations['title'].tolist()  # Esci dal ciclo e mostra nuove raccomandazioni
        else:
            # Salva la valutazione dell'utente
            df.loc[df['title'] == title, 'user_rating'] = entertainment_rating
            print(f"Grazie per la tua valutazione di '{title}'!")

def search_and_recommend(df, title_embeddings, n_recommendations=5):
    """Ricerca un titolo o categoria e genera raccomandazioni."""
    while True:
        print("\n" + "="*60)
        print(" Cosa vuoi fare?")
        print("="*60)
        print(" 1. Cerca per Titolo")
        print(" 2. Cerca per Categoria")
        print(" 3. Torna al Menu principale")
        print("="*60)
        user_input = input(" Seleziona un'opzione (1-3): ").strip().lower()
        
        if user_input == '3':
            return 'menu'
        elif user_input == '1':
            title_input = input(" Inserisci il titolo (es. 'Grey's Anatomy'): ").strip().lower()
            title_info = search_title_with_suggestions(df, title_input)
            
            if 'exact_match' in title_info:
                exact_match = title_info['exact_match']
                show_title_info(exact_match.iloc[0])
                
                # Valutazione del titolo selezionato
                entertainment_rating = get_user_rating(exact_match.iloc[0]['title'], "Intrattenimento")
                df.loc[df['title'] == exact_match.iloc[0]['title'], 'user_rating'] = entertainment_rating

                if entertainment_rating <= 2:
                    # Se la valutazione è 1 o 2, chiedi il genere preferito e raccomanda
                    preferred_genre = input("Sembra che questo titolo non ti sia piaciuto. Che genere preferisci? ").strip().lower()
                    category = map_user_input_to_category(preferred_genre)
                    recommendations = recommend_top_titles(df, content_category=category)
                    
                    if not recommendations.empty:
                        print(f"\nEcco alcune raccomandazioni nel genere '{category}':")
                        for idx, row in recommendations.iterrows():
                            print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
                    else:
                        print(f"\nNessun risultato trovato per la categoria: {category}.")
                
                else:
                    # Se la valutazione è 3 o superiore, mostra le raccomandazioni basate sul titolo selezionato
                    recommendations = content_based_filtering([exact_match.iloc[0]['title']], title_embeddings, df, n_recommendations)
                    
                    print(f"\nRaccomandazioni basate su '{exact_match.iloc[0]['title']}':")
                    for rec_title in recommendations:
                        rec_info = df[df['title'] == rec_title].iloc[0]
                        print(f"- {rec_title} ({rec_info['content_category']}, Preferenze: {rec_info['preferences']})")
                
                return post_search_menu(df, title_embeddings)
                
            elif 'suggestions' in title_info:
                print("\nTitolo non trovato, ma abbiamo trovato questi suggerimenti:")
                for idx, row in title_info['suggestions'].iterrows():
                    print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
                
                selected_title = input("\nInserisci solo il titolo presente sulla lista e non la categoria e preferenza  o 'ricerca' per tornare indietro: ").strip().lower()
                if selected_title == 'ricerca':
                    continue
                
                exact_match = df[df['title'].str.lower() == selected_title]
                if not exact_match.empty:
                    show_title_info(exact_match.iloc[0])
                    
                    # Valutazione del titolo selezionato
                    entertainment_rating = get_user_rating(exact_match.iloc[0]['title'], "Intrattenimento")
                    df.loc[df['title'] == exact_match.iloc[0]['title'], 'user_rating'] = entertainment_rating

                    if entertainment_rating <= 2:
                        # Se la valutazione è 1 o 2, chiedi il genere preferito e raccomanda
                        preferred_genre = input("Sembra che questo titolo non ti sia piaciuto. Che genere preferisci? ").strip().lower()
                        category = map_user_input_to_category(preferred_genre)
                        recommendations = recommend_top_titles(df, content_category=category)
                        
                        if not recommendations.empty:
                            print(f"\nEcco alcune raccomandazioni nel genere '{category}':")
                            for idx, row in recommendations.iterrows():
                                print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
                        else:
                            print(f"\nNessun risultato trovato per la categoria: {category}.")
                    
                    else:
                        # Se la valutazione è 3 o superiore, mostra le raccomandazioni basate sul titolo selezionato
                        recommendations = content_based_filtering([exact_match.iloc[0]['title']], title_embeddings, df, n_recommendations)
                        
                        print(f"\nRaccomandazioni basate su '{exact_match.iloc[0]['title']}':")
                        for rec_title in recommendations:
                            rec_info = df[df['title'] == rec_title].iloc[0]
                            print(f"- {rec_title} ({rec_info['content_category']}, Preferenze: {rec_info['preferences']})")
                    
                    return post_search_menu(df, title_embeddings)
                else:
                    print("\nTitolo non trovato nella lista suggerita. Riprova.")
                    continue
        elif user_input == '2':
            category_input = input(" Inserisci la categoria (es. 'Comedy', 'Drama'): ").strip().lower()
            category = map_user_input_to_category(category_input)
            category_recommendations = recommend_top_titles(df, content_category=category)
            
            if category_recommendations.empty:
                print(f"\nNessun risultato trovato per la categoria: {category}.")
                continue
            
            print(f"\nTitoli nella categoria '{category}':")
            for idx, row in category_recommendations.iterrows():
                print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
            
            selected_title = input("\nInserisci solo il titolo presente sulla lista e non la categoria e preferenza  o 'ricerca' per tornare indietro: ").strip().lower()
            if selected_title == 'ricerca':
                continue
            
            exact_match = df[df['title'].str.lower() == selected_title]
            if not exact_match.empty:
                show_title_info(exact_match.iloc[0])
                
                # Valutazione del titolo selezionato
                entertainment_rating = get_user_rating(exact_match.iloc[0]['title'], "Intrattenimento")
                df.loc[df['title'] == exact_match.iloc[0]['title'], 'user_rating'] = entertainment_rating

                if entertainment_rating <= 2:
                    # Se la valutazione è 1 o 2, chiedi il genere preferito e raccomanda
                    preferred_genre = input("Sembra che questo titolo non ti sia piaciuto. Che genere preferisci? ").strip().lower()
                    category = map_user_input_to_category(preferred_genre)
                    recommendations = recommend_top_titles(df, content_category=category)
                    
                    if not recommendations.empty:
                        print(f"\nEcco alcune raccomandazioni nel genere '{category}':")
                        for idx, row in recommendations.iterrows():
                            print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
                    else:
                        print(f"\nNessun risultato trovato per la categoria: {category}.")
                
                else:
                    # Se la valutazione è 3 o superiore, mostra le raccomandazioni basate sul titolo selezionato
                    recommendations = content_based_filtering([exact_match.iloc[0]['title']], title_embeddings, df, n_recommendations)
                    
                    print(f"\nRaccomandazioni basate su '{exact_match.iloc[0]['title']}':")
                    for rec_title in recommendations:
                        rec_info = df[df['title'] == rec_title].iloc[0]
                        print(f"- {rec_title} ({rec_info['content_category']}, Preferenze: {rec_info['preferences']})")
                
                return post_search_menu(df, title_embeddings)
            else:
                print("\nTitolo non trovato nella lista. Riprova.")
                continue
        else:
            print("\nOpzione non valida. Per favore, seleziona 1, 2 o 3.")


def post_search_menu(df, title_embeddings):
    """Menu dopo la ricerca con opzioni per una nuova ricerca o tornare al menu principale."""
    while True:
        print("\n" + "="*60)
        print(" Cosa vuoi fare adesso?")
        print("="*60)
        print(" 1. Fare una nuova ricerca")
        print(" 2. Torna al menu principale")
        print("="*60)
        
        choice = input(" Seleziona un'opzione (1-2): ").strip()
        
        if choice == '1':
            return search_and_recommend(df, title_embeddings)
        elif choice == '2':
            return 'menu'
        else:
            print("\nOpzione non valida. Per favore, seleziona 1 o 2.")


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

def user_rating_flow(df, recommendations):
    """Permette all'utente di valutare le raccomandazioni."""
    for title in recommendations:
        rating = get_user_rating(title)
        df.loc[df['title'] == title, 'user_rating'] = rating

def get_user_rating(title, dimension="Valutazione"):
    """Chiede all'utente di valutare un titolo su una dimensione specifica."""
    while True:
        try:
            rating = int(input(f"Come valuti il titolo '{title}' per '{dimension}' su una scala da 1 a 5? "))
            if 1 <= rating <= 5:
                return rating
            else:
                print("\nPer favore, inserisci un numero da 1 a 5.")
        except ValueError:
            print("\nInput non valido. Inserisci un numero da 1 a 5.")


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
        'stand-up': 'Stand-Up Comedy',
        'stand-up comedy': 'Stand-Up Comedy',
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
        'musical': 'Music & Musicals',
        'musica': 'Music & Musicals',
    }

    user_input = user_input.lower().strip()
    
    for key in category_mapping:
        if key in user_input:
            return category_mapping[key]
    
    return user_input

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


def show_top_movies(dataframe):
    """Mostra i 5 titoli di film più popolari con preferenze variabili e genera un grafico a torta."""
    top_movies = dataframe[dataframe['content_category'].str.contains('Movie', na=False)].nlargest(5, 'preferences')
    
    # Genera preferenze casuali tra 60 e 100
    top_movies['preferences'] = np.random.randint(60, 101, size=top_movies.shape[0])
    
    if not top_movies.empty:
        print("\nTop 5 Film con preferenze variabili:")
        for idx, row in top_movies.iterrows():
            print(f"{idx + 1}. {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
        
        # Genera il grafico a torta per i film selezionati
        plt.figure(figsize=(8, 6))
        plt.pie(top_movies['preferences'], labels=top_movies['title'], autopct='%1.1f%%', startangle=140)
        plt.title('Top 5 Film più popolari')
        
        # Salva il grafico prima di mostrarlo
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'visualizations', 'statistic_recommander')
        save_path = os.path.join(save_dir, 'top_movies_pie.png')
        plt.savefig(save_path)
        print(f"Grafico salvato in: {save_path}")
        
        plt.show()
        plt.close()
        
        
def show_top_tv_shows(dataframe):
    """Mostra 5 titoli di serie TV con preferenze variabili e genera un grafico a torta."""
    top_tv_shows = dataframe[dataframe['content_category'].str.contains('TV Show', na=False)].nlargest(5, 'preferences')
    
    # Genera preferenze casuali tra 60 e 100
    top_tv_shows['preferences'] = np.random.randint(60, 101, size=top_tv_shows.shape[0])
    
    if not top_tv_shows.empty:
        print("\nTop 5 Serie TV con preferenze variabili:")
        for idx, row in top_tv_shows.iterrows():
            print(f"{idx + 1}. {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
        
        # Genera il grafico a torta per le serie TV selezionate
        plt.figure(figsize=(8, 6))
        plt.pie(top_tv_shows['preferences'], labels=top_tv_shows['title'], autopct='%1.1f%%', startangle=140, colors=plt.cm.viridis(np.linspace(0, 1, 5)))
        plt.title('Top 5 Serie TV con preferenze variabili')

        # Salva il grafico prima di mostrarlo
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'visualizations', 'statistic_recommander')
        save_path = os.path.join(save_dir, 'top_tv_shows_pie.png')
        plt.savefig(save_path)
        print(f"Grafico salvato in: {save_path}")

        plt.show()
        plt.close()


