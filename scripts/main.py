import os
import warnings
import pandas as pd
from preprocess_data import preprocess_data
from create_embedding import create_embeddings_pipeline
from analyze_data import analyze_data
from cross_validation import cross_validate_models
from supervised import supervised_learning
from generate_prolog_files import generate_prolog_files
from search_and_recommendation import map_user_input_to_category, search_title_with_suggestions, recommend_top_titles, show_title_info, get_user_rating, show_statistics_menu

warnings.filterwarnings("ignore", category=RuntimeWarning)
is_classification_done = False


def check_classification_status(baseDir):
    content_embeddings_path = os.path.join(baseDir, '..', 'data', 'content_category_embeddings.npy')
    title_embeddings_path = os.path.join(baseDir, '..', 'data', 'title_embeddings.npy')
    return os.path.exists(content_embeddings_path) and os.path.exists(title_embeddings_path)

def classificazione(baseDir):
    global is_classification_done
    try:
        print_section_header("Classificazione in corso...")
        preprocess_data(baseDir)
        create_embeddings_pipeline(baseDir)

        if check_classification_status(baseDir):
            analyze_data(baseDir)
            supervised_learning(baseDir)
            cross_validate_models(baseDir)
            print_section_header("Classificazione completata.")
            is_classification_done = True
        else:
            raise FileNotFoundError("Uno o più file essenziali per la classificazione non sono stati trovati.")
    except Exception as e:
        print(f"Errore durante la classificazione: {e}")
        is_classification_done = False

def search_and_recommend_titoli_wrapper(baseDir):
    if not is_classification_done:
        print("Errore: La classificazione non è stata eseguita. Esegui prima la classificazione per ricercare e raccomandare titoli.")
        return
    
    df = pd.read_csv(os.path.join(baseDir, '..', 'data', 'processed_data.csv'))
    
    while True:
        try:
            user_input = input("\nVuoi cercare per Titolo, Categoria o tornare al Menu principale? \nScrivi la tua preferenza: ").strip().lower()
            if user_input == "titolo":
                result = search_by_title(df)
                if result == 'menu':
                    return
                elif result == 'ricerca':
                    continue
            elif user_input == "categoria":
                result = search_by_category(df)
                if result == 'menu':
                    return
                elif result == 'ricerca':
                    continue
            elif user_input == "menu":
                return
            else:
                print("\nOpzione non valida. Per favore scegli 'titolo', 'categoria' o 'menu'.")
        except Exception as e:
            print(f"Errore durante la ricerca e raccomandazione: {e}")

def kb(baseDir):
    # Verifica se la classificazione è stata eseguita
    processed_data_path = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    if not os.path.exists(processed_data_path):
        print("Errore: La classificazione non è stata eseguita. Esegui prima la classificazione per generare la KB.")
        return

    print_section_header("Generazione della KB in Prolog...")
    generate_prolog_files(baseDir)
    print_section_header("KB in Prolog generata con successo.")

def search_by_title(df):
    while True:
        user_input = input("\nInserisci un titolo (es. 'Grey's Anatomy', 'The Godfather') o 'back' per tornare alla selezione di ricerca, o 'menu' per tornare al menu principale: ").strip().lower()
        
        if user_input == 'menu':
            return 'menu'
        elif user_input == 'back':
            return 'ricerca'
        
        title_info = search_title_with_suggestions(df, user_input)
        
        if 'exact_match' in title_info:
            show_title_info(title_info['exact_match'].iloc[0])
            rating = get_user_rating(title_info['exact_match'].iloc[0]['title'])
            if rating <= 2:
                preferred_genre = input("Sembra che questo titolo non ti sia piaciuto. Che genere preferisci? ")
                search_by_category(df, map_user_input_to_category(preferred_genre))
            else:
                result = show_statistics_menu(df, os.path.dirname(os.path.abspath(__file__)))
                if result == 'menu':
                    return 'menu'
                elif result == 'ricerca':
                    return 'ricerca'
            break
        elif 'suggestions' in title_info:
            print("\nTitolo non trovato, ma abbiamo trovato questi suggerimenti:")
            for idx, row in title_info['suggestions'].iterrows():
                print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
        else:
            print(f"Nessun risultato trovato per: {user_input}.")
            continue

def search_by_category(df, category=None):
    if category is None:
        user_input = input("\nInserisci una categoria (es. 'Comedy', 'Drama', 'Horror'), 'ricerca' per tornare alla selezione di ricerca, o 'menu' per tornare al menu principale: ").strip().lower()
        
        if user_input == 'menu':
            return 'menu'
        elif user_input == 'ricerca':
            return 'ricerca'
        
        category = map_user_input_to_category(user_input)
    
    category_recommendations = recommend_top_titles(df, content_category=category)
    
    if not category_recommendations.empty:
        print(f"\nTitoli nella categoria '{category}':")
        for idx, row in category_recommendations.iterrows():
            print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
    else:
        print(f"Nessun risultato trovato per la categoria: {category}.")
    
    while True:
        print("\nMenu di Selezione:")
        print("1) Inserire un titolo della lista")
        print("2) Fare una nuova ricerca")
        print("3) Tornare al menu principale")
        title_choice = input("Seleziona un'opzione (1-3): ").strip()

        if title_choice == '1':
            search_by_title(df)
            break
        elif title_choice == '2':
            return 'ricerca'
        elif title_choice == '3':
            return 'menu'
        else:
            print("Opzione non valida. Riprova.")

def display_menu():
    print("\nMenu di Selezione:")
    print("1. Classificazione dei Film e Serie Tv")
    print("2. Ricerca Film o Serie TV")
    print("3. Generazione della KB")
    print("4. Uscita")
    return input("Seleziona un'opzione (1-4): ")

def print_section_header(title):
    print("\n" + "="*50)
    print(f"{title.center(50)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))

    while True:
        try:
            choice = display_menu()
            if choice == '1':
                classificazione(baseDir)
            elif choice == '2':
                search_and_recommend_titoli_wrapper(baseDir)
            elif choice == '3':
                kb(baseDir)
            elif choice == '4':
                print("Grazie per essere stato con noi. Arrivederci.")
                break
            else:
                print("Opzione non valida. Riprova.")
        except KeyboardInterrupt:
            print("\nGrazie per essere stato con noi. Arrivederci.")
            break
