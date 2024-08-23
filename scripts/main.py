import os
import warnings
import pandas as pd
from preprocess_data import preprocess_data
from create_embedding import create_embeddings_pipeline
from analyze_data import analyze_data
from cross_validation import cross_validate_models
from supervised import supervised_learning
from generate_prolog_files import generate_prolog_files
from search_and_recommendation import map_user_input_to_category, recommend_popular, search_title_with_suggestions, recommend_top_titles, show_title_info, get_user_rating, show_statistics_menu

warnings.filterwarnings("ignore", category=RuntimeWarning)
is_classification_done = False

def check_classification_status(baseDir):
    """Verifica se la classificazione è stata eseguita controllando la presenza dei file essenziali."""
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

def kb(baseDir):
    # Verifica se la classificazione è stata eseguita
    processed_data_path = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    if not os.path.exists(processed_data_path):
        print("\nErrore: La classificazione non è stata eseguita. Esegui prima la classificazione per generare la KB.")
        return

    print_section_header("Generazione della KB in Prolog...")
    generate_prolog_files(baseDir)
    print_section_header("KB in Prolog generata con successo.")

def search_and_recommend_titoli_wrapper(baseDir):
    # Verifica se la classificazione è stata eseguita
    df_path = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    if not os.path.exists(df_path):
        print("\nErrore: La classificazione non è stata eseguita. \nEsegui prima la classificazione per avere accesso alla funzionalità di ricerca e raccomandazione.")
        return 'menu' 
    
    # Carica il DataFrame
    df = pd.read_csv(df_path)
    
    while True:
        try:
            print("\n" + "="*50)
            print(" Menù di Ricerca")
            print("-"*50)
            print(" 1. Cerca per Titolo")
            print(" 2. Cerca per Categoria")
            print(" 3. Torna al Menu principale")
            print("="*50)
            user_input = input(" Scrivi il numero della tua preferenza: ").strip()

            if user_input == "1":
                result = search_by_title(df)
            elif user_input == "2":
                result = search_by_category(df)
            elif user_input == "3":
                return 'menu'
            else:
                print("\n Opzione non valida. Per favore scegli 1, 2 o 3.")
                continue
            
            if result == 'menu':
                return
            elif result == 'ricerca':
                continue
        except Exception as e:
            print(f"\nErrore durante la ricerca e raccomandazione: {e}")
        return 'menu'



def search_by_title(df):
    while True:
        print("\n" + "="*50)
        print (" Puoi: ")
        print("-"*50)
        print(" • Scrivere un titolo come nell'esempio --> (es. 'Grey's Anatomy', 'Breaking Bad', 'Game of Thrones')")
        print("")
        print("-"*20 + "Oppure:"+ "-"*20)
        print("")
        print(" • Scrivere 'ricerca' per tornare alla selezione di ricerca")
        print("")
        print(" • Scrivere 'menu' per tornare al menu principale")
        print("="*50)
        user_input = input(" >> ").strip().lower() 

        if user_input == 'menu':
            return 'menu'
        elif user_input == 'ricerca':
            return 'ricerca'
        
        title_info = search_title_with_suggestions(df, user_input)
        
        if 'exact_match' in title_info:
            show_title_info(title_info['exact_match'].iloc[0])
            rating = get_user_rating(title_info['exact_match'].iloc[0]['title'])
            if rating <= 2:
                preferred_genre = input("\n Sembra che questo titolo non ti sia piaciuto. Che genere preferisci? ")
                search_by_category(df, map_user_input_to_category(preferred_genre))
            else:
                # Se il rating è da 3 a 5, raccomanda titoli simili
                recommend_popular(df, title_info['exact_match'].iloc[0]['content_category'], exclude_title=title_info['exact_match'].iloc[0]['title'])
                user_input = show_statistics_menu(df, os.path.dirname(os.path.abspath(__file__)))
                if user_input == 'menu':
                    return 'menu'  # Torna al menu principale
                elif user_input == 'ricerca':
                    return 'ricerca' 
            break
        elif 'suggestions' in title_info:
            print("\n Titolo non trovato, ma abbiamo trovato questi suggerimenti:")
            for row in title_info['suggestions'].iterrows():
                print(f"- {row['title']} ({row['content_category']}, Preferenze: {row['preferences']})")
        else:
            print(f"\n Nessun risultato trovato per: {user_input}.")
            break
        
def search_by_category(df, category=None):
    if category is None:
        print("\n" + "="*50)
        print (" Puoi: ")
        print("-"*50)
        print(" • Scrivere una categoria come nell'esempio --> (es. 'Comedy', 'Drama', 'Horror')")
        print("")
        print("-"*20 + "Oppure:"+ "-"*20)
        print("")
        print(" • Scrivere 'ricerca' per tornare alla selezione di ricerca")
        print("")
        print(" • Scrivere 'menu' per tornare al menu principale")
        print("="*50)
        user_input = input(" >> ").strip().lower()
        
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
        print(f"\nNessun risultato trovato per la categoria: {category}.")
    
    while True:
        print("\n" + "="*50)
        print(" Menu di Selezione")
        print("-"*50)
        print(" 1) Scrivere un titolo ")
        print(" 2) Fare una nuova ricerca")
        print(" 3) Tornare al menu principale")
        print("="*50)
        title_choice = input(" Seleziona un'opzione (1-3): ").strip()

        if title_choice == '1':
            search_by_title(df)
            break
        elif title_choice == '2':
            return 'ricerca'
        elif title_choice == '3':
            return 'menu'
        else:
            print("\n Opzione non valida. Riprova.")


def display_menu():
    print("\n" + "="*50)
    print(" Menu di Selezione")
    print("-"*50)
    print(" 1. Classificazione dei Film e Serie Tv")
    print(" 2. Ricerca Film o Serie TV")
    print(" 3. Generazione della KB")
    print(" 4. Uscita")
    print("="*50)
    return input(" Seleziona un'opzione (1-4): ")

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
                result = search_and_recommend_titoli_wrapper(baseDir)
                if result == 'menu':
                    continue 
            elif choice == '3':
                kb(baseDir)
            elif choice == '4':
                print("\nGrazie per essere stato con noi. Arrivederci.")
                break
            else:
                print("\nOpzione non valida. Riprova.")
        except KeyboardInterrupt:
            print("\nGrazie per essere stato con noi. Arrivederci.")
            break
