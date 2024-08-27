import os
import warnings
import pandas as pd
from preprocess_data import preprocess_data
from create_embedding import create_embeddings_pipeline
from analyze_data import analyze_data
from cross_validation import cross_validate_models
from supervised import supervised_learning
from generate_prolog_files import generate_prolog_files
from search_and_recommendation import search_and_recommend, load_processed_data, load_embeddings, show_top_movies, show_top_tv_shows 

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
    """Genera la Knowledge Base in Prolog."""
    processed_data_path = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    
    if not os.path.exists(processed_data_path):
        print("\nErrore: La classificazione non è stata eseguita. Esegui prima la classificazione per generare la KB.")
        return

    print_section_header("Generazione della KB in Prolog...")
    
    # Chiamata per generare i file della Knowledge Base
    generate_prolog_files(baseDir)
    
    
    print_section_header("KB in Prolog generata con successo.")


def visualizza_titoli_piu_popolari(df, baseDir):
    """Visualizza i titoli più popolari e genera grafici associati."""
    show_top_movies(df)
    show_top_tv_shows(df)


def display_menu():
    """Mostra il menu principale."""
    print("\n" + "="*50)
    print(" Menu di Selezione")
    print("-"*50)
    print(" 1. Classificazione dei Film e Serie Tv")
    print(" 2. Ricerca e Raccomandazioni")
    print(" 3. Visualizza Titoli Più Popolari")
    print(" 4. Generazione della KB")
    print(" 5. Uscita")
    print("="*50)
    return input(" Seleziona un'opzione (1-5): ")

def print_section_header(title):
    """Stampa l'intestazione di una sezione."""
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
                df = load_processed_data(os.path.join(baseDir, '..', 'data', 'processed_data.csv'))
                title_embeddings = load_embeddings(os.path.join(baseDir, '..', 'data', 'title_embeddings.npy'))
                result = search_and_recommend(df, title_embeddings)
                if result == 'menu':
                    continue  # Torna al display_menu
            elif choice == '3':
                df = load_processed_data(os.path.join(baseDir, '..', 'data', 'processed_data.csv'))
                visualizza_titoli_piu_popolari(df, baseDir)
            elif choice == '4':
                kb(baseDir)
            elif choice == '5':
                print("\nGrazie per essere stato con noi. Arrivederci.")
                break
            else:
                print("\nOpzione non valida. Riprova.")
        except KeyboardInterrupt:
            print("\nGrazie per essere stato con noi. Arrivederci.")
            break
        except Exception as e:
            print(f"Errore: {e}")
            break
