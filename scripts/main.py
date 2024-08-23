import os
import warnings
import pandas as pd
from preprocess_data import preprocess_data
from create_embedding import create_embeddings_pipeline
from analyze_data import analyze_data
from cross_validation import cross_validate_models

from supervised import supervised_learning
from generate_prolog_files import generate_prolog_files
from search_and_recommendation import search_by_category, search_by_title

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
    df_path = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    if not os.path.exists(df_path):
        print("\nErrore: La classificazione non è stata eseguita.")
        return 'menu'  # Torna al menu principale

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
                display_menu()
                return 'menu'  # Torna al menu principale
            else:
                print("\nOpzione non valida. Per favore scegli 1, 2 o 3.")
                continue

            if result == 'menu':
                return 'menu'  # Torna al menu principale
            elif result == 'ricerca':
                continue
        except Exception as e:
            print(f"\nErrore durante la ricerca e raccomandazione: {e}")
            return 'menu'  # Torna al menu principale



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
                    continue  # Torna al display_menu
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
