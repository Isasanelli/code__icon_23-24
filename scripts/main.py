import os
import pandas as pd
from preprocess_data_dataset import preprocess_data
from create_embedding import create_embeddings_pipeline
from analyze_data import analyze_data
from clustering import perform_clustering
from cross_validation import cross_validate_models
from supervised import supervised_learning
from generate_prolog_files import generate_prolog_files
from probabilistic_learning import probabilistic_learning
from search_and_recommendation import search_and_recommend

# Variabile di stato per controllare se la classificazione è stata eseguita
is_classification_done = False 

def check_classification_status(baseDir):
    """Verifica se la classificazione è stata eseguita controllando la presenza dei file essenziali."""
    content_embeddings_path = os.path.join(baseDir, '..', 'data', 'content_category_embeddings.npy')
    title_embeddings_path = os.path.join(baseDir, '..', 'data', 'title_embeddings.npy')
    
    if os.path.exists(content_embeddings_path) and os.path.exists(title_embeddings_path):
        return True
    else:
        return False

def classificazione(baseDir):
    global is_classification_done
    try:
        print("Inizio della pipeline per la classificazione...")
        preprocess_data(baseDir)
        create_embeddings_pipeline(baseDir)

        # Verifica se gli embedding sono stati creati correttamente
        if check_classification_status(baseDir):
            analyze_data(baseDir)
            perform_clustering(baseDir)
            supervised_learning(baseDir)
            cross_validate_models(baseDir)
            print("Classificazione completata.")
            is_classification_done = True  # Imposta la variabile a True dopo il completamento
        else:
            raise FileNotFoundError("Uno o più file essenziali per la classificazione non sono stati trovati.")
    except Exception as e:
        print(f"Errore durante la classificazione: {e}")
        is_classification_done = False  # Assicurarsi che sia False se si verifica un errore

def search_and_recommend_titoli(baseDir):
    if not is_classification_done:
        print("Errore: Esegui prima la classificazione.")
        return
    try:
        # Unifica l'input per titolo e categoria
        user_input = input("Inserisci un titolo o una categoria (es. 'Breaking Bad' o 'Comedy'): ")
        if user_input:
            search_and_recommend(baseDir, user_input=user_input)
        else:
            print("Nessun input fornito. Inserisci un titolo o una categoria per effettuare la ricerca.")
    except Exception as e:
        print(f"Errore durante la ricerca e raccomandazione: {e}")

def kb(baseDir):
    if not is_classification_done:
        print("Errore: Esegui prima la classificazione.")
        return
    try:
        print("Inizio della pipeline per la generazione della KB...")
        generate_prolog_files(baseDir)
        print("KB generata con successo.")
    except Exception as e:
        print(f"Errore durante la generazione della KB: {e}")

def apprendimento_probabilistico(baseDir):
    if not is_classification_done:
        print("Errore: Esegui prima la classificazione.")
        return
    try:
        print("Inizio della pipeline per l'apprendimento probabilistico...")
        probabilistic_learning(baseDir)
        print("Apprendimento probabilistico completato.")
    except Exception as e:
        print(f"Errore durante l'apprendimento probabilistico: {e}")

def display_menu():
    print("\nMenu di Selezione:")
    print("1. Classificazione dei Film e Serie Tv")
    print("2. Ricerca Film o Serie TV")
    print("3. Generazione della KB")
    print("4. Apprendimento Probabilistico")
    print("5. Uscita")
    return input("Seleziona un'opzione (1-5): ")

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))
    
    # Verifica lo stato della classificazione all'avvio del programma
    is_classification_done = check_classification_status(baseDir)
    
    while True:
        try:
            choice = display_menu()
            if choice == '1':
                classificazione(baseDir)
            elif choice == '2':
                search_and_recommend_titoli(baseDir)
            elif choice == '3':
                kb(baseDir)
            elif choice == '4':
                apprendimento_probabilistico(baseDir)
            elif choice == '5':
                print("Grazie per essere stato con noi. Arrivederci.")
                break
            else:
                print("Opzione non valida. Riprova.")
        except KeyboardInterrupt:
            print("\nGrazie per essere stato con noi. Arrivederci.")
            break
