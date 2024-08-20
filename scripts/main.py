import os
from preprocess_data_dataset import preprocess_data
from create_embedding import create_embeddings_pipeline
from analyze_data import analyze_data
from cross_validation import cross_validate_models
from supervised import supervised_learning
from generate_prolog_files import generate_prolog_files
from probabilistic_learning import probabilistic_learning
from raccomandazione import raccomandazione  # Importiamo la nuova funzione raccomandazione

def classificazione(baseDir):
    try:
        print("Inizio della pipeline per la classificazione...")
        preprocess_data(baseDir)
        create_embeddings_pipeline(baseDir)

        # Verifica se gli embedding sono stati creati correttamente
        content_embeddings_path = os.path.join(baseDir, '..', 'data', 'content_category_embeddings.npy')
        title_embeddings_path = os.path.join(baseDir, '..', 'data', 'title_embeddings.npy')
        
        if not os.path.exists(content_embeddings_path) or not os.path.exists(title_embeddings_path):
            raise FileNotFoundError(f"Uno o più file di embedding non sono stati trovati: {content_embeddings_path}, {title_embeddings_path}")
        
        analyze_data(baseDir)
        supervised_learning(baseDir)
        cross_validate_models(baseDir)
        print("Classificazione completata.")
    except Exception as e:
        print(f"Errore durante la classificazione: {e}")


def kb(baseDir):
    try:
        print("Inizio della pipeline per la generazione della KB...")
        preprocess_data(baseDir)
        create_embeddings_pipeline(baseDir)
        analyze_data(baseDir)
        generate_prolog_files(baseDir)
        print("KB generata con successo.")
    except Exception as e:
        print(f"Errore durante la generazione della KB: {e}")

def apprendimento_probabilistico(baseDir):
    try:
        print("Inizio della pipeline per l'apprendimento probabilistico...")
        preprocess_data(baseDir)
        create_embeddings_pipeline(baseDir)
        analyze_data(baseDir)
        probabilistic_learning(baseDir)
        print("Apprendimento probabilistico completato.")
    except Exception as e:
        print(f"Errore durante l'apprendimento probabilistico: {e}")

def display_menu():
    print("\nMenu di Selezione:")
    print("1. Classificazione")
    print("2. Raccomandazione")
    print("3. Generazione della KB")
    print("4. Apprendimento Probabilistico")
    print("5. Uscita")
    return input("Seleziona un'opzione (1-5): ")

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        choice = display_menu()
        if choice == '1':
            classificazione(baseDir)
        elif choice == '2':
            raccomandazione(baseDir)  # Richiama la raccomandazione dal nuovo file
        elif choice == '3':
            kb(baseDir)
        elif choice == '4':
            apprendimento_probabilistico(baseDir)
        elif choice == '5':
            print("Uscita dal programma.")
            break
        else:
            print("Opzione non valida. Riprova.")
