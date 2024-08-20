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
from raccomandazione import raccomandazione  # Importiamo la nuova funzione raccomandazione

# Variabile di stato per controllare se la classificazione è stata eseguita
is_classification_done = False 

def classificazione(baseDir):
    global is_classification_done
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
        perform_clustering(baseDir)
        supervised_learning(baseDir)
        cross_validate_models(baseDir)
        print("Classificazione completata.")
        is_classification_done = True  # Imposta la variabile a True dopo il completamento
    except Exception as e:
        print(f"Errore durante la classificazione: {e}")
        is_classification_done = False  # Assicurarsi che sia False se si verifica un errore

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

def raccomandazione(baseDir):
    if not is_classification_done:
        print("Errore: Esegui prima la classificazione.")
        return
    try:
        print("Inizio del sistema di raccomandazione...")
        filepath = os.path.join(baseDir, 'data', 'processed_data.csv')
        df = pd.read_csv(filepath)

        user_liked_category = 'Action - International Movies'  # Questo potrebbe essere dinamico

        recommendations = df[df['content_category'].str.contains(user_liked_category, na=False)]
        
        if recommendations.empty:
            print("Nessuna raccomandazione disponibile per la categoria selezionata.")
        else:
            print(f"Raccomandazioni per la categoria '{user_liked_category}':")
            for title in recommendations['title'].head(10):
                print(f"- {title}")

        print("Sistema di raccomandazione completato.")
    except Exception as e:
        print(f"Errore durante il processo di raccomandazione: {e}")

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
            raccomandazione(baseDir)
        elif choice == '3':
            kb(baseDir)
        elif choice == '4':
            apprendimento_probabilistico(baseDir)
        elif choice == '5':
            print("Uscita dal programma.")
            break
        else:
            print("Opzione non valida. Riprova.")
