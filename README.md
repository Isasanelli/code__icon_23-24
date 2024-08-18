# Progetto di Analisi e Classificazione di Film e Serie TV su Amazon Prime

## Descrizione del Progetto

Questo progetto è stato sviluppato per analizzare un dataset di film e serie TV disponibili su Amazon Prime. Il dataset comprende informazioni come cast, registi, valutazioni, anno di rilascio e durata. Gli obiettivi principali del progetto includono:

## Descrizione dei File Principali

1. **analyze_data.py** :
    Questo script analizza i dati provenienti dal dataset preprocessato. Genera visualizzazioni che mostrano la distribuzione dei film e delle serie TV per categoria, anno di rilascio, ecc.

2. **clustering.py**:
    Questo script esegue il clustering sui film e sulle serie TV basandosi sugli embeddings generati. Il clustering è utile per raggruppare contenuti simili.

3. **create_embedding.py**:
    Questo script genera gli embeddings per i titoli e le categorie di contenuto utilizzando spaCy. Gli embeddings sono poi utilizzati per il clustering e altre analisi.

4. **cross_validation.py**:
    Esegue la validazione incrociata per valutare le prestazioni dei modelli di classificazione (es. Random Forest, Naive Bayes). I risultati della cross-validation sono salvati come file CSV.

5. **generate_prolog_files.py**:
    Genera i file Prolog che contengono i fatti e le regole della base di conoscenza. Questo include sia i fatti relativi alla classificazione sia quelli per le raccomandazioni.

6. **kb.pl**:
    Questo file rappresenta la base di conoscenza principale in Prolog. Contiene tutte le regole e i fatti necessari per effettuare ragionamenti logici, sia per la classificazione che per le raccomandazioni.

7. **preprocess_data_dataset.py**:
    Questo script esegue il preprocessing del dataset originale di Netflix. Pulisce i dati, gestisce i valori mancanti e crea nuove feature (es. lunghezza del titolo, mese e stagione di rilascio).

8. **probabilistic_learning.py**:
    Implementa un modello probabilistico utilizzando Random Forest. Il modello viene valutato e genera una curva ROC per visualizzare le performance.

9. **supervised.py**:
    Contiene l'implementazione di diversi modelli di classificazione supervisionata (es. Decision Tree, Random Forest, AdaBoost, Naive Bayes, K-NN). Genera report di classificazione e curve di apprendimento.

## Struttura del Progetto

Il progetto è organizzato nelle seguenti directory:

```
CODE_ICON_23-24/
│
├── .venv/
│
├── data/
│   ├── content_category_embeddings.npy
│   ├── netflix_titles.csv
│   ├── processed_data.csv
│   ├── title_embeddings.npy
│
├── results/
│   ├── knowledge_base/
│   │   ├── classification.pl
│   │   ├── recommendation.pl
│   ├── models/
│   │   ├── clustering/
│   │   ├── cross_validation/
│   │   ├── probabilistic_learning/
│   │   ├── supervised/
│   ├── visualizations/
│   │   ├── analyze_data/
│   │   ├── clustering/
│   │   ├── probabilistic_learning/
│   │   ├── supervised/
│
├── scripts/
│   ├── analyze_data.py
│   ├── clustering.py
│   ├── create_embedding.py
│   ├── cross_validation.py
│   ├── generate_prolog_files.py
│   ├── kb.pl
│   ├── preprocess_data_dataset.py
│   ├── probabilistic_learning.py
│   ├── supervised.py
│
├── .gitignore
├── README.md
├── requirements.txt


```

## Passi per Eseguire il Codice

1. **Impostare l'ambiente**
    
    ```bash
    python -m venv .venv
    .venv\\Scripts\\activate
    
    ```
    
2. **Installare le Librerie Necessarie**
    - Usa il file `requirements.txt` per installare tutte le librerie necessarie:
        
        ```bash
        pip install -r requirements.txt
        
        ```
        
    - Dopo aver installato `spacy`, scarica il modello di lingua inglese:
        
        ```bash
        python -m spacy download en_core_web_sm
        
        ```
        

## Esecuzione del Progetto

1. **Preprocessamento dei Dati:** Avviare `preprocess_prime_dataset.py` per caricare e pulire il dataset.
2. **Analisi dei Dati:** Eseguire `analyze_data.py` per generare visualizzazioni esplorative.
3. **Clustering:** Avviare `clustering.py` per eseguire il clustering dei film e delle serie TV.
4. **Apprendimento Supervisionato:** Eseguire `supervised.py` per addestrare e valutare un modello supervisionato.
5. **Apprendimento Probabilistico:** Avviare `probabilistic_learning.py` per addestrare un modello probabilistico e generare la curva ROC.
6. **Cross-Validation:** Eseguire `cross_validation.py` per valutare le performance dei modelli con la validazione incrociata.
7. **Generazione di Fatti Prolog:** Avviare `generate_prolog_files.py` per creare fatti Prolog basati sui dati.
8. **Creazione della Knowledge Base:** Eseguire `knowledge_base.py` per generare l'ontologia RDF/OWL.

## Esecuzione degli Script

Per eseguire le componenti del progetto in modo controllato, eseguire i seguenti script nell'ordine indicato:

```bash
python scripts/preprocess_prime_dataset.py
python scripts/create_embedding.py
python scripts/analyze_data.py
python scripts/clustering.py
python scripts/supervised.py
python scripts/probabilistic_learning.py
python scripts/cross_validation.py
python scripts/generate_prolog_files.py
python scripts/knowledge_base.py

```

## Librerie Richieste

Il progetto utilizza le seguenti librerie:

- pandas
- numpy
- spacy
- tensorflow
- nltk
- scikit-learn
- matplotlib
- rdflib
- pyswip
- pgmpy
- SPARQLWrapper
- openpyxl
- wordcloud
- seaborn
- imbalanced-learn

Le librerie possono essere installate utilizzando il file `requirements.txt`.

## Conclusioni

Questo progetto rappresenta un approccio integrato per l'analisi e la modellazione dei dati di Amazon Prime. Combina tecniche di machine learning con sistemi basati sulla conoscenza, inclusa la creazione di ontologie e inferenze basate su Prolog. L'obiettivo finale è creare un sistema robusto per classificare e prevedere vari attributi relativi ai titoli disponibili su Amazon Prime.