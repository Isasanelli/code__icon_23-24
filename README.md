# Progetto di Analisi e Classificazione di Film e Serie TV su Amazon Prime

## Descrizione del Progetto

Questo progetto è stato sviluppato per analizzare un dataset di film e serie TV disponibili su Amazon Prime. Il dataset comprende informazioni come cast, registi, valutazioni, anno di rilascio e durata. Gli obiettivi principali del progetto includono:

1. **Preprocessare i dati:** Pulizia e trasformazione dei dati grezzi per prepararli all'analisi.
2. **Analizzare i dati:** Esplorazione delle distribuzioni e delle correlazioni tra variabili per ottenere insight significativi.
3. **Clustering:** Raggruppare i titoli per caratteristiche simili, identificando pattern nei dati.
4. **Apprendimento Supervisionato:** Creazione di modelli predittivi per classificare i titoli più visti.
5. **Apprendimento Probabilistico:** Sviluppo di modelli probabilistici per la classificazione con focus sulla probabilità di appartenenza a determinate classi.
6. **Cross-Validation:** Valutazione approfondita delle performance dei modelli attraverso tecniche di validazione incrociata.
7. **Generazione di Fatti Prolog:** Creazione di rappresentazioni logiche dei dati utilizzando fatti Prolog per inferenze future.
8. **Creazione di una Knowledge Base:** Sviluppo di un'ontologia per organizzare e gestire la conoscenza relativa ai film e alle serie TV su Amazon Prime.

## Struttura del Progetto

Il progetto è organizzato nelle seguenti directory:

```
amazon_prime_project/
│
├── data/
│   ├── amazon_prime_titles.csv            # Dataset grezzo
│   ├── processed_data.csv                 # Dataset processato
│   ├── description_embeddings.npy         # Embedding delle descrizioni dei titoli
│   └── content_facts.pl                   # Fatti Prolog generati
│
├── scripts/
│   ├── preprocess_prime_dataset.py        # Script per il preprocessing dei dati
│   ├── analyze_data.py                    # Script per l'analisi dei dati
│   ├── clustering.py                      # Script per il clustering
│   ├── supervised.py                      # Script per l'apprendimento supervisionato
│   ├── probabilistic_learning.py          # Script per l'apprendimento probabilistico
│   ├── create_embedding.py                # Script per la generazione di embeddings
│   ├── generate_prolog_files.py           # Script per la generazione di fatti Prolog
│   ├── knowledge_base.py                  # Script per la base di conoscenza e l'ontologia
│   ├── cross_validation.py                # Script per la cross-validation
│   
│
├── results/
│   ├── visualizations/                    # Directory di output per le visualizzazioni generate
│   │   ├── analyze_data/                  # Visualizzazioni relative all'analisi dei dati
│   │   │   ├── release_year_distribution_movie.png
│   │   │   ├── release_year_distribution_tv_show.png
│   │   │   ├── type_distribution.png
│   │   ├── clustering/                    # Visualizzazioni relative al clustering
│   │   │   ├── clusters_visualization_pca_Movie.png
│   │   │   ├── clusters_visualization_pca_TV_Show.png
│   │   └── probabilistic_learning/        # Visualizzazioni relative all'apprendimento probabilistico
│   │       └── roc_curve.png
│   ├── models/                            # Directory di output per i modelli salvati e le metriche
│       ├── cross_validation/              # Metriche di cross-validation
│       │   ├── rf_cross_validation_metrics.csv
│       │   ├── nb_cross_validation_metrics.csv
│       ├── knowledge_base/                # Ontologia e Knowledge Base
│       │   └── amazon_prime_ontology.owl
│       ├── probabilistic_learning/        # Metriche di apprendimento probabilistico
│       │   └── classification_report.csv
│       ├── supervised/                    # Metriche di apprendimento supervisionato
│           └── classification_report.csv
│
└── README.md

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