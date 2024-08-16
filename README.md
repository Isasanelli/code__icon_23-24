
# Progetto di Analisi dei Dati Amazon Prime

## Panoramica
Questo progetto è stato creato per analizzare un dataset di film e tv show disponibili su Amazon Prime. Il dataset include informazioni come cast, registi, valutazioni, anno di rilascio e durata. Gli obiettivi principali sono classificare i titoli più visti e ricercati, classificare gli attori più cercati e sviluppare sistemi basati sulla conoscenza utilizzando i dati.

## Struttura del Progetto
Il progetto è organizzato in diverse componenti chiave:
- **Preprocessing dei Dati**: Pulizia e preparazione del dataset per l'analisi.
- **Analisi dei Dati**: Analisi esplorativa dei dati (EDA) per comprendere la distribuzione dei dati.
- **Clustering**: Raggruppamento di titoli simili utilizzando algoritmi di clustering.
- **Apprendimento Supervisionato**: Costruzione di modelli di classificazione per prevedere i titoli più visti.
- **Apprendimento Probabilistico**: Implementazione di modelli probabilistici per gestire l'incertezza nelle previsioni.
- **Prolog e Base di Conoscenza**: Generazione di fatti in formato Prolog e costruzione di un'ontologia per la rappresentazione della conoscenza.
- **Cross-Validation**: Valutazione delle prestazioni dei modelli utilizzando la cross-validation.
- **Applicazione Principale**: Uno script che integra tutte le componenti sopra elencate ed esegue l'intero flusso di lavoro.

## Struttura della Directory
La struttura delle directory del progetto è la seguente:

```
amazon_prime_project/
│
├── data/
│   ├── amazon_prime_titles.csv         # Dataset grezzo
│   ├── processed_data.csv              # Dataset processato
│   ├── description_embeddings.npy      # File contenente gli embeddings generati
│   └── content_facts.pl                # Fatti Prolog generati
│
├── scripts/
│   ├── preprocess_prime_dataset.py     # Script per il preprocessing dei dati
│   ├── analyze_data.py                 # Script per l'analisi dei dati
│   ├── clustering.py                   # Script per il clustering
│   ├── supervised.py                   # Script per l'apprendimento supervisionato
│   ├── probabilistic_learning.py       # Script per l'apprendimento probabilistico
│   ├── create_embedding.py             # Script per la generazione di embeddings
│   ├── generate_prolog_files.py        # Script per la generazione di fatti Prolog
│   ├── knowledge_base.py               # Script per la base di conoscenza e l'ontologia
│   ├── cross_validation.py             # Script per la cross-validation
│   └── utils.py                        # Script con funzioni di utilità generali
│
├── results/
│   ├── visualizations/                 # Directory di output per le visualizzazioni generate
│   │   ├── analyze_data/               # Visualizzazioni generate durante l'analisi dei dati
│   │   │   ├── release_year_distribution_movie.png    # Distribuzione per anno dei film
│   │   │   ├── release_year_distribution_tv_show.png  # Distribuzione per anno delle serie TV
│   │   │   ├── type_distribution.png                  # Distribuzione per tipo di contenuto
│   │   ├── clustering/                 # Visualizzazioni generate durante il clustering
│   │   │   ├── clusters_visualization_pca_Movie.png   # Visualizzazione PCA dei cluster per i film
│   │   │   ├── clusters_visualization_pca_TV Show.png # Visualizzazione PCA dei cluster per le serie TV
│   │   ├── probabilistic_learning/     # Visualizzazioni generate durante l'apprendimento probabilistico
│   │   │   └── roc_curve.png           # Curva ROC per il modello probabilistico
│
│   ├── models/                         # Directory di output per i modelli salvati e le metriche
│   │   ├── cross_validation/           # Risultati della cross-validation
│   │   │   ├── nb_cross_validation_metrics.csv        # Metriche del modello Naive Bayes
│   │   │   ├── rf_cross_validation_metrics.csv        # Metriche del modello Random Forest
│   │   ├── knowledge_base/             # Base di conoscenza e ontologia
│   │   │   └── amazon_prime_ontology.owl              # Ontologia generata per Amazon Prime
│   │   ├── probabilistic_learning/     # Risultati dell'apprendimento probabilistico
│   │   │   └── classification_report.csv              # Report di classificazione del modello probabilistico
│   │   ├── supervised/                 # Risultati dell'apprendimento supervisionato
│   │   │   └── classification_report.csv              # Report di classificazione del modello supervisionato
│
├── .venv/                              # Ambiente virtuale Python (non sempre incluso)
├── .gitignore                          # File per escludere file/directory dal version control
├── README.md                           # Documentazione del progetto
└── requirements.txt                    # File delle dipendenze Python del progetto

```

## Passi per Eseguire il Codice
0. **Imposta ambiente**
    - ```bash
      python -m venv .venv  
      ```
    - ```bash
      .venv\Scripts\activate
      ```
1. **Installare le Librerie Necessarie**
   - Usa il file `requirements.txt` fornito per installare tutte le librerie necessarie:
     ```bash
     pip install -r requirements.txt
     ```
   - Dopo aver installato `spacy`, scarica il modello di lingua inglese:
     ```bash
     python -m spacy download en_core_web_sm
     ```
   - Facoltativamente, per `nltk`, puoi scaricare tutti i dati necessari eseguendo:
     ```python
     import nltk
     nltk.download('all')
     ```


2. **Esegui i Singoli Script**
   - E' preferibile eseguire singole componenti del progetto separatamente, per avere più controllo sui vari dati creati. E' importate seguire il seguente ordine:
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
Le seguenti librerie sono utilizzate in questo progetto:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- rdflib
- pyswip
- pgmpy
- SPARQLWrapper
- openpyxl
- wordcloud
- spacy
- tensorflow
- nltk

Tutte queste librerie possono essere installate utilizzando il file `requirements.txt`.

## Informazioni Aggiuntive
Questo progetto rappresenta un approccio completo all'analisi e alla modellazione dei dati di Amazon Prime. Combina tecniche di machine learning tradizionali con sistemi basati sulla conoscenza, inclusa la creazione di ontologie e inferenze basate su Prolog. L'obiettivo finale è costruire un sistema robusto che possa classificare e prevedere vari attributi relativi ai titoli disponibili su Amazon Prime.
