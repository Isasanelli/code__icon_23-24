
# Progetto di Analisi dei Dati Amazon Prime

## Panoramica
Questo progetto è stato creato per analizzare un dataset di film disponibili su Amazon Prime. Il dataset include informazioni come cast, registi, valutazioni, anno di rilascio e durata. Gli obiettivi principali sono classificare i titoli più visti e ricercati, classificare gli attori più cercati e sviluppare sistemi basati sulla conoscenza utilizzando i dati.

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
│   └── processed_data.csv              # Dataset processato
│
├── notebooks/
│   ├── exploratory_analysis.ipynb      # Jupyter notebook per EDA
│   └── model_development.ipynb         # Jupyter notebook per lo sviluppo del modello
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
│   └── main_application.py             # Script principale per eseguire l'intero progetto
│
└── results/
    ├── visualizations/                 # Directory di output per le visualizzazioni generate
    └── models/                         # Directory di output per i modelli salvati e le metriche
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

2. **Esegui lo Script Principale**
   - Per eseguire l'intero progetto, dal preprocessing dei dati alla creazione della base di conoscenza, esegui semplicemente:
     ```bash
     python scripts/main_application.py
     ```
   - Questo script eseguirà automaticamente tutti gli altri script nell'ordine corretto.

3. **Esegui i Singoli Script**
   - Se preferisci eseguire singole componenti del progetto separatamente, puoi eseguire ogni script uno per volta:
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
