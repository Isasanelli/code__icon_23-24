
# Netflix Titles Classification and Recommendation System

## Descrizione del Progetto

Questo progetto si focalizza sulla creazione di un sistema di classificazione e raccomandazione per i titoli più popolari presenti su Netflix, basato su tecniche di Machine Learning e su una Knowledge Base (KB) sviluppata in Prolog. Il sistema consente di preprocessare i dati, creare embeddings, addestrare modelli di classificazione supervisionati, generare raccomandazioni basate sulle preferenze dell'utente, e infine, generare automaticamente una KB in Prolog per supportare il ragionamento simbolico.

## Funzionalità Principali

- **Preprocessing dei Dati**: Pulizia e trasformazione dei dati per prepararli alla fase di machine learning.
- **Creazione di Embeddings**: Generazione di rappresentazioni numeriche per i titoli e le categorie.
- **Classificazione Supervisionata**: Addestramento di vari modelli di classificazione (Decision Tree, Random Forest, SVM, XGBoost, MLP) e validazione incrociata per garantire risultati robusti.
- **Raccomandazione**: Sistema di raccomandazione basato su titolo e categoria, con feedback dell'utente per migliorare la precisione delle raccomandazioni.
- **Generazione della KB in Prolog**: Creazione automatica di una KB basata sui risultati del machine learning, che include fatti e regole per il ragionamento.
- **Interfaccia Utente da Terminale**: Navigazione semplice con menu per eseguire tutte le operazioni principali.

## Struttura del Progetto

CODE_ICON_23-24/
│
├── .idea/
├── .venv/
├── data/
│   └── netflix_titles.csv
│
├── results/
│   ├── models/
│   │   └── supervised/
│   ├── visualizations/
│   │   ├── analyze_data/
│   │   ├── cross_validation/
│   │   ├── statistic_recommander/
│   │   └── supervised/
│
├── scripts/
│   ├── __pycache__/
│   ├── analyze_data.py
│   ├── create_embedding.py
│   ├── cross_validation.py
│   ├── generate_prolog_files.py
│   ├── main.py
│   ├── preprocess_data_dataset.py
│   ├── search_and_recommendation.py
│   └── supervised.py
│
├── .gitignore
├── README.md
└── requirements.txt


## Requisiti

- **Python 3.12.5**
- Librerie Python:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `imblearn`
  - `xgboost`
  - `nltk` (se necessario per l'elaborazione del testo)
- **SWI-Prolog** (per eseguire la KB generata in Prolog) versione 8.0.3-1
  
- Installare
   ```
   python -m spacy download en_core_web_sm
   ```

## Installazione

1. **Clona il repository:**
   ```bash
   git clone https://github.com/tuo-username/tuo-progetto.git
   cd tuo-progetto
   ```

2. **Installa i requisiti:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepara i dati:**
   Assicurati che i dati necessari siano presenti nella directory `data` con il nome `processed_data.csv`.

## Esecuzione del Progetto

1. **Avvia il programma principale:**
   ```bash
   python main.py
   ```

2. **Opzioni disponibili nel Menu principale:**
   - `Classificazione dei Film e Serie TV`: Esegue il preprocessing, l'addestramento dei modelli, e la validazione incrociata.
   - `Ricerca Film o Serie TV`: Permette di cercare titoli o categorie e ricevere raccomandazioni personalizzate.
   - `Generazione della KB`: Genera la Knowledge Base in Prolog basata sui risultati del machine learning.
   - `Uscita`: Termina l'esecuzione del programma.

## Come Usare il Sistema

- **Classificazione**: Avvia il processo di classificazione dei dati, che include la creazione di embeddings, l'addestramento di modelli, e la validazione dei risultati.
- **Ricerca e Raccomandazione**: Inserisci un titolo o una categoria per ricevere raccomandazioni personalizzate.
- **Knowledge Base**: La KB in Prolog verrà generata automaticamente dopo la classificazione, permettendo di effettuare interrogazioni basate sui fatti e le regole generate.

## Collaboratori

Belforte Matteo
Sasanelli Ilenia


## Licenza

Questo progetto è rilasciato sotto la licenza MIT. Vedi il file LICENSE per maggiori dettagli.
