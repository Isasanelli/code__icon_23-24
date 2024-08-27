
# Netflix Titles Classification and Recommendation System

## Descrizione del Progetto

Questo progetto si focalizza sulla creazione di un sistema di classificazione e raccomandazione per i titoli più popolari presenti su Netflix, basato su tecniche di Machine Learning e su una Knowledge Base (KB) sviluppata in Prolog. Il sistema consente di preprocessare i dati, creare embeddings, addestrare modelli di classificazione supervisionati, generare raccomandazioni basate sulle preferenze dell'utente, e infine, generare automaticamente una KB in Prolog per supportare il ragionamento simbolico.


## Struttura del Progetto
```
CODE_ICON_23-24/
│
├── data/
│   └── netflix_titles.csv
│
|── documenti
|     ├── documentazione_progetto
|     └── comandi_console_prolog
|     
|
├── results/
│   ├── models/
│   │   └── supervised/
|   |           ├── AdaBoost/
|   |           ├── DecisionTree/
|   |           ├── K-NN/
|   |           ├── RandomForest/
|   |           ├── XGBoost/
|   |
│   ├── visualizations/
│   │   ├── analyze_data/
│   │   ├── cross_validation/
│   │   ├── statistic_recommander/
│   │   └── supervised/
|              ├── AdaBoost/
|              ├── DecisionTree/
|              ├── K-NN/
|              ├── RandomForest/
|              ├── XGBoost/
|
├── prolog/
│   ├── knowledge_base_fact.pl
│   ├── knowledge_base_rules.pl
|
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
```
## Descrizione struttura del progetto

- **data/**: Contiene i file di dati necessari per il progetto.
  - `netflix_titles.csv`: Dataset originale dei titoli.
  - `processed_data.csv`: Dataset preprocessato.
  - `content_category_embeddings.npy`, `title_embeddings.npy`: Embeddings generati dalle pipeline di creazione degli embeddings.

- **documenti/** : contiene la documentazione progetto e un file che descrive i comandi da console per swi prolog
  
- **results/**: Contiene i risultati del progetto.
  - **models/supervised/**: Modelli di apprendimento supervisionato.
    - `AdaBoost/`
    - `DecisionTree/`
    - `KNN/`
    - `RandomForest/`
    - `XGBoost/`
  - **prolog/**: File della knowledge base Prolog.
    - `knowledge_base_fact.pl`: Fatti generati dai dati preprocessati.
    - `knowledge_base_rules.pl`: Regole generate per la raccomandazione dei contenuti.
  - **visualizations/**: Grafici e visualizzazioni generate durante l'analisi e la validazione incrociata.
    - `analyze_data/`: Grafici generati durante l'analisi dei dati.
    - `cross_validation/`: Visualizzazioni generate durante la validazione incrociata.
    - `statistic_recommander/`: Grafici relativi alle raccomandazioni statistiche.

- **scripts/**: Contiene gli script Python principali per l'elaborazione e l'analisi dei dati.
  - `analyze_data.py`: Script per l'analisi esplorativa dei dati.
  - `create_embedding.py`: Script per la generazione degli embeddings utilizzando SpaCy.
  - `cross_validation.py`: Script per eseguire la validazione incrociata dei modelli di apprendimento supervisionato.
  - `generate_prolog_files.py`: Script per la generazione di fatti e regole in Prolog.
  - `main.py`: Script principale che gestisce il flusso dell'intero progetto.
  - `preprocess_data.py`: Script per il preprocessing dei dati.
  - `search_and_recommendation.py`: Script per la ricerca e raccomandazione di contenuti.
  - `supervised.py`: Script per l'apprendimento supervisionato e la valutazione dei modelli.



## Requisiti

- **Python 3.12.5**
Per eseguire il progetto, assicurati di avere installato i seguenti pacchetti Python, elencati in `requirements.txt`:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `spacy`
- `xgboost`
- `imblearn`
  
- **SWI-Prolog** (per eseguire la KB generata in Prolog) versione 8.0.3-1
  ```
   pip install spacy
    ```
  
- Dopo aver installato spacy, installare
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

## Esecuzione del Progetto

1. **Avvia il programma principale:**
   ```bash
   python main.py
   ```
## Menu Principale

Il file `main.py` fornisce un'interfaccia interattiva per eseguire le diverse fasi del progetto. Le opzioni del menu includono:

- Classificazione dei Film e Serie TV
- Ricerca e Raccomandazioni
- Visualizza Titoli Più Popolari
- Generazione della KB
- Uscita

## Come Usare il Sistema

- **Classificazione**: Avvia il processo di classificazione dei dati, che include la creazione di embeddings, l'addestramento di modelli, e la validazione dei risultati.
- **Ricerca e Raccomandazione**: Inserisci un titolo o una categoria per ricevere raccomandazioni personalizzate.
- **Visualizza Titoli più popolari**: Permette di visualizzare a schermo quali sono i titoli più popolari su netflix, sia per i film che per le serie tv. Per tanto mostra anche i generi più popolari 
- **Knowledge Base**: La KB in Prolog verrà generata automaticamente dopo la classificazione, permettendo di effettuare interrogazioni basate sui fatti e le regole generate.

## Utilizzo della Knowledge Base in Prolog

Dopo aver generato la KB in Prolog, puoi utilizzare SWI-Prolog ( versione 8.0.3) per caricare i file e eseguire query sulla base di conoscenza generata. Segui questi passi:

1. **Avvia SWI-Prolog** e naviga nella directory contenente i file della KB:
   ```prolog
   ?- cd('posizione della tua direcotory di progetto /results/prolog').
   ```

2. **Carica i file Prolog**:
   ```prolog
   ?- [facts].  % Carica il file dei fatti
   ?- [rules].  % Carica il file delle regole
   ```

3. **Esegui query per ottenere raccomandazioni**:
   ```prolog
   ?- recommend(Content).
   ```
Questo comando restituirà  più titoli. 
Inserendo il comando  `( ; )`  verranno mostrati più titoli. \
Premere successivamente il `( . )` per terminare \

**Per altri comandi, consulatare il file.txt chiamato `comandi_console_prolog.txt` presente nella cartella documenti**

## Collaboratori

Belforte Matteo \
Sasanelli Ilenia


## Licenza

Questo progetto è rilasciato sotto la licenza MIT. Vedi il file LICENSE per maggiori dettagli.
