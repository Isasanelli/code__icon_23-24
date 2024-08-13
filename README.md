
# Sistema di Raccomandazione di Film basato su Ontologie e Apprendimento Automatico

## Descrizione del Progetto

Questo progetto mira a sviluppare un sistema di raccomandazione di film utilizzando i titoli di Amazon Prime, integrando tecniche di rappresentazione della conoscenza, machine learning, ragionamento probabilistico e ontologie. L'obiettivo è creare un sistema che combini modelli supervisionati e non supervisionati con tecniche avanzate di clustering e embedding, arricchito dall'integrazione di conoscenza di fondo proveniente dal Web Semantico.

## Struttura del Progetto

1. **Analisi e Preprocessamento dei Dati**
2. **Creazione di Embedding e Knowledge Graph**
3. **Generazione di File Prolog e Creazione della Knowledge Base**
4. **Clustering Avanzato**
5. **Apprendimento Supervisionato con Valutazione Appropriata**
6. **Integrazione di Conoscenza di Fondo (BK) dal Web Semantico**
7. **Modelli di Apprendimento Probabilistico**

## Requisiti

Per eseguire questo progetto, è necessario installare le seguenti librerie Python:

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

## Installazione

Assicurati di avere Python 3.7+ installato sul tuo sistema. 

Installa l'ambiente virtuale:

```bash
python -m venv .venv
```

Attiva l'ambiente virtuale

```bash
.venv\Scripts\Activate
```

Poi, installa tutte le dipendenze usando pip:

```bash
pip install -r requirements.txt
```

## Struttura dei File

- `analyze_data.py` : Analisi e preprocessamento dei dati.
- `preprocess_prime_dataset.py` : Preprocessamento specifico e pulizia dei dati.
- `create_embedding.py` : Creazione degli embeddings.
- `generate_prolog_files.py` : Generazione dei file Prolog.
- `knowledge_Base.py` : Creazione della knowledge base.
- `clustering.py` : Clustering avanzato.
- `supervised.py` : Apprendimento supervisionato con valutazione appropriata.
.

## Come Eseguire il Progetto

- eliminare i csv e png presenti nelle cartelle source e chart prima di eseguire i comandi: 

### 1. Analisi e Preprocessamento dei Dati

Esegui `analyze_data.py` per analizzare e preprocessare i dati:

```bash
python  .\scripts\analyze_data.py
```

Esegui `preprocess_prime_dataset.py` per eseguire il preprocessamento specifico:

```bash
python  .\scripts\preprocess_prime_dataset.py
```

### 2. Creazione di Embedding e Knowledge Graph


Esegui `create_embedding.py` per creare gli embeddings:

```bash
python  .\scripts\create_embedding.py
```

### 3. Generazione di File Prolog e Creazione della Knowledge Base

Esegui `generate_prolog_files.py` per generare i file Prolog:

```bash
python  .\scripts\generate_prolog_files.py
```

Esegui `knowledge_Base.py` per creare la knowledge base.

Prima di eseguire il file, è importante aver installato SWI-Prolog
Basti visitare il sito : 

<https://www.swi-prolog.org/download/stable> 

Dopo aver installato `SWI-Prolog ( versione 8.0.3)`, è necessario configurare la variabile d'ambiente SWI_HOME_DIR per puntare alla directory di
installazione di SWI-Prolog.

Andare su `proprietà di sistema`.

In basso cliccare su `variabili d'ambiente`

considerare solo il pannello inferiore

Nella sezione `Variabili di sistema` considerare il nome `path` e fare clic due volte.

Fai clic su `Modifica`.
Fai clic su `Nuovo` e inserisci il percorso della directory : `C:\Program Files\swipl\bin`.

Fai clic su "OK" per salvare la variabile.

Apri il terminale PowerShell come amministratore.

Imposta la variabile d'ambiente per puntare alla directory di installazione di  SWI-Prolog:
  
```bash
[System.Environment]::SetEnvironmentVariable('SWI_HOME_DIR', 'C:\Program Files\swipl', 'Machine')
```

Esegui KB:

```bash
python  .\scripts\knowledge_Base.py
```

### 4. Clustering Avanzato

Esegui `clustering.py` per eseguire il clustering avanzato:

```bash
python  .\scripts\clustering.py
```

### 5. Apprendimento Supervisionato con Valutazione Appropriata

Esegui `supervised.py` per addestrare e valutare i modelli di apprendimento supervisionato:

```bash
python  .\scripts\supervised.py
```

## Conclusioni

Questo progetto integra vari aspetti avanzati del programma di insegnamento, fornendo un approccio completo e dinamico all'integrazione di ontologie, apprendimento supervisionato e non supervisionato, ragionamento probabilistico e modelli neurali. Ogni passo è documentato e i codici possono essere eseguiti seguendo le istruzioni sopra fornite.

## Autore

- Ilenia Sasanelli 
- Ruggiero Moschese

## Licenza

Questo progetto è concesso in licenza sotto i termini della licenza MIT.
