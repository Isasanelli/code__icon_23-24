import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from collections import Counter

def load_processed_data(filepath):
    """Carica i dati preprocessati dal file CSV."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocessa il DataFrame per l'apprendimento supervisionato."""
    # Mapping dei rating in valori numerici
    rating_mapping = {
        'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 'NC-17': 5,
        'Unrated': np.nan, '13+': 3, '16+': 4, '18+': 5, '7+': 2, 'ALL': 1
    }
    
    df['numeric_rating'] = df['rating'].map(rating_mapping)
    df = df.dropna(subset=['numeric_rating'])
    
    df = df.copy()  
    df['title_length'] = df['title'].apply(len)
    df['release_month'] = pd.to_datetime(df['date_added']).dt.month
    df['release_season'] = pd.to_datetime(df['date_added']).dt.month % 12 // 3 + 1
    
    # Creazione del target 'most_watched' in base alla mediana del rating
    df['most_watched'] = df.groupby('content_category')['numeric_rating'].transform(lambda x: x > x.median())
    
    return df

def filter_embeddings(df, embeddings):
    """Filtra gli embedding in base all'indice del DataFrame."""
    if len(embeddings) > len(df):
        embeddings = embeddings[df.index]
    return embeddings

def plot_learning_curves_with_table(model, X, y, model_name, plot_output_dir, report_df):
    """Genera e salva le curve di apprendimento per un modello, includendo una tabella con i risultati."""
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy')
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    # Rimuovi la colonna 'support' dal report_df
    report_df = report_df.drop(columns=['support'], errors='ignore')

    # Creare la figura e gli assi per il grafico e la tabella
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    # Curva di apprendimento
    ax[0].plot(train_sizes, mean_train_errors, label='Training Error', color='green')
    ax[0].plot(train_sizes, mean_test_errors, label='Testing Error', color='red')
    ax[0].set_title(f'Learning Curve for {model_name}')
    ax[0].set_xlabel('Training Set Size')
    ax[0].set_ylabel('Error')
    ax[0].legend()

    # Tabella con le metriche
    ax[1].axis('off')
    table = ax[1].table(cellText=report_df.values,
                        colLabels=report_df.columns,
                        rowLabels=report_df.index,
                        cellLoc='center',
                        loc='center')
    table.scale(1, 1.5)

    # Modifica i margini per creare spazio tra il grafico e la tabella
    plt.subplots_adjust(wspace=0.5)  

    output_path = os.path.join(plot_output_dir, f'{model_name}_learning_curve_with_table.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()


def train_model(model, X, y, model_name, model_output_base_dir, plot_output_base_dir):
    """Addestra un modello e salva i risultati."""
    # Creazione delle directory specifiche per ciascun modello
    model_output_dir = os.path.join(model_output_base_dir, model_name)
    plot_output_dir = os.path.join(plot_output_base_dir, model_name)

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(plot_output_dir, exist_ok=True)

    # Utilizza KFold per la validazione incrociata
    cv = KFold(n_splits=5)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
    print(f'{model_name} Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})')

    # Predizione dei valori e calcolo delle metriche
    y_pred = cross_val_predict(model, X, y, cv=cv)
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(columns=['support'], errors='ignore')  # Rimuovi 'support'
    report_path = os.path.join(model_output_dir, f'{model_name}_classification_report.csv')
    report_df.to_csv(report_path, index=True)

    # Genera e salva la curva di apprendimento con la tabella
    plot_learning_curves_with_table(model, X, y, model_name, plot_output_dir, report_df)

    # Genera e salva la matrice di confusione
    cm = confusion_matrix(y, y_pred)
    cm_path = os.path.join(plot_output_dir, f'{model_name}_confusion_matrix.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(cm_path)
    plt.show()
    plt.close()

    # Aggiunge altre metriche
    roc_auc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    print(f'{model_name} ROC-AUC: {roc_auc:.4f}')
    print(f'{model_name} F1-Score: {f1:.4f}')
    print(f'{model_name} Precision: {precision:.4f}')
    print(f'{model_name} Recall: {recall:.4f}')

def supervised_learning(baseDir):
    """Gestisce l'intero processo di apprendimento supervisionato."""
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    embeddings_path = os.path.join(baseDir, '..', 'data', 'content_category_embeddings.npy')
    title_embeddings_path = os.path.join(baseDir, '..', 'data', 'title_embeddings.npy')

    # Controlla se i file degli embeddings esistono nel percorso corretto
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Il file {embeddings_path} non esiste. Verifica il percorso.")
    if not os.path.exists(title_embeddings_path):
        raise FileNotFoundError(f"Il file {title_embeddings_path} non esiste. Verifica il percorso.")
    
    
    # Definisce le directory di output
    model_output_base_dir = os.path.join(baseDir, '..', 'results', 'models', 'supervised')
    plot_output_base_dir = os.path.join(baseDir, '..', 'results', 'visualizations', 'supervised')

    os.makedirs(model_output_base_dir, exist_ok=True)
    os.makedirs(plot_output_base_dir, exist_ok=True)
    
    # Caricamento e preprocessing dei dati
    df = load_processed_data(filepath)
    df = preprocess_data(df)
    
    # Caricamento e filtraggio degli embedding
    embeddings = np.load(embeddings_path)
    embeddings = filter_embeddings(df, embeddings)
    
    # Creazione delle feature e del target
    features = np.hstack([embeddings, df[['numeric_rating', 'release_year', 'title_length', 'release_month', 'release_season']].values])
    y = df['most_watched']
    
    # Distribuzione delle classi prima di SMOTE
    print("Distribuzione delle classi prima di SMOTE:", Counter(y))
    
    smote = SMOTE(random_state=42)
    features_resampled, y_resampled = smote.fit_resample(features, y)
    
    # Distribuzione delle classi dopo SMOTE
    print("Distribuzione delle classi dopo SMOTE:", Counter(y_resampled))

    # Divide i dati per il test
    X_train, X_test, y_train, y_test = train_test_split(features_resampled, y_resampled, test_size=0.3, random_state=42)
    
    # Definisce i modelli
    models = {
        'DecisionTree': DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5),
        'RandomForest': RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_split=10, min_samples_leaf=5),
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME', n_estimators=50, learning_rate=1),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance')
    }

    # Addestra e valuta ciascun modello
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        train_model(model, X_train, y_train, model_name, model_output_base_dir, plot_output_base_dir)

    print("Apprendimento supervisionato completato e risultati salvati.")
