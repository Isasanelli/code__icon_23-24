import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from imblearn.over_sampling import SMOTE
from collections import Counter
from generate_prolog_files import generate_prolog_files



def load_processed_data(filepath):
    """Carica i dati preprocessati dal file CSV e restituisce un DataFrame."""
    return pd.read_csv(filepath, encoding='utf-8')

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
    
    # Creazione delle colonne 'title_length' e 'release_month'
    df.loc[:, 'title_length'] = df['title'].apply(len)
    df.loc[:, 'release_month'] = pd.to_datetime(df['date_added']).dt.month
    df.loc[:, 'release_season'] = pd.to_datetime(df['date_added']).dt.month % 12 // 3 + 1
    
    # Creazione del target 'most_watched' in base alla mediana del rating
    df.loc[:, 'most_watched'] = df.groupby('content_category')['numeric_rating'].transform(lambda x: x > x.median())
    
    return df


def filter_embeddings(df, embeddings):
    """Filtra gli embedding in base all'indice del DataFrame."""
    if len(embeddings) > len(df):
        embeddings = embeddings[df.index]
    return embeddings

def print_section_header(title):
    """Stampa un'intestazione di sezione ben formattata."""
    print("\n" + "="*50)
    print(f"{title.center(50)}")
    print("="*50 + "\n")

def print_model_performance(model_name, scores, roc_auc, f1, precision, recall):
    """Stampa le metriche di performance in un formato leggibile."""
    print(f"{model_name} Performance Summary:")
    print(f" - Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f" - ROC-AUC: {roc_auc:.4f}")
    print(f" - F1-Score: {f1:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall: {recall:.4f}")
    print("-" * 50 + "\n")

def train_and_evaluate_model(model, X, y, model_name, model_output_base_dir, plot_output_base_dir):
    """Addestra un modello, ottimizza gli iperparametri e salva i risultati."""
    print_section_header(f"Training {model_name}")
    
    model_output_dir = os.path.join(model_output_base_dir, model_name)
    plot_output_dir = os.path.join(plot_output_base_dir, model_name)

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(plot_output_dir, exist_ok=True)

    param_grid = {
        'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    }

    search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='accuracy') if model_name in param_grid else None
    if search:
        print("Performing hyperparameter search...")
        search.fit(X, y)
        model = search.best_estimator_
        print(f"Best parameters for {model_name}: {search.best_params_}\n")

    cv = KFold(n_splits=5)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
    print(f'{model_name} Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})\n')

    y_pred = cross_val_predict(model, X, y, cv=cv)
    
    # Genera e salva i risultati
    roc_auc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    print_model_performance(model_name, scores, roc_auc, f1, precision, recall)


    if len(y_pred) > 0:
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(model_output_dir, f'{model_name}_classification_report.csv')
        report_df.to_csv(report_path, index=True, encoding='utf-8')  # Ensure UTF-8 encoding

    # Genera e salva le curve di apprendimento se report_df is not None
    if report_df is not None:
        plot_learning_curves_with_table(model, X, y, model_name, plot_output_dir, report_df)
    
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

def apply_standard_scaler(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)



def plot_learning_curves_with_table(model, X, y, model_name, plot_output_dir, report_df):
    """Genera e salva le curve di apprendimento per un modello, includendo una tabella con i risultati."""
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy')
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    report_df = report_df.drop(columns=['support'], errors='ignore')

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    ax[0].plot(train_sizes, mean_train_errors, label='Training Error', color='green')
    ax[0].plot(train_sizes, mean_test_errors, label='Testing Error', color='red')
    ax[0].set_title(f'Learning Curve for {model_name}')
    ax[0].set_xlabel('Training Set Size')
    ax[0].set_ylabel('Error')
    ax[0].legend()

    ax[1].axis('off')
    table = ax[1].table(cellText=report_df.values,
                        colLabels=report_df.columns,
                        rowLabels=report_df.index,
                        cellLoc='center',
                        loc='center')
    table.scale(1, 1.5)

    plt.subplots_adjust(wspace=0.5)  

    output_path = os.path.join(plot_output_dir, f'{model_name}_learning_curve_with_table.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()

def supervised_learning(baseDir):
    """Gestisce l'intero processo di apprendimento supervisionato."""
    print_section_header("Apprendimento supervisionato in corso...")

    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    embeddings_path = os.path.join(baseDir, '..', 'data', 'content_category_embeddings.npy')
    title_embeddings_path = os.path.join(baseDir, '..', 'data', 'title_embeddings.npy')

    if not os.path.exists(embeddings_path) or not os.path.exists(title_embeddings_path):
        raise FileNotFoundError("File degli embeddings mancanti. Verifica il percorso.")

    model_output_base_dir = os.path.join(baseDir, '..', 'results', 'models', 'supervised')
    plot_output_base_dir = os.path.join(baseDir, '..', 'results', 'visualizations', 'supervised')

    os.makedirs(model_output_base_dir, exist_ok=True)
    os.makedirs(plot_output_base_dir, exist_ok=True)

    # Caricamento dei dati preprocessati con encoding UTF-8
    df = pd.read_csv(filepath, encoding='utf-8')
    df = preprocess_data(df)

    # Caricamento degli embeddings
    embeddings = np.load(embeddings_path)
    embeddings = filter_embeddings(df, embeddings)

    # Combinazione delle caratteristiche
    features = np.hstack([embeddings, df[['numeric_rating', 'release_year', 'title_length', 'release_month', 'release_season']].values])
    y = df['most_watched']

    smote = SMOTE(random_state=42)
    features_resampled, y_resampled = smote.fit_resample(features, y)

    X_train, X_test, y_train, y_test = train_test_split(features_resampled, y_resampled, test_size=0.3, random_state=42)

    # Modelli di classificazione
    models = {
        'DecisionTree': DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5),
        'RandomForest': RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_split=10, min_samples_leaf=5),
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME', n_estimators=50, learning_rate=1),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'XGBoost': XGBClassifier(eval_metric='logloss', verbosity=0)
    }

    for model_name, model in models.items():
        train_and_evaluate_model(model, X_train, y_train, model_name, model_output_base_dir, plot_output_base_dir)
    
    # Generazione del file Prolog KB
    generate_prolog_files(baseDir)

    print_section_header("Apprendimento supervisionato completato e Prolog KB generato.")

