import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from collections import Counter

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    rating_mapping = {
        'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 'NC-17': 5,
        'Unrated': np.nan, '13+': 3, '16+': 4, '18+': 5, '7+': 2, 'ALL': 1
    }
    
    df['numeric_rating'] = df['rating'].map(rating_mapping)
    df = df.dropna(subset=['numeric_rating'])
    df['most_watched'] = df.groupby('content_category')['numeric_rating'].transform(lambda x: x > x.median())
    
    return df

def filter_embeddings(df, embeddings):
    if len(embeddings) > len(df):
        embeddings = embeddings[df.index]
    return embeddings

def plot_learning_curves(model, X, y, model_name, plot_output_dir):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy')
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, mean_train_errors, label='Training Error', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Testing Error', color='red')
    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Error')
    plt.legend()

    output_path = os.path.join(plot_output_dir, f'{model_name}_learning_curve.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def train_model(model, X, y, model_name, model_output_base_dir, plot_output_base_dir):
    # Creazione delle directory specifiche per ciascun modello
    model_output_dir = os.path.join(model_output_base_dir, model_name)
    plot_output_dir = os.path.join(plot_output_base_dir, model_name)

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(plot_output_dir, exist_ok=True)

    cv = RepeatedKFold(n_splits=5, n_repeats=2)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
    print(f'{model_name} Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})')

    # Salva le curve di apprendimento
    plot_learning_curves(model, X, y, model_name, plot_output_dir)
    
    # Salva il report di classificazione
    report_df = pd.DataFrame({'Model': [model_name], 'Accuracy': [scores.mean()], 'Std Dev': [scores.std()]})
    report_path = os.path.join(model_output_dir, f'{model_name}_classification_report.csv')
    report_df.to_csv(report_path, index=False)

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    embeddings_path = os.path.join(baseDir, '..', 'data', 'content_category_embeddings.npy')

    # Definisce le directory di output
    model_output_base_dir = os.path.join(baseDir, '..', 'results', 'models', 'supervised')
    plot_output_base_dir = os.path.join(baseDir, '..', 'results', 'visualizations', 'supervised')

    os.makedirs(model_output_base_dir, exist_ok=True)
    os.makedirs(plot_output_base_dir, exist_ok=True)
    
    df = load_processed_data(filepath)
    df = preprocess_data(df)
    
    embeddings = np.load(embeddings_path)
    embeddings = filter_embeddings(df, embeddings)
    
    features = np.hstack([embeddings, df[['numeric_rating', 'release_year']].values])
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
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'NaiveBayes': GaussianNB()
    }

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        train_model(model, X_train, y_train, model_name, model_output_base_dir, plot_output_base_dir)
