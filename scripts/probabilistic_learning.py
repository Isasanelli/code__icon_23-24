import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # Importa SMOTE per il bilanciamento delle classi

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df['most_watched'] = df['rating'] > df['rating'].median()
    return df

def balance_classes(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def train_probabilistic_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Bilancia le classi nel set di allenamento
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)
    
    model = GaussianNB()
    
    # Esegue una GridSearch per trovare il miglior valore di var_smoothing
    params = {'var_smoothing': np.logspace(0,-9, num=100)}
    grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='f1')
    grid_search.fit(X_train_bal, y_train_bal)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Best var_smoothing:", grid_search.best_params_)
    print(report)
    
    return best_model, report

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))

    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    df = load_processed_data(filepath)
    
    df = preprocess_data(df)
    
    features = ['rating', 'release_year', 'duration']
    X = df[features]
    y = df['most_watched']
    
    model, report = train_probabilistic_model(X, y)
    
    # Salva i risultati
    output_dir = os.path.join(baseDir, '..', 'results', 'models', 'probabilistic_learning')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    report_path = os.path.join(output_dir, 'classification_report.csv')
    pd.DataFrame(report).transpose().to_csv(report_path)
    
    print(f"Report salvato in {report_path}")
    print("Modello probabilistico migliorato addestrato e valutato")
