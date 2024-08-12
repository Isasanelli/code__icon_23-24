import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

KNOWLEDGE_BASE_PATH = "../source/knowledge_base.csv"

def load_data() -> pd.DataFrame:
    """Carica i dati dalla Knowledge Base."""
    return pd.read_csv(KNOWLEDGE_BASE_PATH)

def train_model(df: pd.DataFrame):
    """Addestra il modello di classificazione."""
    # Aggiunta di altre feature per migliorare la classificazione
    X = df[['same_director', 'similarity_score', 'Year', 'Country', 'Date_Added']]
    
    # Sostituisci 'same_director' con la feature target che desideri classificare
    y = df['same_director']  # Modifica questo per usare il target corretto

    # Converti le feature categoriali in dummy variables (one-hot encoding)
    X = pd.get_dummies(X, columns=['Country', 'Date_Added'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modello RandomForest con tuning degli iperparametri
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"ROC-AUC: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    df = load_data()
    train_model(df)

if __name__ == "__main__":
    main()
