import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(__file__)

# Carica i dati
dataset_path = os.path.join(script_dir, '../source/dataset_clustered.csv')
df = pd.read_csv(dataset_path)
X = df.drop(['title', 'Cluster'], axis=1)
y = df['Cluster']

# Divide i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scala i dati
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Addestra il modello
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

# Predice le etichette per il set di test
y_pred = model.predict(X_test_scaled)

# Genera la matrice di confusione
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
conf_matrix_path = os.path.join(script_dir, '../charts/supervised/confusion_matrix.png')
plt.savefig(conf_matrix_path)
plt.close()

# Calcola l'accuratezza e il report di classificazione
acc_score = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Salva il risultato in un file di testo
result_text_path = os.path.join(script_dir, '../charts/supervised/classification_report.txt')
with open(result_text_path, 'w') as f:
    f.write('Confusion Matrix:\n')
    f.write(f'{cm}\n\n')
    f.write('Classification Report:\n')
    f.write(f'{class_report}\n\n')
    f.write(f'Accuracy Score: {acc_score}\n')

print('Confusion Matrix:')
print(cm)
print('\nClassification Report:')
print(class_report)
print(f'\nAccuracy Score: {acc_score}\n')
