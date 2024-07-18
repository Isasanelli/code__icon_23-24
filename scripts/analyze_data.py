import pandas as pd
import os
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment

# Assicurati che il percorso del file sia corretto
file_path = os.path.join(os.path.dirname(__file__), '..', 'source', 'amazon_prime_titles.csv')

# Verifica se il file esiste
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"Il file {file_path} non esiste. Assicurati che il percorso sia corretto.")

# Carica il dataset
dataset = pd.read_csv(file_path)

# Converti colonne selezionate in categorie
categorical_columns = ['type', 'rating', 'listed_in']
for col in categorical_columns:
    dataset[col] = dataset[col].astype('category')

# Genera statistiche descrittive per tutte le colonne
description = dataset.describe(include='all')

# Salva le statistiche in un file CSV
output_csv_path = os.path.join(os.path.dirname(__file__), '..', 'source', 'statistics.csv')
description.to_csv(output_csv_path)

# Salva le statistiche in un file Excel per una migliore visualizzazione
output_excel_path = os.path.join(os.path.dirname(__file__), '..', 'source', 'statistics_excel.xlsx')
description.to_excel(output_excel_path, engine='openpyxl')

# Aggiungi formattazione al file Excel
wb = load_workbook(output_excel_path)
ws = wb.active

# Applica l'allineamento del testo e auto-dimensionamento delle colonne
for row in ws.iter_rows():
    for cell in row:
        cell.alignment = Alignment(wrap_text=True)

# Auto-dimensionamento delle colonne
for col in ws.columns:
    max_length = 0
    column = col[0].column_letter
    for cell in col:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(cell.value)
        except:
            pass
    adjusted_width = (max_length + 2)
    ws.column_dimensions[column].width = adjusted_width

wb.save(output_excel_path)

# Genera visualizzazioni grafiche
output_charts_path = os.path.join(os.path.dirname(__file__), '..', 'charts', 'analyze_data')
os.makedirs(output_charts_path, exist_ok=True)

# Istogramma per release_year
plt.figure(figsize=(10, 6))
dataset['release_year'].hist(bins=30, edgecolor='black', color='skyblue')
plt.title('Distribuzione degli Anni di Rilascio')
plt.xlabel('Anno di Rilascio')
plt.ylabel('Numero di Film/Serie')
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.savefig(os.path.join(output_charts_path, 'release_year_distribution.png'))


# Grafico a barre per type
plt.figure(figsize=(10, 6))
dataset['type'].value_counts().plot(kind='bar', edgecolor='black', color='skyblue')
plt.title('Distribuzione per Tipo (Movie/TV Show)')
plt.xlabel('Tipo')
plt.ylabel('Conteggio')
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.savefig(os.path.join(output_charts_path, 'type_distribution.png'))

# Grafico a barre per rating
plt.figure(figsize=(10, 6))
dataset['rating'].value_counts().plot(kind='bar', edgecolor='black', color='skyblue')
plt.title('Distribuzione dei Rating')
plt.xlabel('Rating')
plt.ylabel('Conteggio')
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.savefig(os.path.join(output_charts_path, 'rating_distribution.png'))

print(f"Analisi completata. Le statistiche sono state salvate in '{output_csv_path}' e '{output_excel_path}'. Le visualizzazioni grafiche sono state salvate nella cartella '{output_charts_path}'.")
