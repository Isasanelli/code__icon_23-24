import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df, column, title, output_dir, top_n=None, interval=None, filter_type=None):
    plt.figure(figsize=(12, 8))
    
    if filter_type:
        df = df[df['type'] == filter_type]
    
    if interval:
        # Aggrega per intervallo (ad esempio, decennio)
        df['grouped_year'] = (df[column] // interval) * interval
        data = df['grouped_year'].value_counts().sort_index()
        sns.barplot(x=data.index, y=data.values, palette='viridis')
        plt.xlabel(f'{column.capitalize()} (Grouped by {interval} years)', fontsize=14)
    else:
        data = df[column].value_counts()
        if top_n:
            data = data.head(top_n)
        sns.barplot(x=data.index, y=data.values, palette='viridis')
        plt.xlabel(column.capitalize(), fontsize=14)
    
    plt.title(title, fontsize=16)
    plt.ylabel('Count', fontsize=14)
    
    # Configura la rotazione delle etichette
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=12)
    
    # Limita il numero di etichette visualizzate
    if interval is None and len(data) > 20:  # Limita a 20 etichette se non è impostato un intervallo
        plt.xticks(ticks=plt.xticks()[0][::int(len(data)/20)])
    
    # Salva il grafico nella directory specificata
    output_filename = f"{column}_distribution"
    if filter_type:
        output_filename += f"_{filter_type.lower()}"
    output_path = os.path.join(output_dir, f'{output_filename}.png')
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Determina il percorso della directory corrente
    baseDir = os.path.dirname(os.path.abspath(__file__))

    # Definisce il percorso assoluto del file CSV di input
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')

    # Definisce la directory per salvare i grafici
    output_dir = os.path.join(baseDir, '..', 'results', 'visualizations', 'analyze_data')

    # Crea la directory se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Carica i dati
    df = pd.read_csv(filepath)
    
    # Distribuzione dei titoli per decennio per film
    plot_distribution(df, 'release_year', 'Distribution of Movies by Decade', output_dir, interval=10, filter_type='Movie')
    
    # Distribuzione dei titoli per decennio per serie TV
    plot_distribution(df, 'release_year', 'Distribution of TV Shows by Decade', output_dir, interval=10, filter_type='TV Show')
    
    # Confronto della distribuzione dei film e delle serie TV
    plot_distribution(df, 'type', 'Distribution of Movies vs TV Shows', output_dir)
    
    print(f"Grafici salvati nella directory {output_dir}")
