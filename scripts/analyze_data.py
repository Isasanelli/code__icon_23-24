import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def plot_distribution(df, column, title, output_dir, top_n=None, interval=None, filter_type=None):
    """Genera e salva un grafico di distribuzione per una colonna specificata del DataFrame."""
    plt.figure(figsize=(12, 8))
    
    if filter_type:
        df = df[df['content_category'].str.contains(filter_type, na=False)]
    
    if interval:
        df = df.copy()
        df['grouped_year'] = (df[column] // interval) * interval
        data = df['grouped_year'].value_counts().sort_index()
        sns.barplot(x=data.index, y=data.values, hue=data.index, palette='viridis', dodge=False)
        plt.xlabel(f'{column.capitalize()} (Grouped by {interval} years)', fontsize=14)
    else:
        data = df[column].value_counts(dropna=False)
        if top_n:
            data = data.head(top_n)
        sns.barplot(x=data.index, y=data.values, hue=data.index, palette='viridis', dodge=False)
        plt.xlabel(column.capitalize(), fontsize=14)
    
    plt.title(title, fontsize=16)
    plt.ylabel('Count', fontsize=14)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=12)
    
    if interval is None and len(data) > 20:
        plt.xticks(ticks=plt.xticks()[0][::max(1, len(data)//20)])
    
    output_filename = f"{column}_distribution"
    if filter_type:
        output_filename += f"_{filter_type.lower()}"
    output_path = os.path.join(output_dir, f'{output_filename}.png')
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_pie_chart(df, column, title, output_dir, top_n=None):
    """Genera e salva un grafico a torta per una colonna specificata del DataFrame."""
    plt.figure(figsize=(10, 7))
    
    data = df[column].value_counts(dropna=False)
    if top_n:
        data = data.head(top_n)
    
    plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(data)))
    plt.title(title, fontsize=16)
    
    output_filename = f"{column}_pie_chart.png"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()

def apply_standard_scaler(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def save_statistics(df, output_dir):
    """Salva le statistiche più popolari in un CSV."""
    popular_stats = df.groupby('content_category').agg({
        'show_id': 'count',
        'preferences': 'mean',
        'release_year': 'mean',
    }).reset_index().rename(columns={
        'show_id': 'total_count',
        'preferences': 'avg_preference',
        'release_year': 'avg_release_year'
    })
    
    output_path = os.path.join(output_dir, 'popular_statistics.csv')
    popular_stats.to_csv(output_path, index=False)
    print(f"Statistiche salvate in {output_path}")

def analyze_data(baseDir):
    """Funzione principale per l'analisi dei dati e la generazione dei grafici."""
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    output_dir = os.path.join(baseDir, '..', 'results', 'visualizations', 'analyze_data')

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(filepath)
    
    # Distribuzione Film vs Serie TV basata su 'content_category'
    plt.figure(figsize=(8, 6))
    df['type'] = df['content_category'].apply(lambda x: 'Movie' if 'Movie' in x else 'TV Show')
    sns.countplot(x='type', data=df, hue='type', palette="Set2", dodge=False)
    plt.title('Number of Movies vs TV Shows on Netflix')
    plt.savefig(os.path.join(output_dir, 'movies_vs_tvshows.png'))
    plt.show()

    # Visualizza la distribuzione per anno di rilascio (film e serie TV separate)
    plot_distribution(df, 'release_year', 'Distribution of Movies by Decade', output_dir, interval=10, filter_type='Movie')
    plot_distribution(df, 'release_year', 'Distribution of TV Shows by Decade', output_dir, interval=10, filter_type='TV Show')
    
    # Visualizza la distribuzione per categoria di contenuto
    plot_distribution(df, 'content_category', 'Distribution of Content Categories', output_dir, top_n=20)

    # Salva le statistiche più popolari
    save_statistics(df, output_dir)

    print(f"Grafici salvati nella directory {output_dir}")
