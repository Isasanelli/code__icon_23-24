import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Soppressione dei warning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

def plot_distribution(df, column, title, output_dir, top_n=None, interval=None, filter_type=None):
    plt.figure(figsize=(12, 8))
    
    if filter_type:
        df = df[df['content_category'].str.contains(filter_type)]
    
    if interval:
        df = df.copy()  # Evita SettingWithCopyWarning creando una copia
        df.loc[:, 'grouped_year'] = (df[column] // interval) * interval
        data = df['grouped_year'].value_counts().sort_index()
        sns.barplot(x=data.index, y=data.values, palette='viridis', hue=data.index, legend=False)
        plt.xlabel(f'{column.capitalize()} (Grouped by {interval} years)', fontsize=14)
    else:
        data = df[column].value_counts()
        if top_n:
            data = data.head(top_n)
        sns.barplot(x=data.index, y=data.values, palette='viridis', hue=data.index, legend=False)
        plt.xlabel(column.capitalize(), fontsize=14)
    
    plt.title(title, fontsize=16)
    plt.ylabel('Count', fontsize=14)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=12)
    
    if interval is None and len(data) > 20:
        plt.xticks(ticks=plt.xticks()[0][::int(len(data)/20)])
    
    output_filename = f"{column}_distribution"
    if filter_type:
        output_filename += f"_{filter_type.lower()}"
    output_path = os.path.join(output_dir, f'{output_filename}.png')
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_region_distribution(df, output_dir):
    if 'availability_region' not in df.columns:
        print("Colonna 'availability_region' non trovata nel DataFrame.")
        return  # Interrompi l'analisi o gestisci l'assenza della colonna
    
    plot_distribution(df, 'availability_region', 'Distribution of Content by Region', output_dir)

def plot_preference_distribution(df, output_dir):
    if 'user_preference' not in df.columns:
        print("Colonna 'user_preference' non trovata nel DataFrame.")
        return  # Interrompi l'analisi o gestisci l'assenza della colonna
    
    plot_distribution(df, 'user_preference', 'Distribution of User Preferences', output_dir)

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(baseDir, '..', 'data', 'processed_data.csv')
    output_dir = os.path.join(baseDir, '..', 'results', 'visualizations', 'analyze_data')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(filepath)
    
    # Visualizza la distribuzione per anno di rilascio (film e serie TV separate)
    plot_distribution(df, 'release_year', 'Distribution of Movies by Decade', output_dir, interval=10, filter_type='Movie')
    plot_distribution(df, 'release_year', 'Distribution of TV Shows by Decade', output_dir, interval=10, filter_type='TV Show')
    
    # Visualizza la distribuzione per categoria di contenuto
    plot_distribution(df, 'content_category', 'Distribution of Content Categories', output_dir, top_n=20)
    
    print(f"Grafici salvati nella directory {output_dir}")
