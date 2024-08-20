import pandas as pd
import os

def raccomandazione(baseDir):
    try:
        print("Inizio del sistema di raccomandazione...")
        
        # Load the processed data
        filepath = os.path.join(baseDir, 'data', 'processed_data.csv')
        df = pd.read_csv(filepath)

        # Simple content-based filtering example:
        user_liked_category = 'Action - International Movies'  # This could be dynamic

        recommendations = df[df['content_category'].str.contains(user_liked_category, na=False)]
        
        if recommendations.empty:
            print("Nessuna raccomandazione disponibile per la categoria selezionata.")
        else:
            print(f"Raccomandazioni per la categoria '{user_liked_category}':")
            for title in recommendations['title'].head(10):  # Limiting to top 10 recommendations
                print(f"- {title}")

        print("Sistema di raccomandazione completato.")
    except Exception as e:
        print(f"Errore durante il processo di raccomandazione: {e}")
