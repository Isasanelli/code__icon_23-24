import os
from pyswip import Prolog

def create_kb() -> Prolog:
    prolog = Prolog()
    base_path = os.path.dirname(__file__)
    
    # Ensure the paths are absolute and adjust for Prolog syntax
    facts_path = os.path.abspath(os.path.join(base_path, "..", "file_prolog", "prime_facts.pl")).replace("\\", "/")
    locations_path = os.path.abspath(os.path.join(base_path, "..", "file_prolog", "prime_locations.pl")).replace("\\", "/")

    if not os.path.exists(facts_path):
        raise FileNotFoundError(f"Il file {facts_path} non esiste.")
    if not os.path.exists(locations_path):
        raise FileNotFoundError(f"Il file {locations_path} non esiste.")

    try:
        prolog.consult(facts_path)
        prolog.consult(locations_path)
        print("Files consulted successfully.")
    except Exception as e:
        print(f"Error consulting files: {e}")

    return prolog

def execute_queries(prolog):
    # Esempio di query: Trova tutti i film del 2019
    query1 = list(prolog.query("fact(Title, 2019, Genre)"))
    print(f"Film del 2019: {len(query1)} trovati")
    
    # Esempio di query: Trova tutti i film del genere 'Comedy'
    query2 = list(prolog.query("fact(Title, Year, 'Comedy')"))
    print(f"Film di genere 'Comedy': {len(query2)} trovati")

    # Esempio di query: Trova tutte le location in 'United States'
    query3 = list(prolog.query("location(Title, 'United States', Date)"))
    print(f"Film localizzati in 'United States': {len(query3)} trovati")

    # Esempio di query: Trova tutti i film del '2021'
    query4 = list(prolog.query("fact(Title, Year, Genre), sub_atom(Year, 0, 4, _, '2021')"))
    print(f"Film del  2021: {len(query4)} trovati")

    # Esempio di query: Trova tutti i film del '2021' in 'United States'
    query5 = list(prolog.query("fact(Title, Year, Genre), sub_atom(Year, 0, 4, _, '2021'), location(Title, 'United States', Date)"))
    print(f"Film del 2021 in 'United States': {len(query5)} trovati")

    

def main():
    try:
        kb = create_kb()
        print("Knowledge Base creata con successo.")
        
        execute_queries(kb)
        
    except Exception as e:
        print(f"Errore durante la creazione della Knowledge Base: {e}")

if __name__ == "__main__":
    main()
