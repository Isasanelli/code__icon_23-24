import os
import time
import pandas as pd

# Configura manualmente il percorso di SWI-Prolog
os.environ['SWI_HOME_DIR'] = r'C:\Program Files\swipl'
os.environ['PATH'] += os.pathsep + r'C:\Program Files\swipl\bin'

from pyswip import Prolog

def assert_location_facts(kb):
    with open("file_prolog/prime_locations.pl", "r") as loc_file:
        lines = loc_file.readlines()
        for line in lines:
            kb.assertz(line.strip())

def create_kb() -> Prolog:
    prolog = Prolog()
    prolog.consult("file_prolog/prime_facts.pl")
    assert_location_facts(prolog)
    return prolog

def calculate_features(kb, title_id) -> dict:
    features_dict = {}
    features_dict["title"] = title_id
    title_id = f"'{title_id}'"
    features_dict["same_director"] = len(list(kb.query(f'same_director({title_id}, T2)')))
    features_dict["same_title"] = len(list(kb.query(f'same_title({title_id}, T2)')))
    return features_dict

def produce_working_dataset(kb: Prolog, path: str):
    print(f"Producing dataset at {path}")
    start = time.time()
    titles_complete = pd.read_csv("source/titles_selected.csv")
    extracted_values_df = None
    first = True
    for title_id in titles_complete["Title"]:
        features_dict = calculate_features(kb, title_id)
        if first:
            extracted_values_df = pd.DataFrame([features_dict])
            first = False
        else:
            extracted_values_df = pd.concat([extracted_values_df, pd.DataFrame([features_dict])], ignore_index=True)
    extracted_values_df.to_csv(path, index=False)
    end = time.time()
    print("Total time: ", end-start)

knowledge_base = create_kb()
produce_working_dataset(knowledge_base, "source/working_dataset.csv")
