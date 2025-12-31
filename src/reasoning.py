from owlready2 import *
import pandas as pd
import os

#Funzione che carica l'ontologia specificata
def load_ontology(path):
    onto = get_ontology(path)
    onto.load()
    return onto

def populate_ontology():
    empty_onto = os.path.join(os.path.dirname(__file__), '..', 'ontology', 'himym_empty.rdf')
    csv_file = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'himym_episodewise.csv')
    df = pd.read_csv(csv_file)
    onto = load_ontology(empty_onto)
    with onto:
        for i, row in df.iterrows():
            season = str(row["Season"])
            episode = str(row["Episode"]).zfill(2)
            title = str(row["Title"])
            ind = "Episode_" + season + "_" + episode
            new_ep = onto.Episodio(ind)
            new_ep.haTitolo.append(title)
        output =  os.path.join(os.path.dirname(__file__), '..', 'ontology', 'himym_populated.rdf')
        onto.save(output)

def run_reasoning():
    path_ontology = os.path.join(os.path.dirname(__file__), '..', 'ontology', 'himym_populated.rdf')
    onto = load_ontology(path_ontology)
    with onto:
        sync_reasoner(infer_property_values=True)
    semantic_results = {}
    if hasattr(onto, "Episodio"):
        episode_list = onto.Episodio.instances()
    for ep in episode_list:
        name = ep.name  # Es. Episode_S01_E01
        if hasattr(onto, "Episodio_chiave") and ep in onto.Episodio_chiave.instances():
            target= "Eccellente"

        elif hasattr(onto, "Episodio_filler") and ep in onto.Episodio_filler.instances():
            target = "Scarso"

        else:
            target = "Buono"

        semantic_results[name] = target

    print(f" -> Classificazione semantica completata per {len(episode_list)} episodi.")
    return semantic_results

# Test per vedere se funziona da solo
if __name__ == "__main__":
    risultati = run_reasoning()



