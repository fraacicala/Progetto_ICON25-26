from owlready2 import *
import pandas as pd

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

if __name__ == "__main__":
    populate_ontology()

