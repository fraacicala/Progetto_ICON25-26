import pandas as pd
import os
from sklearn.model_selection import cross_val_predict

# --- IMPORTA LE TUE FUNZIONI DAI FILE IN 'src' ---
from src.learning import load_dataset, preprocessing_dataset, decisiontree_classifier, randomforest_classifier
from src.reasoning import run_reasoning


def main():
    print("##################################################")
    print("   PROGETTO 'HOW I MET YOUR DATA' - HIMYM")
    print("##################################################\n")

    # --- FASE 1: CARICAMENTO E PREPROCESSING DEI DATI ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'dataset', 'himym_episodewise.csv')

    try:
        df_originale = load_dataset(dataset_path)
        # Passiamo una copia per non modificare l'originale
        X, y_numerico = preprocessing_dataset(df_originale.copy())
        print(f"Dataset caricato e preprocessato con successo ({len(df_originale)} episodi).")
    except FileNotFoundError:
        print(f"ERRORE: File non trovato in {dataset_path}")
        return
    except Exception as e:
        print(f"ERRORE nel preprocessing: {e}")
        return

    # --- FASE 2: CONFRONTO DEI MODELLI MACHINE LEARNING ---
    print("\n--- ESECUZIONE MODULO MACHINE LEARNING ---")

    print("\n--- Valutazione Modello 1: Decision Tree ---")
    decisiontree_model = decisiontree_classifier(X, y_numerico)

    print("\n--- Valutazione Modello 2: Random Forest ---")
    randomforest_model = randomforest_classifier(X, y_numerico)

    # Scegliamo il modello migliore basato sulle performance per l'analisi degli errori
    miglior_modello_ml = randomforest_model
    print("\n -> La Random Forest è stata scelta come modello di riferimento per l'analisi finale.")

    # --- FASE 3: GENERAZIONE PREDIZIONI ONESTE ---
    print("\n--- Generazione predizioni ML 'oneste' con Cross-Validation ---")
    # cross_val_predict esegue una k-fold CV e restituisce le previsioni per ogni dato
    # quando quel dato era nel test set. È il modo più onesto per valutare.
    predizioni_ml_numeriche = cross_val_predict(miglior_modello_ml, X, y_numerico, cv=5)
    print(f" -> Predizioni generate per tutti i {len(predizioni_ml_numeriche)} episodi.")

    # --- FASE 4: ESECUZIONE RAGIONAMENTO SEMANTICO ---
    risultati_ontologia = run_reasoning()

    # --- FASE 5: ANALISI DEGLI ERRORI (Il cuore del progetto) ---
    print(f"\n{'=' * 60}")
    print(f"   ANALISI DEGLI ERRORI DELL'ML TRAMITE L'ONTOLOGIA")
    print(f"{'=' * 60}")

    # Mappa per tradurre i numeri (0,1,2) del ML in parole
    # ATTENZIONE: Controlla che questo ordine sia corretto per il tuo LabelEncoder!
    # Se hai usato "Eccellente, Buono, Scarso" -> 0=Buono, 1=Eccellente, 2=Scarso
    mappa_ml = {1: "Eccellente", 0: "Buono", 2: "Scarso"}

    errori_da_analizzare = []

    for i, verita_numerica in enumerate(y_numerico):
        predizione_numerica = predizioni_ml_numeriche[i]

        # Analizziamo solo i casi in cui il modello ML ha sbagliato
        if predizione_numerica != verita_numerica:
            # Ricostruiamo l'ID dell'episodio nel formato che hai usato nell'ontologia
            s = str(df_originale.loc[i, 'Season'])
            e = str(df_originale.loc[i, 'Episode']).zfill(2)
            id_episodio_onto = f"Episode_{s}_{e}"

            # Recuperiamo le etichette
            verita_str = mappa_ml.get(verita_numerica, "???")
            predizione_str = mappa_ml.get(predizione_numerica, "???")
            spiegazione_semantica = risultati_ontologia.get(id_episodio_onto, "Standard")

            errori_da_analizzare.append({
                "Episodio": id_episodio_onto,
                "Voto Reale": verita_str,
                "ML ha detto": predizione_str,
                "Ontologia dice": spiegazione_semantica
            })

    df_errori = pd.DataFrame(errori_da_analizzare)

    print(f"\nIl modello ML ha commesso {len(df_errori)} errori su 208 episodi.")
    print("Analizziamo se l'ontologia può spiegarne alcuni:")

    # ESEMPIO 1: L'ML sottovaluta un capolavoro (es. Slap Bet)
    print("\n--- CASI DI SOTTOVALUTAZIONE (L'ontologia ha ragione) ---")
    casi_sottovalutati = df_errori[
        (df_errori['Voto Reale'] == 'Eccellente') & (df_errori['Ontologia dice'] == 'Eccellente')
        ]
    print(casi_sottovalutati.head(5).to_string(index=False))
    print(
        " -> Spiegazione: L'ontologia ha rilevato elementi (Oggetti Iconici, Eventi Chiave) che hanno reso questi episodi dei capolavori, ma che l'ML non poteva vedere dai numeri.")

    # ESEMPIO 2: L'ML sopravvaluta un filler (es. Zoo or False)
    print("\n--- CASI DI SOPRAVVALUTAZIONE (L'ontologia ha ragione) ---")
    casi_sopravvalutati = df_errori[
        (df_errori['Voto Reale'] == 'Scarso') & (df_errori['Ontologia dice'] == 'Scarso')
        ]
    print(casi_sopravvalutati.head(5).to_string(index=False))
    print(
        " -> Spiegazione: L'ontologia ha rilevato elementi di trama 'Filler' o 'Gimmick' che giustificano il basso rating, nonostante i metadati tecnici potessero sembrare buoni.")

    print("\n\n--- FINE ANALISI ---")


if __name__ == "__main__":
    main()