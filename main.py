import pandas as pd
import os
import learning  # Importa il tuo modulo di learning
import reasoning  # Importa il tuo modulo di reasoning
# --- IMPORTAZIONI AGGIUNTE (SOLO PER IL PLOTTING) ---
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ===================================================================
#   FUNZIONE DI UTILITY PER I GRAFICI (aggiunta qui per non toccare altri file)
# ===================================================================
def mostra_feature_importance(model, feature_names, title):
    """
    Genera e mostra un grafico a barre orizzontali per l'importanza delle feature.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10, 7))
        plt.title(title, fontsize=16)
        plt.barh(range(len(indices)), importances[indices], color='darkorange', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importanza Relativa (Gini Importance)", fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print(f"[ATTENZIONE] Il modello {type(model).__name__} non ha l'attributo 'feature_importances_'.")


def main():
    """
    Orchestratore dell'esperimento comparativo ML vs ML+OntoBK.
    """
    print("===================================================================")
    print(" AVVIO PROGETTO: INTEGRAZIONE DI MACHINE LEARNING E CONOSCENZA SEMANTICA")
    print("===================================================================")

    # --- Configurazione dei percorsi ---
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'himym_episodewise.csv')

    # ===================================================================
    #   SCENARIO 1: ESECUZIONE BASELINE (SOLO MACHINE LEARNING)
    # ===================================================================
    print("\n--- [SCENARIO 1] ESECUZIONE MODELLI SU DATASET ORIGINALE ---")

    dataset_originale = learning.load_dataset(dataset_path)
    X_orig, y_orig = learning.preprocessing_dataset(dataset_originale.copy())
    feature_names_orig = X_orig.columns  # Prendiamo i nomi delle colonne

    print("\n[INFO] Addestramento e valutazione del Decision Tree (Baseline)...")
    dt_model_orig = learning.decisiontree_classifier(X_orig, y_orig)
    # --- AGGIUNTA #1 ---
    mostra_feature_importance(dt_model_orig, feature_names_orig, "Feature Importance - DT (Baseline)")

    print("\n[INFO] Addestramento e valutazione del Random Forest (Baseline)...")
    rf_model_orig = learning.randomforest_classifier(X_orig, y_orig)
    # --- AGGIUNTA #2 ---
    mostra_feature_importance(rf_model_orig, feature_names_orig, "Feature Importance - RF (Baseline)")

    print("\n--- [SCENARIO 1] COMPLETATO ---")

    # ===================================================================
    #   FASE DI ARRICCHIMENTO CON CONOSCENZA DI FONDO (BK)
    # ===================================================================
    print("\n--- [FASE DI ARRICCHIMENTO] INTEGRAZIONE CONOSCENZA SEMANTICA ---")

    # (Questa sezione rimane identica, non la tocco)
    populated_onto_path = os.path.join(os.path.dirname(__file__), 'ontology', 'himym_populated.rdf')
    if not os.path.exists(populated_onto_path):
        print("[INFO] Ontologia popolata non trovata. Avvio il processo di popolamento...")
        reasoning.populate_ontology()
        print("[INFO] Popolamento completato.")
    else:
        print("[INFO] Utilizzo dell'ontologia popolata esistente.")
    print("[INFO] Avvio del ragionatore semantico per classificare gli episodi...")
    semantic_results = reasoning.run_reasoning()
    semantic_df = pd.DataFrame(list(semantic_results.items()), columns=['Episode_ID', 'Semantic_Class'])
    dataset_arricchito = dataset_originale.copy()
    dataset_arricchito['Episode_ID'] = dataset_arricchito.apply(
        lambda row: f"Episode_{int(row['Season'])}_{int(row['Episode']):02d}", axis=1
    )
    dataset_arricchito = pd.merge(dataset_arricchito, semantic_df, on='Episode_ID', how='left')
    print("\n[SUCCESS] Dataset arricchito con la nuova feature 'Semantic_Class'.")

    # ===================================================================
    #   SCENARIO 2: ESECUZIONE CON DATASET ARRICCHITO (ML + BK)
    # ===================================================================
    print("\n--- [SCENARIO 2] ESECUZIONE MODELLI SU DATASET ARRICCHITO CON BK ---")

    # (Questa sezione rimane identica, non la tocco)
    dataset_arricchito['Semantic_Class'] = dataset_arricchito['Semantic_Class'].fillna(
        'Non Anotato')  # Aggiungo un riempimento per sicurezza
    text_columns = ['Director', 'Writers', 'Verdict', 'Semantic_Class']
    for column in text_columns:
        if column in dataset_arricchito.columns:
            le = LabelEncoder()
            dataset_arricchito[column] = le.fit_transform(dataset_arricchito[column])
    columns_features_arricchite = ['Season', 'Episode', 'Director', 'Writers', 'Viewers', 'Votes',
                                   'Semantic_Class']
    X_arr = dataset_arricchito[columns_features_arricchite]
    y_arr = dataset_arricchito['Verdict']
    feature_names_arr = X_arr.columns  # Prendiamo i nomi delle colonne

    print("\n[INFO] Addestramento e valutazione del Decision Tree (con BK)...")
    dt_model_arr = learning.decisiontree_classifier(X_arr, y_arr)
    # --- AGGIUNTA #3 ---
    mostra_feature_importance(dt_model_arr, feature_names_arr, "Feature Importance - DT (con BK)")

    print("\n[INFO] Addestramento e valutazione del Random Forest (con BK)...")
    rf_model_arr = learning.randomforest_classifier(X_arr, y_arr)
    # --- AGGIUNTA #4 ---
    mostra_feature_importance(rf_model_arr, feature_names_arr, "Feature Importance - RF (con BK)")

    print("\n--- [SCENARIO 2] COMPLETATO ---")
    print("\n===================================================================")
    print(" ESPERIMENTO COMPARATIVO CONCLUSO")
    print("===================================================================")


if __name__ == "__main__":
    main()