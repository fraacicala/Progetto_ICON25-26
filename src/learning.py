import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, learning_curve



# Funzione che si occupa di caricare il dataset da un file CSV. Utilizzata apposita funzione della libreria pandas.
# La variabile path rappresenta il percorso del dataset.
# Restituisce df, il dataset caricato sotto forma di DataFrame di pandas.
def load_dataset(path):
    df = pd.read_csv(path)
    return df


# Funzione che si occupa del preprocessing del dataset.
# Utilizza LabelEncoder per convertire i valori testuali delle colonne in valori numerici mediante la funzione fit_trasform().
# Seleziona l'insieme X di feature e la variabile target y dal DataFrame con le funzioni di pandas.
def preprocessing_dataset(df):
    text_columns = ['Director', 'Writers', 'Verdict']
    for column in text_columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna('missing')
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    columns_features = ['Season', 'Episode', 'Director', 'Writers', 'Viewers', 'Votes']
    X = df[columns_features]
    y = df['Verdict']
    return X, y


# Funzione che genera e stampa il grafico dell'importanza che viene assegnata a ciascuna feature dai modelli.
# Estrae l'importanza dalle feature del modello e li ordina.
# Utilizzando la libreria matlotlib, genera il grafico impostando diverse caratteristiche estetiche.
def show_feature_importance(model, feature_names, title):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10, 7))
        plt.title(title, fontsize=16)
        plt.barh(range(len(indices)), importances[indices], color='blue', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importanza Relativa", fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Il modello {type(model).__name__} non ha l'attributo 'feature_importances_'.")

# Funzione che genera e stampa la curva di apprendimento dei modelli.
# Essa mostra l'andamento del punteggio di validazione e training.
# Lo fa addestrando il modello su sottoinsiemi di dati e facendi i calcoli sia sul set di training che su un set di validazione incrociata.
# I punteggi sono matrici dove ogni riga è una dimensione del training set e ogni colonna è un fold.
# Crea il grafico tramite la libreria matplotlib e imposta diverse caratteristiche visive.
def show_learning_curve(estimator, X, y, title="Curva di Apprendimento"):
    print(f"Generazione grafico per: {title}...")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Numero di esempi di training")
    plt.ylabel("Accuratezza (Score)")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score (Test)")
    plt.legend(loc="best")
    plt.show()


# Funzione per allenare un classificatore Decision Tree utilizzando una validazione incrociata annidata.
# Il ciclo interno viene utilizzato da GridSearchCV per trovare gli iperparametri migliori.
# Il ciclo esterno valuta le performance del modello, simulando la presenza di dati nuovi.
# Viene definita una griglia di parametri che poi GridSearchCV andrà a testare combinandoli nei diversi modi possibili.
# Vengono definite le metriche da calcolare (accuratezza, precisione, richiamo, f1).
# I risultati stampati sono la media delle performance ottenute sui set di test di fold del ciclo esterno. Restituisce il modello migliore.

def decisiontree_classifier(X, y):
    inner_loop = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    outer_loop = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    model = DecisionTreeClassifier(random_state=0)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21, 2),
        'min_samples_leaf': range(1, 21, 2)
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_loop)
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    nested_results = cross_validate(grid, X, y, cv=outer_loop, scoring=metrics)
    print(
        f"Accuracy Media:  {nested_results['test_accuracy'].mean():.2%} (+/- {nested_results['test_accuracy'].std():.2%})")
    print(f"Precision Media: {nested_results['test_precision_weighted'].mean():.2%}")
    print(f"Recall Media:    {nested_results['test_recall_weighted'].mean():.2%}")
    print(f"F1-Score Media:  {nested_results['test_f1_weighted'].mean():.2%}")
    grid.fit(X, y)
    best_model = grid.best_estimator_
    print(f"Parametri migliori:  {grid.best_params_}")

    show_learning_curve(best_model, X, y, title="Learning Curve - Decision Tree")
    feature_names = X.columns
    show_feature_importance(best_model, feature_names, "Feature Importance - DT")

    return best_model

# Funzione per allenare un classificatore Random Forest utilizzando una validazione incrociata annidata.
# Il ciclo interno viene utilizzato da GridSearchCV per trovare gli iperparametri migliori.
# Il ciclo esterno valuta le performance del modello, simulando la presenza di dati nuovi.
# Viene definita una griglia di parametri che poi GridSearchCV andrà a testare combinandoli nei diversi modi possibili.
# Vengono definite le metriche da calcolare (accuratezza, precisione, richiamo, f1).
# I risultati stampati sono la media delle performance ottenute sui set di test di fold del ciclo esterno. Restituisce il modello migliore.
def randomforest_classifier(X, y):
    inner_loop = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    outer_loop = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    model = RandomForestClassifier(random_state=0, class_weight='balanced')
    param_grid = {
        'n_estimators': [50, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_loop)
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    nested_results = cross_validate(grid, X, y, cv=outer_loop, scoring=metrics)
    print(
        f"Accuracy Media (Nested):  {nested_results['test_accuracy'].mean():.2%} (+/- {nested_results['test_accuracy'].std():.2%})")
    print(f"Precision Media (Nested): {nested_results['test_precision_weighted'].mean():.2%}")
    print(f"Recall Media (Nested):    {nested_results['test_recall_weighted'].mean():.2%}")
    print(f"F1-Score Media (Nested):  {nested_results['test_f1_weighted'].mean():.2%}")
    grid.fit(X, y)
    best_model = grid.best_estimator_
    print(f"Parametri migliori:  {grid.best_params_}")

    show_learning_curve(best_model, X, y, title="Learning Curve - Random Forest")
    feature_names = X.columns
    show_feature_importance(best_model, feature_names, "Feature Importance - RF")

    return best_model