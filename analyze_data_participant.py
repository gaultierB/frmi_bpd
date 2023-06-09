import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json


def get_patient(df):
    return df[df['group'] == 'Patient']


def get_control(df):
    return df[df['group'] == 'Control']


def convert_type_all_df(df):
    df['participant_id'] = df['participant_id'].astype(str)
    df['sex'] = df['sex'].astype(str)
    df['group'] = df['group'].astype(str)
    df['hand'] = df['hand'].astype(str)
    df['drug-1'] = df['drug-1'].astype(str)
    df['drug-2'] = df['drug-2'].astype(str)
    df['drug-3'] = df['drug-3'].astype(str)
    return df


def heatmap(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matrice de corrélation')
    plt.show()
    return correlation_matrix


def bar_chart(df):
    # Diagramme à barres pour chaque colonne
    for column in df.columns:
        if df[column].dtype == 'object':
            plt.figure(figsize=(8, 6))
            df[column].value_counts().plot(kind='bar', edgecolor='black')
            plt.xlabel(column)
            plt.ylabel('Fréquence')
            plt.title(f'Distribution des valeurs dans la colonne {column}')
            plt.show()


def histogram_phys_abuse(df):
    # Histogramme des scores CTQ-PhysAbuse
    plt.hist(df['CTQ-PhysAbuse'])
    plt.xlabel('Score')
    plt.ylabel('Fréquence')
    plt.title('Distribution des scores CTQ-PhysAbuse')
    plt.show()


def clear_no_drug(df):
    # DONT WORK
    a = df.replace("n/a", '', inplace=True)
    print(a)
    print(df['drug-2'])


def box_graph(df, name_graph):
    # TODO create colonne stat for usage drug

    # Créer une liste pour stocker les données de chaque colonne
    data = []

    # Itérer sur chaque colonne du DataFrame et ajouter les données à la liste
    numeric_columns = []
    for column in df.columns:
        if df[column].dtype == 'float64' or df[column].dtype == 'int64' and column != "age":
            data.append(df[column].dropna().values)
            numeric_columns.append(column)

    # Créer une figure et un axe pour le plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer les diagrammes en boîte
    ax.boxplot(data)

    # Ajouter des étiquettes d'axe x pour chaque colonne
    ax.set_xticklabels(numeric_columns, rotation=45, ha='right', fontsize=6)

    # Personnaliser le plot
    ax.set_xlabel('Colonnes')
    ax.set_ylabel('Valeurs')
    ax.set_title(name_graph)
    fig.savefig(name_graph + '.png')
    plt.close(fig)


def graph_cloud_point(df, numeric_columns):
    # Graphique en nuage de points pour chaque paire de colonnes numériques
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            plt.figure(figsize=(8, 6))
            plt.scatter(df[numeric_columns[i]], df[numeric_columns[j]])
            plt.xlabel(numeric_columns[i])
            plt.ylabel(numeric_columns[j])
            plt.title(f'Relation entre {numeric_columns[i]} et {numeric_columns[j]}')
            plt.show()


def a(df):
    # Graphique en secteurs pour une colonne catégorielle
    categorical_column = 'votre_colonne_catégorielle'
    plt.figure(figsize=(8, 6))
    df[categorical_column].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.xlabel(categorical_column)
    plt.title(f"Répartition des valeurs dans la colonne {categorical_column}")
    plt.show()


def graph_line(df, numeric_columns):
    # Graphique en ligne pour une colonne temporelle
    temporal_column = 'votre_colonne_temporelle'
    plt.figure(figsize=(12, 6))
    plt.plot(df[temporal_column], df[numeric_columns[0]])
    plt.xlabel(temporal_column)
    plt.ylabel(numeric_columns[0])
    plt.title(f"Variation de {numeric_columns[0]} au fil du temps")
    plt.show()


def graph_numeric_value(df, numeric_columns):
    # Diagramme de dispersion pour chaque paire de colonnes numériques
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            plt.figure(figsize=(8, 6))
            plt.scatter(df[numeric_columns[i]], df[numeric_columns[j]])
            plt.xlabel(numeric_columns[i])
            plt.ylabel(numeric_columns[j])
            plt.title(f'Relation entre {numeric_columns[i]} et {numeric_columns[j]}')
            plt.show()


# TODO fix label
def analyse_score_participants_zanarini(df, n_zanarini):
    fichier_json = "ds000214-download/participants.json"

    # Lecture du fichier JSON
    with open(fichier_json, "r") as f:
        donnees = json.load(f)
    json_zanarini = donnees["Zanarini-" + n_zanarini]
    # Utilisation des données
    zanarini_label = list(json_zanarini["Levels"].values())
    zanarini_data = df.get("Zanarini-" + n_zanarini)

    # Comptage des occurrences de chaque niveau de symptôme
    zanarini_data_count = pd.Series(0, index=[0, 1, 2, 3, 4])
    value = zanarini_data.value_counts()
    zanarini_data_count.update(value)

    # Création du graphique à barres
    plt.bar(zanarini_label, zanarini_data_count.values)

    # Configuration des étiquettes sur l'axe des x
    plt.xticks(zanarini_label, zanarini_label)

    # Ajout de labels et de titre
    plt.xlabel("NANI Zanarini-" + n_zanarini + " Score")
    plt.ylabel("Count")
    plt.title(json_zanarini["LongName"])
    plt.text(0.5, -0.2, json_zanarini["Description"], transform=plt.gca().transAxes, ha='center')

    # Affichage du graphique
    plt.show()


def get_json_participant():
    fichier_json = "ds000214-download/participants.json"

    # Lecture du fichier JSON
    with open(fichier_json, "r") as f:
        return json.load(f)


def pretty_show_column(df):
    columns = df.columns
    data = get_json_participant()
    print(data)
    for column in columns:
        name = column
        description = ""
        if column in data:
            if "LongName" in data[column].keys():
                name = data[column]["LongName"]
            if "Description" in data[column].keys():
                description = data[column]["Description"]
        print("Nom colonne : " + name + " Type : " + str(df[column].dtype) + " Description : " + description)


if __name__ == '__main__':
    df_all = pd.read_table('ds000214-download/participants.tsv')
    df_all = convert_type_all_df(df_all)
    pretty_show_column(df_all)
    # clear_no_drug(df_all)
    df_patient = get_patient(df_all)
    df_control = get_control(df_all)
    # numeric_columns = df_all.select_dtypes(include=['int', 'float']).columns
    # pas lisible graph_cloud_point(df_all, numeric_columns)
    box_graph(df_patient, 'Diagramme en boite des personnes patientes')
    box_graph(df_control, 'Diagramme en boite des personnes controlés')
    # heatmap(df_patient)

    # histogram_phys_abuse(df_all)
    # for i in range(1, 10):
    # analyse_score_participants_zanarini(df_all, str(i))
