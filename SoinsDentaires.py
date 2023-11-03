# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:58:04 2023

@author: nickz
"""

import pandas as pd 
import numpy as np 
import streamlit as st  
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import r2_score
import geopandas as gpd
import folium
from folium.plugins import TimeSliderChoropleth

df = pd.read_csv("Dentaire_France.csv", sep = ',')

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    
    st.write("### Contexte du projet")
    
    st.write("Ce projet est une brève étude d'un fichier en open data sur les soins dentaires en France.")
    
    st.write("Les données étudiées sont à l'emplacement suivant : https://data.opendatasoft.com/explore/dataset/medecins%40public/table/?flg=fr-fr&refine.libelle_profession=Chirurgien-dentiste")
    
    st.write("Nous explorerons ce dataset et nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude.")
    
    st.image("dentaire_1.jpg")
    
elif page == pages[1]:
    st.write("### Exploration des données")
    
    st.write("Les 5 premières lignes du fichiers :")
    
    st.dataframe(df.head())

    st.write("Dimensions du dataframe :")

    st.write(df.shape)

    
elif page == pages[2]:
    st.write("### Analyse de données")
    
    comptes_familles_actes = df["Famille de l'acte technique réalisé"].value_counts()

    etiquettes = comptes_familles_actes.index

    valeurs = comptes_familles_actes.values

    plt.figure(figsize=(10, 10))
    wedges, _ = plt.pie(valeurs, labels=None, startangle=140)

    plt.title("Répartition des familles d'actes techniques réalisés")

    plt.legend(wedges, etiquettes, title="Familles d'actes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    total = sum(valeurs)
    pourcentages = [(v / total) * 100 for v in valeurs]
    for i, pct in enumerate(pourcentages):
        angle = wedges[i].theta2 + (wedges[i].theta1 - wedges[i].theta2) / 2
        radius = 1.2  
        x = radius * np.cos(np.deg2rad(angle))  
        y = radius * np.sin(np.deg2rad(angle))  
        if pct >= 1:  
            plt.text(x, y, f"{pct:.1f}%", ha='center', va='center', rotation=0)

    plt.show()
    st.pyplot(plt)
    
    grouped = df.groupby(['Département', "Famille de l'acte technique réalisé"])

    famille_acte_counts = grouped.size().reset_index(name='Nombre')

    idx = famille_acte_counts.groupby('Département')['Nombre'].idxmax()

    familles_max_representation = famille_acte_counts.loc[idx]

    familles_max_representation.head(20)
    
    familles_max_representation["Famille de l'acte technique réalisé"].value_counts()
    
    dpt = gpd.read_file('departements-version-simplifiee.geojson')

    dpt.rename(columns={'nom': 'Département'}, inplace=True)

    dpt.head()
    
    dpt['Département'] = dpt['Département'].str.upper()

    from unidecode import unidecode
    dpt['Département'] = dpt['Département'].apply(unidecode)
    
    dpt['Département'].unique()
    familles_max_representation['Département'].unique()
    ensemble_geojson = set(dpt['Département'])
    ensemble_actes = set(familles_max_representation['Département'])


    departements_manquants_actes = ensemble_geojson - ensemble_actes


    departements_manquants_geojson = ensemble_actes - ensemble_geojson


    print("Départements manquants dans liste_actes:", departements_manquants_actes)
    print("Départements manquants dans liste_geojson:", departements_manquants_geojson)
    # Liste des départements à exclure
    departements_a_exclure = ['MAYOTTE', 'MARTINIQUE', 'LA REUNION', 'GUYANE', 'GUADELOUPE']

    # Filtrer le DataFrame pour exclure les départements spécifiques
    familles_max_representation = familles_max_representation[~familles_max_representation['Département'].isin(departements_a_exclure)]
    ensemble_geojson = set(dpt['Département'])
    ensemble_actes = set(familles_max_representation['Département'])


    departements_manquants_actes = ensemble_geojson - ensemble_actes


    departements_manquants_geojson = ensemble_actes - ensemble_geojson


    print("Départements manquants dans liste_actes:", departements_manquants_actes)
    print("Départements manquants dans liste_geojson:", departements_manquants_geojson)
    # Fusionner les DataFrames en utilisant la colonne commune "Département"
    dpt_2 = dpt.merge(familles_max_representation, on='Département', how='inner')
    dpt_2.head()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Choisissez une colormap adaptée à vos données, par exemple 'viridis'
    dpt_2.plot(column="Famille de l'acte technique réalisé", cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    ax.set_title("Actes techniques les plus réalisés par département")
    ax.get_legend().set_bbox_to_anchor((1.3, 1))
    plt.show()
    st.pyplot(plt)

elif page == pages[3]:
    
    st.write("### Modélisation")
    st.write("A venir")
