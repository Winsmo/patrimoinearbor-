import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import plotly.express as px


def courbe_inertie(K, inertias):
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertias, 'bo-')
    plt.xlabel('Nombre de clusters K')
    plt.ylabel('Inertie')
    plt.title('Methode du coude pour déterminer le nombre optimal de clusters')
    plt.show()

def indice_davies_bouldin(K, db_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(K, db_scores, 'bo-')
    plt.xlabel('Nombre de clusters K')
    plt.ylabel('Indice de Davies-Bouldin')
    plt.title('Indice de Davies-Bouldin pour déterminer le nombre optimal de clusters')
    plt.show()

def coef_silhouette(K, silhouette_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Nombre de clusters K')
    plt.ylabel('Coefficient de silhouette')
    plt.title('Coefficient de silhouette pour déterminer le nombre optimal de clusters')
    plt.show()

def main():
    # Chargement des données
    data = pd.read_csv("./Data_Arbre.csv")

    # Sélection des colonnes pertinentes et création d'une copie explicite
    data1 = data[['haut_tot', 'longitude', 'latitude']].copy()

    # Vérification des types de données
    print(data1.dtypes)


    # Méthode du coude et Davies-Bouldin Index
    inertie = []
    db_scores = []
    silhouette_scores = []
    K = range(2, 6)  # Davies-Bouldin et silhouette ne sont pas définis pour k=1
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data1)
        inertie.append(kmeans.inertia_)
        labels = kmeans.labels_
        db_score = davies_bouldin_score(data1, labels)
        db_scores.append(db_score)
        silhouette_avg = silhouette_score(data1, labels)
        silhouette_scores.append(silhouette_avg)

    # Pour la méthode du coude
    optimal_k_elbow = K[inertie.index(max(inertie))]
    print(f"Le nombre optimal de clusters pour la methode du coude est : {optimal_k_elbow}")

    # Pour l'indice de Davies-Bouldin
    optimal_k_db = K[db_scores.index(min(db_scores))]
    print(f"Le nombre optimal de clusters pour l'indice de Davies-Bouldin est : {optimal_k_db}")

    # Pour le coefficient de silhouette
    optimal_k_silhouette = K[silhouette_scores.index(max(silhouette_scores))]
    print(f"Le nombre optimal de clusters pour le coefficient de silhouette est : {optimal_k_silhouette}")

    # Appel des fonctions pour tracer les courbes
    courbe_inertie(K, inertie)
    indice_davies_bouldin(K, db_scores)
    coef_silhouette(K, silhouette_scores)


    # Demander à l'utilisateur de rentrer le nombre de clusters
    nb_clusters = int(input("Nombre de clusters voulu :"))


    # Appliquer le KMeans avec le nombre de clusters choisi par l'utilisateur
    kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
    clusters = kmeans.fit_predict(data1)

    # Ajouter les clusters aux données
    data1['cluster'] = pd.Series(clusters, name='cluster')

    # Visualisation des clusters sur une carte avec Plotly
    fig = px.scatter_mapbox(data1, 
        lat="latitude", 
        lon="longitude", 
        color="cluster", 
        size="haut_tot",
        color_continuous_scale=px.colors.qualitative.Set1,
        size_max=15, 
        zoom=10,
        mapbox_style="carto-positron",
        title="Visualisation des arbres par clusters")
    
    fig.show()
    
main()



############################################################################
#
## Score de silhouette (Silhouette Score)
#silhouette_scores = []
#K = range(2, 11)
#for k in K:
#    kmeans = KMeans(n_clusters=k, random_state=42)
#    cluster_labels = kmeans.fit_predict(data1)
#    silhouette_avg = silhouette_score(data1, cluster_labels)
#    silhouette_scores.append(silhouette_avg)
#
## Tracer le score de silhouette
#plt.figure(figsize=(8, 5))
#plt.plot(K, silhouette_scores, 'bo-')
#plt.xlabel('Nombre de clusters K')
#plt.ylabel('Score de silhouette')
#plt.title('Score de silhouette pour déterminer le nombre optimal de clusters')
#plt.show()
#
## Choix du nombre de clusters basé sur les résultats précédents, par exemple 3
#optimal_k = 3
#kmeans = KMeans(n_clusters=optimal_k, random_state=42)
#data1['cluster'] = kmeans.fit_predict(data1)
#
## Affichage des résultats
#plt.scatter(data1['haut_tot'], c=data1['cluster'], cmap='viridis')
#plt.xlabel('Hauteur totale (haut_tot)')
#plt.ylabel('?')
#plt.title('K-Means Clustering avec K optimal')
#plt.show()
#