import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# Charger les données
data = pd.read_csv("./Data_Arbre.csv")

# Sélection des colonnes pertinentes
data1 = data[['haut_tot', 'tronc_diam']].copy()

# Vérification des types de données
print(data1.dtypes)

# Méthode du coude pour déterminer le nombre optimal de clusters
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data1)
    inertia.append(kmeans.inertia_)

# Tracer la courbe d'inertie
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Nombre de clusters K')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.show()

# Calculer le score de silhouette pour différents nombres de clusters
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(data1)
    silhouette_avg = silhouette_score(data1, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Tracer le score de silhouette
plt.figure(figsize=(8, 5))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Nombre de clusters K')
plt.ylabel('Score de silhouette')
plt.title('Score de silhouette pour déterminer le nombre optimal de clusters')
plt.show()

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
data1['cluster_agglo'] = agglo.fit_predict(data1[['haut_tot', 'tronc_diam']])
plt.figure(figsize=(8, 5))
plt.scatter(data1['haut_tot'], data1['tronc_diam'], c=data1['cluster_agglo'], cmap='viridis')
plt.xlabel('haut_tot')
plt.ylabel('tronc_diam')
plt.title('Agglomerative Clustering')
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=1, min_samples=5)
data1['cluster_dbscan'] = dbscan.fit_predict(data1[['haut_tot', 'tronc_diam']])
plt.figure(figsize=(8, 5))
plt.scatter(data1['haut_tot'], data1['tronc_diam'], c=data1['cluster_dbscan'], cmap='viridis')
plt.xlabel('haut_tot')
plt.ylabel('tronc_diam')
plt.title('DBSCAN Clustering')
plt.show()


