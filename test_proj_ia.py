import pandas as pd
import matplotlib.pyplot as plt
from fastcluster import linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram

# Chargement des données
data = pd.read_csv("./Data_Arbre.csv")

# Sélection des colonnes pertinentes et création d'une copie explicite
data1 = data[['haut_tot', 'tronc_diam']].copy()

# Vérification des types de données
print(data1.dtypes)

# Méthode du coude avec dendrogramme
Z = linkage(data1, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogramme pour déterminer le nombre de clusters')
plt.xlabel('Échantillons')
plt.ylabel('Distance')
plt.show()

# Calculer le score de silhouette pour différents nombres de clusters
silhouette_scores = []
K = range(2, 11)
for k in K:
    agglo = AgglomerativeClustering(n_clusters=k)
    cluster_labels = agglo.fit_predict(data1)
    silhouette_avg = silhouette_score(data1, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Tracer le score de silhouette
plt.figure(figsize=(8, 5))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Nombre de clusters K')
plt.ylabel('Score de silhouette')
plt.title('Score de silhouette pour déterminer le nombre optimal de clusters')
plt.show()

# Appliquer AgglomerativeClustering avec le nombre optimal de clusters
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # K commence à 2
agglo = AgglomerativeClustering(n_clusters=optimal_clusters)
data1['agglo_cluster'] = agglo.fit_predict(data1)

# Appliquer DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=5)
data1['dbscan_cluster'] = dbscan.fit_predict(data1)

# Affichage des résultats pour AgglomerativeClustering et DBSCAN
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data1['haut_tot'], data1['tronc_diam'], c=data1['agglo_cluster'], cmap='viridis')
plt.xlabel('haut_tot')
plt.ylabel('tronc_diam')
plt.title(f'Agglomerative Clustering (n_clusters={optimal_clusters})')

plt.subplot(1, 2, 2)
plt.scatter(data1['haut_tot'], data1['tronc_diam'], c=data1['dbscan_cluster'], cmap='viridis')
plt.xlabel('haut_tot')
plt.ylabel('tronc_diam')
plt.title('DBSCAN Clustering')

plt.tight_layout()
plt.show()

# Afficher la silhouette pour Agglomerative Clustering
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(data1) + (optimal_clusters + 1) * 10])

cluster_labels = data1['agglo_cluster']
silhouette_avg = silhouette_score(data1[['haut_tot', 'tronc_diam']], cluster_labels)
print(f"Le score de silhouette moyen pour Agglomerative Clustering avec {optimal_clusters} clusters est {silhouette_avg}")

sample_silhouette_values = silhouette_samples(data1[['haut_tot', 'tronc_diam']], cluster_labels)

y_lower = 10
for i in range(optimal_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / optimal_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10

ax1.set_title("Silhouette plot for Agglomerative Clustering")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")

ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])
ax1.set_xticks([i / 10.0 for i in range(-1, 11)])

plt.show()



