import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Chargement des données
data = pd.read_csv("./Data_Arbre.csv")

# Sélection des colonnes pertinentes et création d'une copie explicite
data1 = data[['haut_tot', 'tronc_diam']].copy()

# Vérification des types de données
print(data1.dtypes)

# Méthode du coude (Elbow Method)
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

# Score de silhouette (Silhouette Score)
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

# Choix du nombre de clusters basé sur les résultats précédents, par exemple 3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data1['cluster'] = kmeans.fit_predict(data1)

# Affichage des résultats
plt.scatter(data1['haut_tot'], data1['tronc_diam'], c=data1['cluster'], cmap='viridis')
plt.xlabel('Hauteur totale (haut_tot)')
plt.ylabel('Diamètre du tronc (tronc_diam)')
plt.title('K-Means Clustering avec K optimal')
plt.show()

