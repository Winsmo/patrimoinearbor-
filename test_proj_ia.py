import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Chargement des données
data = pd.read_csv("./Data_Arbre.csv")

# Sélection des colonnes pertinentes et création d'une copie explicite
data1 = data[['haut_tot', 'longitude', 'latitude']].copy()

# Vérification des types de données
print(data1.dtypes)

# Méthode du coude (Elbow Method) et Davies-Bouldin Index
inertia = []
db_scores = []
silhouette_scores = []
K = range(2, 10)  # Davies-Bouldin index et silhouette score ne sont pas définis pour k=1
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data1)
    inertia.append(kmeans.inertia_)
    
    # Calculer l'indice de Davies-Bouldin pour chaque K
    labels = kmeans.labels_
    db_score = davies_bouldin_score(data1, labels)
    db_scores.append(db_score)
    
    # Calculer le coefficient de silhouette pour chaque K
    silhouette_avg = silhouette_score(data1, labels)
    silhouette_scores.append(silhouette_avg)

# Tracer la courbe d'inertie
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Nombre de clusters K')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.show()

# Tracer la courbe de l'indice de Davies-Bouldin
plt.figure(figsize=(8, 5))
plt.plot(K, db_scores, 'bo-')
plt.xlabel('Nombre de clusters K')
plt.ylabel('Indice de Davies-Bouldin')
plt.title('Indice de Davies-Bouldin pour déterminer le nombre optimal de clusters')
plt.show()

# Tracer la courbe du coefficient de silhouette
plt.figure(figsize=(8, 5))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Nombre de clusters K')
plt.ylabel('Coefficient de silhouette')
plt.title('Coefficient de silhouette pour déterminer le nombre optimal de clusters')
plt.show()

# Choisir le nombre optimal de clusters
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(data1)

# Ajouter les clusters aux données
data1['cluster'] = clusters

print(f"Le nombre optimal de clusters est : {optimal_k}")


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