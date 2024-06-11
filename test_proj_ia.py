import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# Chargement des données
data = pd.read_csv("./Data_Arbre.csv")

# Sélection des colonnes pertinentes et création d'une copie explicite
data1 = data[['haut_tot','tronc_diam']].copy()

# Vérification des types de données
print(data1.dtypes)

# Choix du nombre de clusters
agglo = AgglomerativeClustering(n_clusters=3)

# Appliquer le clustering
data1['cluster'] = agglo.fit_predict(data1[['haut_tot','tronc_diam']])

# Affichage des résultats
plt.scatter(data1['haut_tot'], data1['tronc_diam'], c=data1['cluster'], cmap='viridis')
plt.xlabel('haut_tot')
plt.ylabel('tronc_diam')
plt.title('Agglomerative Clustering')
plt.show()


