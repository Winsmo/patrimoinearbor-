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

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv("./Data_Arbre.csv")

# Sélection des colonnes pertinentes
data1 = data[['tronc_diam', 'haut_tot']]

# Choix du nombre de clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Appliquer le clustering
data1['cluster'] = kmeans.fit_predict(data1)

# Affichage des résultats
plt.scatter(data1['haut_tot'], data1['tronc_diam'], c=data1['cluster'], cmap='viridis')
plt.xlabel('haut_tot')
plt.ylabel('tronc_diam')
plt.title('K-Means Clustering')
plt.show()



