print("hello")
print("world")
print("coucou")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Chargement des données
data = pd.read_csv("C:/Users/jeanj/OneDrive/Bureau/projet_bid_data/Patrimoine_Arboré_(RO).csv")



# Sélection des colonnes pertinentes
data1 = data['X', 'Y', 'haut_tot', 'feuillage']
