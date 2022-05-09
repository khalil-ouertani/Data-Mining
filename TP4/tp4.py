from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import cluster, metrics


fromage=pd.read_table(r"C:\Users\moham\Downloads\fromage1.txt",sep="\t",header=0,index_col=0)
#print(fromage)

# Croisement 2 à 2 des variables
pd.plotting.scatter_matrix(fromage, figsize=(9,9), diagonal='kde')
plt.show()

np.random.seed(0)
kmeans = cluster.KMeans(n_clusters = 4)
kmeans.fit(fromage)

#index triés des groupes
idk = np.argsort(kmeans.labels_)
#affichage des observations et leurs groupes
kmeans_tab = pd.DataFrame(fromage.index[idk],kmeans.labels_[idk])
print(kmeans_tab)

# L2 norm distance to each cluster center
print(kmeans.transform(fromage))

# Utilisation de la métrique "silhouette"
res = np.arange(9, dtype ="double")
for k in np.arange(9):
    km = cluster.KMeans(n_clusters = k+2)
    km.fit(fromage)
    res[k] = metrics.silhouette_score(fromage,km.labels_)

print (res)

# Graphique
plt.title("silhouette")
plt.xlabel("# of clusters")
plt.plot(np.arange(2,11,1),(res))
plt.show()
# Meilleur nombre de clusters = 3/5


# --------------- CAH --------------- 
print("---------- CAH ----------")


#librairies pour la CAH
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

Z = linkage(fromage,method='ward', metric='euclidean')

#affichage du dendrogramme
plt.title("CAH")
plt.title('CAH avec matérialisation des 4 classes')
dendrogram(Z,labels=fromage.index,orientation='left',color_threshold=255)
plt.show()

groupes_cah = fcluster(Z, t = 255,criterion='distance')
print(groupes_cah)

#index triés des groupes
idg = np.argsort(groupes_cah)
#affichage des observations et leurs groupes
cah_tab = pd.DataFrame(fromage.index[idg],groupes_cah[idg])
print(cah_tab)

""" print("---------- CROSSTAB ---------")
ct = pd.crosstab(kmeans_tab, cah_tab) """


# ---------- PCA ----------
print("---------- PCA ----------")

from sklearn.decomposition import PCA

acp = PCA(n_components =2).fit_transform(fromage)
for couleur,k in zip(['red','blue','lawngreen', 'aqua'],[0,1,2,3]):
    plt.scatter(acp[km.labels_==k,0],acp[km.labels_==k,1],c=couleur)
plt.show()


# ---------- CAH Agglomerative ----------
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
cluster.fit_predict(fromage)

# --------- DIANA ----------
print("--------- DIANA ----------")
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram

# Z va contenir notre 'linkage'
Z = []

# Lancer l'algorithme
def run(data):
  # Initialization de Z
  global Z
  Z = []
  index = split(data,data)
  fig = plt.figure(figsize=(25, 10))
  dn = dendrogram(Z,labels=data.index, color_threshold=14000)
  plt.show()

# Retourne l'index d'une ligne du dataset
def get_index(line, full_dataset):
  indexes = np.argwhere(full_dataset.index==line.index[0])
  return indexes[0][0]

# Retourne la valeur d'un élément
def get_value(index,full_dataset):
  n = len(full_dataset)
  if(index >= n):
    return Z[index-n][2]
  else:
    return sum(full_dataset.iloc[index])

# Retourne la distance entre deux éléments
def get_distance(index1, index2,full_dataset):
  return get_value(index1,full_dataset) + get_value(index2,full_dataset)

# Retourne le nombre des éléments du dataset original
# appartenant à ce cluster
def get_originals(index,full_dataset):
  n = len(full_dataset)
  if(index >= n):
    return Z[index-n][3]
  else:
    return 1

# split() est une fonction récursive (diviser pour régner) qui permet
# de diviser une dataset en deux sous-groupes en utilisant K-means
# jusqu'à qu'on arrive à des sous-groupes composés d'un seul élément
# puis on alimente Z
def split(data, full_dataset):
  n = len(full_dataset)
  if len(data) < 2:
          return get_index(data,full_dataset)
  
  # Initialiser l'algorithme K-means
  km = KMeans(n_clusters=2)
  km.fit(data)
  
  class1 = data[km.labels_==0]
  class2 = data[km.labels_==1]
  index1 = split(class1,full_dataset)
  index2 = split(class2,full_dataset)

  # Ajouter une ligne dans le tableau Z
  Z.append([index1, # Index du premier sous-groupe
            index2, # Index du deuxiéme sous-groupe
            get_distance(index1, index2,full_dataset), # La valeur du groupe
            get_originals(index1,full_dataset)+get_originals(index2,full_dataset)]) # Le nombre des originaux du groupe
  
  # Retourner l'index de l'élément
  return len(Z)-1+n

run(fromage)