import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import sys

#Load data
print("Load data")
#dt = pd.read_csv("../HiSeq/master.csv",sep=";")
dt = pd.read_csv("../Kasperstuff/masterNew.csv",sep=";")
print(dt.head())

#The martin way
first_column = dt.columns[0]
X = dt.drop([first_column], axis=1)
feature_labels = X.columns
y = X.iloc[:,-1]
X=X.iloc[:, :-1]
print(X)
print(y)

y_labels = [ 
	'Normal',
	'Basal',
	'Her2',
	'LumA',
	'LumB'
]

y = np.array([y_labels.index(i) for i in y])

#KMeans
#km = KMeans(n_clusters=5).fit(X)
#km.predict(X)
#print(km.labels_[::10])
#print(set(km.labels_))

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

kmeans_kwargs = {
	"init": "random",
	"n_init": 10,
	"max_iter": 300,
	"random_state": 42,
}

# Notice you start at 2 clusters for silhouette coefficient
man = 20
for k in range(2, man):
	kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
	kmeans.fit(scaled_features)
	score = silhouette_score(scaled_features, kmeans.labels_)
	silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(2, man), silhouette_coefficients)
plt.xticks(range(2, man))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
#plt.show()
plt.savefig("KMeans.png", dpi=600)