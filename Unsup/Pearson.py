import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import sys

"""#Load data
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

matrix = X.corr()
matrix.to_csv("pearson.csv",sep=";")"""


dt = pd.read_csv("pearson.csv",sep=";")

with open("pearson_comp.txt", "w") as f:
	seen = set()
	for i in range(1,len(dt.columns)):
		gene = dt.columns[i-1]
		for k,v in enumerate(dt[dt.columns[i]]):
			comp = dt.columns[k]
			pair = tuple(sorted([gene,comp]))
			if not pair in seen and k != (i-1) and v>=0.9:
				seen.add(pair)
				print(gene,comp,v,file=f)
