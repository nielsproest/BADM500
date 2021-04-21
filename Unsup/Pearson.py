import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import sys

if False:
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

	matrix = X.corr()
	matrix.to_csv("pearson.csv",sep=";")

if False:
	dt = pd.read_csv("pearson.csv",sep=";")

	cap = 0.0 #0.9
	with open("pearson_comp_all.txt", "w") as f:
		seen = set()
		for i in range(1,len(dt.columns)):
			gene = dt.columns[i-1]
			for k,v in enumerate(dt[dt.columns[i]]):
				comp = dt.columns[k]
				pair = tuple(sorted([gene,comp]))
				if not pair in seen and k != (i-1) and v>=cap:
					seen.add(pair)
					print(gene,comp,v,file=f)

if True:
	content = []
	for i in open("pearson_comp_all.txt"):
		i = i.replace("Unnamed: 0", "this_is_an_error")
		#print(i.split(" "))
		content.append(float(i.split(" ")[2]))
		#WARNING: You will run out of space in the list
	content.sort()

	#Uses 32GB+
	#lines = open("pearson_comp_all.txt").readlines()
	#content = [k.split(" ") for k in lines]
	#content = [[k,v,float(i.strip())] for k,v,i in content]
	#content.sort(key=lambda x: x[2])
	content.reverse()
	#content=content[:300]

	plt.plot([i for k,v,i in content])
	plt.ylabel("Pearson's correlation coefficient")
	plt.savefig("pearsons_server.png",dpi=600)
