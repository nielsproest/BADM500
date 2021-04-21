import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

# Defining Model
model = TSNE(learning_rate=100)

# Fitting Model
transformed = model.fit_transform(X) #WRONG

# Plotting 2d t-Sne
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]

plt.scatter(x_axis, y_axis, c=y)
plt.savefig("TSNE.png", dpi=600)