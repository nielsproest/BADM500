

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import *
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn_extra.cluster import KMedoids

df = pd.read_csv("kasMasterNew.csv",sep=';')

first_column = df.columns[0]
df = df.drop([first_column], axis=1)
feature_labels = df.columns
last = df.iloc[:,-1]
df = df.iloc[:, :-1]

X = df
y = np.ravel(last.to_numpy())

#Run kMedios

kmedoids = KMedoids(n_clusters=5, random_state=0, method='pam').fit(X)
print("Classes of new classification : ", kmedoids.labels_)
print("cluster Centers : ", kmedoids.cluster_centers_)

#Run PCA :

X2 = df.values
print("PCA, Prior reduction shape: ", X2.shape)

pca = PCA(n_components=2, random_state=2020) 
pca.fit(X2)
X_pca_final = pca.transform(X2)
print("first column PCA :", X_pca_final.T[0])
df_new = pd.DataFrame(X_pca_final)
df_new.to_csv("PCA-2on5newclusters.csv", index=False)

#Show clusters on 2 dimensial PCA
sns_plot = sns.scatterplot(x=X_pca_final.T[0], y=X_pca_final.T[1], data=X_pca_final,
                hue=kmedoids.labels_, palette=sns.color_palette("husl", 5))
sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
plt.show()
sns_plot.figure.savefig('PCA-Cluster.png', bbox_inches='tight', dpi=500)
