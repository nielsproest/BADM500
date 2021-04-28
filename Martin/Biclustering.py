import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import *
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import bicluster

df = pd.read_csv("kasMasterNew.csv",sep=';')

first_column = df.columns[0]
df = df.drop([first_column], axis=1)
feature_labels = df.columns
last = df.iloc[:,-1]
df = df.iloc[:, :-1]

print(df.head)
X = df
y = np.ravel(last.to_numpy())

#Run PCA :

X2 = df.values
print("PCA, Prior reduction shape: ", X2.shape)

pca = PCA(n_components=2, random_state=2020) 
pca.fit(X2)
X_pca_final = pca.transform(X2)
print("first column PCA :", X_pca_final.T[0])
df_new = pd.DataFrame(X_pca_final)
df_new.to_csv("PCA-2on5newclusters.csv", index=False)

#Run spectral bicluster
modelSC = bicluster.SpectralBiclustering(n_clusters=5, svd_method='randomized',n_svd_vecs=None, mini_batch=False, init='k-means++', n_init=10, random_state=42)

modelSC.fit(X2)

print("Classes of new classification : ", modelSC.row_labels_)

sns_plot = sns.scatterplot(x=X_pca_final.T[0], y=X_pca_final.T[1], data=X_pca_final,
                hue=modelSC.row_labels_, palette=sns.color_palette("husl", 5))
sns_plot.title('SpectralBiclustering - 5 clusters')
sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
sns_plot.figure.savefig('PCA-Bicluster.png', bbox_inches='tight', dpi=500)
plt.show()