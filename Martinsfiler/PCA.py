from numpy import where
import pandas as pd
import numpy as np# data visualzation
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

#df = pd.read_csv("mDF05.csv", sep=',')
#y = pd.read_csv("Classes05.csv")
df = pd.read_csv("matricen05-out.csv")

#df = pd.DataFrame(np.array(master))
""" columns = df.columns.tolist() # get the columns
cols_to_use = columns[:len(columns)-1] # drop the last one
df = pd.read_csv(np.array(master), usecols=cols_to_use) """
#df = df.iloc[:, :-1]
# Delete first
first_column = df.columns[0]
df = df.drop([first_column], axis=1)
last = df.iloc[:,-1] # last column of data frame (id)
#last.to_csv("last.csv")
df = df.iloc[:, :-1]
#df.to_csv('file.csv')

X = df.values
print("Prior reduction shape: ", X.shape)

""" scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X) """

pca = PCA(n_components=2, random_state=2020) 
pca.fit(X)
X_pca_95 = pca.transform(X)

print("95 percent shape: ", X_pca_95.shape)

print("Variance explained by first component = ", np.cumsum(pca.explained_variance_ratio_ * 100)[0])
print("Variance explained by first 2 components = ", np.cumsum(pca.explained_variance_ratio_ * 100)[1])
""" print("Variance explained by first 5 components = ", np.cumsum(pca.explained_variance_ratio_ * 100)[4])
print("Variance explained by first 10 components = ", np.cumsum(pca.explained_variance_ratio_ * 100)[9])
print("Variance explained by first 20 components = ", np.cumsum(pca.explained_variance_ratio_ * 100)[19])
print("Variance explained by first 30 components = ", np.cumsum(pca.explained_variance_ratio_ * 100)[29])
print("Variance explained by first 40 components = ", np.cumsum(pca.explained_variance_ratio_ * 100)[39])
print("Variance explained by first 100 components = ", np.cumsum(pca.explained_variance_ratio_ * 100)[99])
print("Variance explained by first 200 components = ", np.cumsum(pca.explained_variance_ratio_ * 100)[199])
print("Variance explained by first 300 components = ", np.cumsum(pca.explained_variance_ratio_ * 100)[299]) """

df_new = pd.DataFrame(X_pca_95)
df_new.to_csv("PCA-2compNew.csv", index=False)


""" plt.figure(figsize=(10,7))
sns.scatterplot(x=X_pca_12[:,0], y=X_pca_12[:,1], s = 70, palette=['green', 'blue'])
plt.title("2D scatterplot", pad = 15)
plt.xlabel("First pricipal component")
plt.ylabel("Second pricipal component")
plt.savefig("2D_scatterplot.png")
 """