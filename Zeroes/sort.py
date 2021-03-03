# evaluate pca with logistic regression algorithm for classification
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import pandas as pd
# define dataset
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)

#Load data
print("Load data")
df = pd.read_csv("../HiSeq/master.csv",sep=';')
#print(dt.head())
'''
feature_cols = [dt.columns.values[x] for x in range(1,len(dt)-1)]
X = dt[feature_cols] # Features
y = dt[dt.columns.values[-1]] # Target variable'''

types = df.columns.values
genes = types[1:-1]
samples = list(df["sampleID"])

print(types[:5])
print(genes[:5])
print(samples[:5])

zerosarr = []

for column in df.columns[1:]:
#    print(df[column])
    y = df[column].to_numpy()
    num_zeros = (y == 0).sum()
    zerosarr.append(num_zeros)

zerosarr.sort()
zerospct = [(x/1144)*100 for x in zerosarr]

plt.plot(zerosarr)
# x-axis label 
plt.xlabel('genes') 
# frequency label 
plt.ylabel('# of zeroes') 

plt.annotate('27 zeroes, or 2.4%', xy=(15000,28), xytext=(8000,110),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('10% of genes have >857 zero-values', xy=(18478,857), xytext=(2000,600),
            arrowprops=dict(facecolor='black', shrink=0.05))


plt.annotate('More than 1400 genes (7%) have >90% zero-values', xy=(19100,1032), xytext=(1000,1000),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.savefig("test.png")
# var() is variance, sorting by highest variance first.
#valcounts = dt.value_counts()
#print(valcounts[1])


'''
# taking the indexes from top50 and reinserting ID's into the index
variance50 = variance[:50].index.insert(0, "Sample ID")
print(variance)
# filtering the data by the genes in the top 50
# Result: 2237 rows x 50 columns
data50 = dt.filter(variance50)

df = pd.DataFrame(np.array(data50), columns=variance50)
df.to_csv("master1234.csv", index=False)'''