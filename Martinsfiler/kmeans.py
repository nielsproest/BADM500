from sklearn.cluster import kmeans_plusplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

dt = pd.read_csv("PCA-2compNew.csv")
X = dt.values

# Incorrect number of clusters
y_pred = KMeans(n_clusters=3, init ='k-means++',random_state=42).fit_predict(X)



sns_plot = sns.scatterplot(X[:, 0], X[:, 1], c=y_pred, palette=sns.color_palette("husl", 5))
sns_plot.figure.savefig('New-kmeans.png', dpi=500)
