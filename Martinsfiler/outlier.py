# data preparation
from numpy import where
import pandas as pd
import numpy as np# data visualzation
import matplotlib.pyplot as plt
import seaborn as sns# outlier/anomaly detection
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn import svm

#d = pd.read_csv("dou.csv")

d = pd.read_csv("PCA-comp2.csv")
#df = pd.DataFrame(np.array(d), columns = ["x", "y"])
"Component 1"
df = pd.DataFrame(np.array(d), columns = ["Component 1", "Component 2"])
outliers_fraction = 0.05
# model specification
model0 = EllipticEnvelope(contamination=outliers_fraction)
#model1 = LocalOutlierFactor(n_neighbors = 2, metric = "manhattan", contamination = 0.001)# model fitting
#model2 = IsolationForest(contamination=outliers_fraction, random_state=42)
#model3 = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",gamma=0.1)
y_pred = model0.fit_predict(df)# filter outlier index
outlier_index = where(y_pred == -1) # negative values are outliers and positives inliers# filter outlier values
outlier_values = df.iloc[outlier_index]# plot data
print(outlier_values)
""" plt.scatter(df["x"], df["y"], color = "b", s = 5)# plot outlier values
plt.scatter(outlier_values["x"], outlier_values["y"], color = "r", s = 5)
plt.title("Outlier scatterplot", pad = 15)
plt.xlabel("MCM10")
plt.ylabel("THSD4")

plt.savefig("Outlier_scatterplot.png")
#plt.show() """

