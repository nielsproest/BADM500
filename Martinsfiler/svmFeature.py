from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("matricen50.csv")
print("data shape: ", df.shape)
print("data head: ", df.head)

first_column = df.columns[0]
df = df.drop([first_column], axis=1)
feature_labels = df.columns
last = df.iloc[:,-1]
df = df.iloc[:, :-1]

X = df.values
print("data X: ", X)
y = np.ravel(last.to_numpy())
print("y: ", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("accuracy_score: ", accuracy_score(y_test,svclassifier.predict(X_test)))


def f_importances(coef, names, top):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))
    print("Top 40 features names: ", names[::-1][0:top])
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top],fontsize=7)
    plt.savefig("SVM-feature-importance2.png")
    print("coef shape : ", svclassifier.coef_.shape)

f_importances(abs(svclassifier.coef_[0]), feature_labels, top=40)

""" print("coef med abs ", abs(svclassifier.coef_[0]))
print("coef uden abs",(svclassifier.coef_[0]))
print("coef uden abs",(svclassifier.coef_)) """
