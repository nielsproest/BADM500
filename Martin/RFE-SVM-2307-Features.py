import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("kasMasterNew.csv",sep=';')
#print("data shape: ", df.shape)
#print("data head: ", df.head)

first_column = df.columns[0]
df = df.drop([first_column], axis=1)
feature_labels = df.columns
last = df.iloc[:,-1]
df = df.iloc[:, :-1]

X = df
y = np.ravel(last.to_numpy()
            )
#print("y: ", y)
#print("data X: ", X)
#print("features: ",feature_labels[:-1])
yy = np.ravel(last.to_numpy())

import numpy as np
from sklearn.model_selection import StratifiedKFold
from random import *
import numpy as np

acc_score = []
cvlist = [x for x in range(1,11)]
model = list()

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scaledX = scale.fit_transform(X)

print("data X Scaled: ", scaledX)
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
estimator = SVC(kernel="linear")
selector = RFE(estimator, n_features_to_select=2307, step=100)
selector = selector.fit(scaledX, y)

print("Supprt: ", selector.support_)
print("Ranking", selector.ranking_)
f = selector.get_support(1) #the most important features
X3 = df[df.columns[f]] # final features

scaledX = scale.fit_transform(X3) #Get scaledX of selected features

for x in range(1,11):
    skf = StratifiedKFold(n_splits=10, random_state=randint(1, 1000), shuffle=True)
    skf.get_n_splits(scaledX, y)
    acc_score = []
    for train_index, test_index in skf.split(scaledX, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = scaledX[train_index], scaledX[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svclassifier = SVC(decision_function_shape='ovo', kernel='linear', class_weight='balanced', C=1, random_state=430)
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        #print(confusion_matrix(y_test,y_pred))
        #print(classification_report(y_test,y_pred))
        #print("accuracy_score: ", accuracy_score(y_test,svclassifier.predict(X_test)))
        acc_score.append(accuracy_score(y_test,svclassifier.predict(X_test)))
    model.append(acc_score)
               

print("Average score:", np.mean(acc_score))
print("Model:", model)
plt.boxplot(model)
plt.title('RFE - Choosing 2307 features with SVC estimator \n Then Fitting new SVC model on the selected features')
plt.xlabel('Different random initializations (10 folds)')
plt.ylabel('Average score - 10 folds')
plt.savefig("SVM-RFE-2307-features-ovo-balanced.png",dpi=500)
plt.show()