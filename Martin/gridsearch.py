from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from random import *
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile

df = pd.read_csv("kasMasterNew.csv",sep=';')
#print("data shape: ", df.shape)
#print("data head: ", df.head)

first_column = df.columns[0]
df = df.drop([first_column], axis=1)
feature_labels = df.columns
last = df.iloc[:,-1]
df = df.iloc[:, :-1]

X = df
y = np.ravel(last.to_numpy())


sel = SelectPercentile(mutual_info_classif, percentile=50).fit(X, y)
model = list()
    
X_t = sel.transform(X)

#print("y: ", y)
#print("data X: ", X)
#print("features: ",feature_labels[:-1])
#yy = np.ravel(last.to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X_t,y, test_size = 0.3, random_state = 0)
print(X_train.shape, X_test.shape)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

scale = StandardScaler()
scaler = scale.fit(X_train)
X_train1 = scaler.transform(X_train)
X_test1 = scaler.transform(X_test)

estimator = SVC(kernel="linear", decision_function_shape='ovo', class_weight='balanced', random_state=42)
selector = RFE(estimator, n_features_to_select=77, step=100)
selector = selector.fit(X_train1, y_train)

print("Support: ", selector.support_)
print("Ranking", selector.ranking_)

X_test2, y_test2 = X_test1, y_test

X_train11 = pd.DataFrame(X_train1)
X_test11 = pd.DataFrame(X_test2)

f = selector.get_support(1)

X_train = X_train11[X_train11.columns[f]]
X_test = X_test11[X_test11.columns[f]]


from sklearn.model_selection import GridSearchCV
param_dist = {'decision_function_shape' : ('ovo','ovr'), 'C' :[0.01, 0.05, 0.1, 0.5, 1,10,20,50,100], 'class_weight' : ('balanced', None)}

clf = SVC(kernel='linear',random_state=42)

#'class_weight' : {'Basal':1, 'Her2':1, 'LumA':1, 'LumB': [0.7, 1.3, 3, 5], 'Normal':1},
#from sklearn.model_selection import GridSearchCV

#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#svc = svm.SVC()
#clf = GridSearchCV(svc, parameters)
#clf.fit(iris.data, iris.target)

#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)

#print('Accuracy on test set: ')
#print(accuracy_score(y_test2, y_pred))
#print(confusion_matrix(y_test2,y_pred))
#print(classification_report(y_test2,y_pred))    

#svc = svm.SVC()
clf_cv = GridSearchCV(clf, param_dist)

#clf_cv = RandomizedSearchCV(clf, param_dist, cv = 59, random_state = 44)
clf_cv.fit(X_train, y_train)

# Print the tuned parameters and score
#print("Tuned Decision Parameters: {}".format(clf_cv.best_params_))
#print("Best score is {}".format(clf_cv.best_score_))
           
           
print("Tuned SCV Parameters: {}".format(clf_cv.best_params_)) 
print("Best score is {}".format(clf_cv.best_score_))   