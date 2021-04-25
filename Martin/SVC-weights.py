import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from random import *    
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
#Mutual information
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

sel = SelectPercentile(mutual_info_classif, percentile=60).fit(X, y)
model = list()
    
X_t = sel.transform(X)

model = list()

for x in range(1,4):
    folds = 5
    skf = StratifiedKFold(n_splits=folds, random_state=N)
    skf.get_n_splits(X_t, y)
    acc_score = []
    
    for train_index, test_index in skf.split(X_t, y):
        X_train, X_test = X_t[train_index], X_t[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        scale = StandardScaler()
        scaler = scale.fit(X_train)
        X_train1 = scaler.transform(X_train)
        X_test1 = scaler.transform(X_test)

        estimator = SVC(kernel="linear", decision_function_shape='ovo', cache_size=200, class_weight={'Basal':1, 'Her2':1.1, 'LumA':1, 'LumB':1.2, 'Normal':1}, random_state=randint(1, 10000))
        selector = RFE(estimator, n_features_to_select=2307, step=100)
        selector = selector.fit(X_train1, y_train)

        print("Support: ", selector.support_)
        print("Ranking", selector.ranking_)

        X_test2, y_test2 = X_test1, y_test
        
        X_train11 = pd.DataFrame(X_train1)
        X_test11 = pd.DataFrame(X_test2)

        f = selector.get_support(1)

        X_train = X_train11[X_train11.columns[f]]
        X_test = X_test11[X_test11.columns[f]]

        clf = SVC(decision_function_shape='ovo',kernel='linear', class_weight={'Basal':1, 'Her2':1, 'LumA':1, 'LumB':, 'Normal':1},random_state=randint(1, 10000))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        #print('Accuracy on test set: ')
        print(accuracy_score(y_test2, y_pred))
        print(confusion_matrix(y_test2,y_pred))
        #print(classification_report(y_test2,y_pred))
        
        acc_score.append(accuracy_score(y_test2, y_pred))
    model.append(acc_score)
    print("Average score:", np.mean(acc_score))
    #model.append(acc_score)             

print("Model:", model)

plt.boxplot(model)
plt.title('RFE - Choosing 2307 features with SVC estimator \n Then Fitting new SVC model on the selected features')
plt.xlabel('Different random initializations (folds)')
plt.ylabel('Average score - folds')
plt.savefig("MI-SVM-RFE-2307-features-ovo-LumbWeight-SmoteAll.png",dpi=500)
plt.show()