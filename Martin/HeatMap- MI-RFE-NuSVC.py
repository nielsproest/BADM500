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
from sklearn.svm import NuSVC

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

# run mutual information til 60 % reduction
sel = SelectPercentile(mutual_info_classif, percentile=60).fit(X, y)
model = list()    
X_t = sel.transform(X)

from sklearn.svm import NuSVC
model = list()

for x in range(1,2):
    folds = 2
    skf = StratifiedKFold(n_splits=folds, random_state=randint(1, 1000), shuffle=True, )
    skf.get_n_splits(X_t, y)
    acc_score = []
    for train_index, test_index in skf.split(X_t, y):
        X_train, X_test = X_t[train_index], X_t[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scale = StandardScaler()
        scaler = scale.fit(X_train)
        X_train1 = scaler.transform(X_train)
        X_test1 = scaler.transform(X_test)
        
        estimator = NuSVC(kernel='linear', nu = 0.2, class_weight= 'balanced', decision_function_shape = 'ovo', random_state=42)
        #estimator = SVC(kernel="linear", decision_function_shape='ovo', cache_size=200, class_weight={'Basal':1, 'Her2':1, 'LumA':1, 'LumB':1.3, 'Normal':1}, random_state=randint(1, 10000))
        selector = RFE(estimator, n_features_to_select=47, step=100)
        selector = selector.fit(X_train1, y_train)

        print("Support: ", selector.support_)
        print("Ranking", selector.ranking_)

        #from imblearn.over_sampling import SMOTE
        #oversample = SMOTE()
        X_test2, y_test2 = X_test1, y_test
        #X_test2, y_test2 = oversample.fit_resample(X_test1, y_test)

        X_train11 = pd.DataFrame(X_train1)
        X_test11 = pd.DataFrame(X_test2)

        f = selector.get_support(1)

        X_train = X_train11[X_train11.columns[f]]
        X_test = X_test11[X_test11.columns[f]]

        clf = NuSVC(kernel='linear', nu = 0.2, class_weight= 'balanced', decision_function_shape = 'ovo', random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        #print('Accuracy on test set: ')
        #print(accuracy_score(y_test2, y_pred))
        print(confusion_matrix(y_test2,y_pred))
        #print(classification_report(y_test2,y_pred))
        
        acc_score.append(accuracy_score(y_test2, y_pred))
    model.append(acc_score)
    print("Average score:", np.mean(acc_score))
    #model.append(acc_score)             

print("Model:", model)

plt.boxplot(model)
plt.title('RFE - Choosing 46 features with NuSVC estimator \n Then Fitting new NuSVC model on the selected features')
plt.xlabel('Different random initializations (folds)')
plt.ylabel('Average score - folds')
plt.savefig("MI-NuSVM-RFE-46-features-ovo-balanced2233.png",dpi=500)
plt.show()

#Create HeatMap

pam50= ['ACTR3B','ANLN','BAG1','BCL2','BIRC5','BLVRA','CCNB1','CCNE1','CDC20','CDC6','CDH3','CENPF','CEP55','CXXC5','EGFR','ERBB2','ESR1','EXO1','FGFR4','FOXA1','FOXC1','KIF2C','KRT14','KRT17','KRT5','MAPT','MDM2','MELK','MIA','MKI67','MLPH','MMP11','MYBL2','MYC','NAT1','ORC6L','PGR','PHGDH','PTTG1','RRM2','SFRP1','SLC39A6','TMEM45B','TYMS','UBE2C','UBE2T']

best77 = df[df.columns[f]] # of latest best features
p = df[pam50]

from numpy.random import seed
from scipy.stats import pearsonr

# seed random number generator
seed(1)
# prepare data
#best77 = best77.iloc[:, 0:2]
#p = p.iloc[:, 0:2]
l = list()
# calculate Pearson's correlation
col = list()
for ind, colum in enumerate(best77.columns):
    #corr, _ = pearsonr(p.columns[ind], column)
    #l.appende(corr)
    for ins, column in enumerate(p.columns):
        #print(colum)
        #print(column)
        #print(p.columns[ind])
        #print(ind)
        #print(best77.columns[ins])
        #print(ins)
        corr, _ = pearsonr(p[column], best77[colum])
        #print(abs(corr))
        col.append(abs(corr))
    l.append(col)
    col=[] 
#print(l)  

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

fig, ax = plt.subplots(figsize=(46, 46))
#plt.xlabel(best77.columns)
#plt.ylabel(p.columns)

#fig.title('Pearson Absolute Values', fontsize=40)

x_axis_labels = best77.columns.to_numpy() # labels for x-axis
y_axis_labels =  p.columns.to_numpy()# labels for y-axis

# create seabvorn heatmap with required labels
b = sb.heatmap(np.array(l).T.tolist(),  xticklabels=x_axis_labels, yticklabels=y_axis_labels,
           linewidth=0.4, cbar_kws={"shrink": .8}, fmt=".2f",square=True)
b.axes.set_title("SVM Feature selection - Pam50 vs Best47\n ",fontsize=60)
b.set_xlabel("Best 47",fontsize=40)
b.set_ylabel("Pam50",fontsize=40)
b.tick_params(labelsize=30)
sb.set(font_scale=3)
plt.show()
b.savefig('HeatMap-Pam50vsFeatureSelectionBest22.jpg',dpi=400)