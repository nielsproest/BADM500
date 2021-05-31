import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFE

df = pd.read_csv("../Kasperstuff/masterNew.csv",sep=';')
df.index = df["sampleID"]

Y = df["PAM50_Label"]
X = df.drop(["sampleID", "PAM50_Label"], axis=1)
feature_labels = X.columns

print(df)
print(X)
print(Y)

"""estimator = SVC(kernel="linear")
selector = RFE(estimator, n_features_to_select=50, step=1000, verbose=1)
selector = selector.fit(X, Y)

print("Support: ", selector.support_)
print("Ranking", selector.ranking_)

f = selector.get_support(1)

X = df.drop([i for i in X.columns.values if not i in f], axis=1)"""
#svm50 = ['TCEAL8', 'WDR67', 'KRT20', 'ERBB2', 'FAM189A2', 'PRPH2', 'PPIL5', 'INTS4L1', 'TEX19', 'PPP1R14C', 'SPAG16', 'LCN6', 'CKS1B', 'SOHLH1', 'ZIC2', 'MIA', 'PNLIPRP2', 'XRRA1', 'RACGAP1P', 'POU3F3', 'CLDN18', 'RCVRN', 'RGPD1', 'PLA2G10', 'FGFBP1', 'FBXO5', 'MBOAT1', 'CNIH3', 'FOXC1', 'PARP2', 'TMEM59L', 'FADS1', 'MAGEB18', 'TCAM1P', 'TCF19', 'SCN3B', 'RNF2', 'TYRP1', 'TMEM194B', 'PAGE5', 'RHO', 'ABCC2', 'PPY2', 'PGR', 'HMGB2', 'CES7', 'TRIM6-TRIM34', 'EFNA2', 'LGSN', 'ZNF451']
svm50 = ['ABCC2', 'AGR3', 'C11orf90', 'C19orf77', 'C6orf97', 'CCDC6',
       'CCNB1', 'CHRAC1', 'CITED4', 'CKS1B', 'CLEC4F', 'EFNA2', 'ERBB2',
       'ESR1', 'FADS1', 'FGD1', 'FGD3', 'FGFBP1', 'FOXC1', 'GLB1',
       'GPR160', 'KRT20', 'LAMC3', 'LCN1', 'LCT', 'LOC145837', 'MAG',
       'MAPT', 'MBOAT1', 'MIA', 'NPFFR1', 'NUDT9P1', 'OLAH', 'PHYHD1',
       'PIF1', 'PPP4R4', 'RETN', 'RNASE3', 'SLC6A11', 'SLC6A2', 'SMC4',
       'SPAG5', 'SPOCK3', 'TAC1', 'TACC2', 'TCAM1P', 'TCEAL7', 'TCF19',
       'TMEM194B', 'WDR67']

X = X.drop([i for i in X.columns.values if not i in svm50], axis=1)

model = SVC(kernel="linear", C=0.1, decision_function_shape='ovo', class_weight=None, random_state=42)
kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=0)
results = cross_validate(model, X, Y, cv=kfold, return_estimator=True)

score = sum(results["test_score"]/len(results["test_score"]))
print("Score: ", score)

"""for i in range(len(results["estimator"])):
	if results["test_score"][i] == max(results["test_score"]):
		feat = {feature_labels[k]: float(v) for k,v in enumerate(results["estimator"][i].feature_importances_) if v > 0}
		break"""

with open("svm_{}.json".format(score), "w") as f:
	json.dump({k: 1 for k in svm50}, f)
