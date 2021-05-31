from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

svm50 = ['ABCC2', 'AGR3', 'C11orf90', 'C19orf77', 'C6orf97', 'CCDC6',
       'CCNB1', 'CHRAC1', 'CITED4', 'CKS1B', 'CLEC4F', 'EFNA2', 'ERBB2',
       'ESR1', 'FADS1', 'FGD1', 'FGD3', 'FGFBP1', 'FOXC1', 'GLB1',
       'GPR160', 'KRT20', 'LAMC3', 'LCN1', 'LCT', 'LOC145837', 'MAG',
       'MAPT', 'MBOAT1', 'MIA', 'NPFFR1', 'NUDT9P1', 'OLAH', 'PHYHD1',
       'PIF1', 'PPP4R4', 'RETN', 'RNASE3', 'SLC6A11', 'SLC6A2', 'SMC4',
       'SPAG5', 'SPOCK3', 'TAC1', 'TACC2', 'TCAM1P', 'TCEAL7', 'TCF19',
       'TMEM194B', 'WDR67']

ga50 = ['MS4A14', 'SEMA4G', 'SFRP1', 'KRT14', 'MAMDC2', 'MOGAT2', 'NAT1', 'KRT20', 'PGAP3', 'UBE2C', 'ERBB2', 'THSD1', 'LEPREL1', 'KIF18B', 'LIF', 'MPHOSPH6', 'KDM4B', 'GATA3', 'AURKA', 'LOC145837', 'CEP55', 'C17orf37', 'LOC399959', 'MLPH', 'AGR3', 'KRT17', 'GRB7', 'KRT5', 'IL3RA', 'GNA12', 'MIA', 'FOXM1', 'PSAT1', 'PAMR1', 'MYBL2', 'TBC1D9', 'FAM83D', 'EXO1', 'FBXW7', 'TMEM220', 'XBP1', 'LOC728264', 'MS4A7', 'ESR1', 'SKA1', 'SPRY2', 'TPX2', 'HAAO', 'SMOC1', 'DSCC1']
ra50 = ['AGR3', 'GABRP', 'FABP7', 'AGR2', 'VGLL1', 'ESR1', 'FOXA1', 'NAT1',
       'FLJ45983', 'PPP1R14C', 'LOC145837', 'CA12', 'MLPH', 'CNTNAP3', 'SYNM',
       'GATA3', 'ID4', 'FZD9', 'THSD4', 'UBE2C', 'FOXC1', 'KIF14', 'SIDT1',
       'NUF2', 'DLGAP5', 'FOXM1', 'SGOL1', 'HJURP', 'INPP4B', 'CEP55', 'TTK',
       'NFIB', 'BUB1B', 'MAML2', 'CMBL', 'SLC7A13', 'FAM171A1', 'AURKA',
       'C6orf211', 'FAM72B', 'CDKN3', 'C9orf116', 'KIF18A', 'CENPW', 'GINS1',
       'EZH2', 'MAD2L1', 'MTHFD1L', 'KPNA2', 'TLE3']


dd = ga50

df = pd.read_csv("../Kasperstuff/masterNew.csv",sep=';')
df.index = df["sampleID"]

y = df["PAM50_Label"]
X = df.drop(["sampleID", "PAM50_Label"], axis=1)
feature_labels = X.columns

X_t=df[np.unique(dd)].to_numpy()

model = list()
X_t=df[np.unique(dd)].to_numpy()

for x in range(1,11):
	acc_score = []

	X_t=df[np.unique(dd)].to_numpy()

	evaluator = XGBClassifier(min_child_weight= 1, random_state=x, max_depth= 5, learning_rate= 0.2, colsample_bytree = 0.7, n_estimators=800, verbosity = 0)
	#evaluator = RandomForestClassifier(n_estimators=400, random_state=x,n_jobs=-1, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', max_depth = 40)
	#evaluator = SVC(kernel="linear", random_state=x, C = 0.1,decision_function_shape='ovo', class_weight = None)

	clf = make_pipeline(StandardScaler(), evaluator)
	cv = ShuffleSplit(n_splits=5, test_size=0.3)
	model.append(cross_val_score(clf, X_t, y, cv=cv).tolist())
	#print("Average score:", np.mean(acc_score))
	#model.append(acc_score)             

print("Model:", model)
print("Model average :", np.mean(model))
plt.boxplot(model)
plt.title('GA-RFE Final evaluation',fontsize=10)
plt.xlabel('10 - 5Folds',fontsize=10)
plt.ylabel('Average score',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig("GA-RFE-50-features-ovo-21-5newnew.png",dpi=500)
#plt.show()