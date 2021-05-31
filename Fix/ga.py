import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

df = pd.read_csv("../Kasperstuff/masterNew.csv",sep=';')
df.index = df["sampleID"]

Y = df["PAM50_Label"]
X = df.drop(["sampleID", "PAM50_Label"], axis=1)
feature_labels = X.columns

print(df)
print(X)
print(Y)

ga50 = ['MS4A14', 'SEMA4G', 'SFRP1', 'KRT14', 'MAMDC2', 'MOGAT2', 'NAT1', 'KRT20', 'PGAP3', 'UBE2C', 'ERBB2', 'THSD1', 'LEPREL1', 'KIF18B', 'LIF', 'MPHOSPH6', 'KDM4B', 'GATA3', 'AURKA', 'LOC145837', 'CEP55', 'C17orf37', 'LOC399959', 'MLPH', 'AGR3', 'KRT17', 'GRB7', 'KRT5', 'IL3RA', 'GNA12', 'MIA', 'FOXM1', 'PSAT1', 'PAMR1', 'MYBL2', 'TBC1D9', 'FAM83D', 'EXO1', 'FBXW7', 'TMEM220', 'XBP1', 'LOC728264', 'MS4A7', 'ESR1', 'SKA1', 'SPRY2', 'TPX2', 'HAAO', 'SMOC1', 'DSCC1']

X = X.drop([i for i in X.columns.values if not i in ga50], axis=1)

model = XGBClassifier(min_child_weight= 1, max_depth= 5, learning_rate= 0.2, colsample_bytree = 0.7, n_estimators=800, verbosity = 0)
kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=0)
results = cross_validate(model, X, Y, cv=kfold, return_estimator=True)

score = sum(results["test_score"]/len(results["test_score"]))
print("Score: ", score)
for i in range(len(results["estimator"])):
	if results["test_score"][i] == max(results["test_score"]):
		feat = {feature_labels[k]: float(v) for k,v in enumerate(results["estimator"][i].feature_importances_) if v > 0}
		break

with open("grad_{}.json".format(score), "w") as f:
	json.dump(feat, f)
