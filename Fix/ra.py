import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

random_forest50 = ['AGR3', 'GABRP', 'FABP7', 'AGR2', 'VGLL1', 'ESR1', 'FOXA1', 'NAT1',
       'FLJ45983', 'PPP1R14C', 'LOC145837', 'CA12', 'MLPH', 'CNTNAP3', 'SYNM',
       'GATA3', 'ID4', 'FZD9', 'THSD4', 'UBE2C', 'FOXC1', 'KIF14', 'SIDT1',
       'NUF2', 'DLGAP5', 'FOXM1', 'SGOL1', 'HJURP', 'INPP4B', 'CEP55', 'TTK',
       'NFIB', 'BUB1B', 'MAML2', 'CMBL', 'SLC7A13', 'FAM171A1', 'AURKA',
       'C6orf211', 'FAM72B', 'CDKN3', 'C9orf116', 'KIF18A', 'CENPW', 'GINS1',
       'EZH2', 'MAD2L1', 'MTHFD1L', 'KPNA2', 'TLE3']

X = X.drop([i for i in X.columns.values if not i in random_forest50], axis=1)

model = RandomForestClassifier(n_estimators=400, n_jobs=-1, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', max_depth = 40)
kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=0)
results = cross_validate(model, X, Y, cv=kfold, return_estimator=True)

#print(results["test_score"])
#print(results)
#input("stop")

score = sum(results["test_score"]/len(results["test_score"]))
print("Score: ", score)
for i in range(len(results["estimator"])):
	if results["test_score"][i] == max(results["test_score"]):
		feat = {feature_labels[k]: float(v) for k,v in enumerate(results["estimator"][i].feature_importances_) if v > 0}
		break

with open("rand_{}.json".format(score), "w") as f:
	json.dump(feat, f)

"""import matplotlib as plt



plt.boxplot(model)
plt.title('RFE - Choosing 47 features with RF estimator',fontsize=10)
plt.xlabel('10 - 5Folds',fontsize=10)
plt.ylabel('Average score',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig("RFE-50-RF.png",dpi=500)
plt.show()"""