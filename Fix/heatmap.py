import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

name = "rand_0.8802378984109488"
fine_name = "Random Forest"

with open("{}.json".format(name)) as f:
	data = json.load(f)
	data = [[k,data[k]] for k in data]
	data.sort(key=lambda x: x[1])
	ourmap = data[-50:]
	#print(ourmap)
feat = [i[0] for i in ourmap]

pam = [i.strip() for i in open("pam50.txt")]

#print(len(feat))
#print(len(pam))

dt = pd.read_csv("../Kasperstuff/masterNew.csv",sep=";")

d = {}
for i in feat:
	d[i] = dt[i]
for i in pam:
	d[i] = dt[i]

df = pd.DataFrame(data=d)

print(df)
matrix = df.corr()

matrix = matrix.drop([i for i in pam if not i in feat], axis=1)
matrix = matrix.drop([i for i in feat if not i in pam], axis=0)
matrix = matrix.abs()
matrix = matrix.reindex(sorted(matrix.columns), axis=1)
matrix = matrix.reindex(sorted(matrix.index), axis=0)

print(matrix)

ax = sns.heatmap(matrix, xticklabels=True, yticklabels=True)

plt.title("{} - Feature selection PAM50 vs Best50".format(fine_name))
plt.xlabel("Best 50")
plt.ylabel("PAM50")
plt.show()
#plt.savefig("{}.png".format(name),dpi=600)

seen = {}

for index, row in matrix.iterrows():
	for column in matrix.columns.values:
		val = row[column]
		if not name in seen and val >= 0.75:

			if row.name in seen and seen[row.name] > val:
				continue
			#print(row.name, column, val)

			seen[row.name] = val

"""def maxi(vals):
	big = 0
	bigi = 0
	for k,v in enumerate(vals):
		if v > big:
			big = v
			bigi = k
	return bigi

for column in matrix.columns.values:
	idx = maxi(matrix[column])
	row = matrix.index[idx]
	val = matrix[column][row]
	if val >= 0.75:
		seen[column] = val"""

with open("{}_corr.json".format(name), "w") as f:
	json.dump(seen, f, indent="\t")

