#import tensorflow as tf 
import pandas as pd
import numpy as np

print("Load data")
df = pd.read_csv("HiSeqV2",sep='\t')
print("Process")

values = []
labels = []

for i in df.loc[:0]:
	labels.append(i)

for index, row in df.iterrows():
	vals = [row[i] for i in labels]

	failed = False
	for j in range(1,len(vals)):
		i = vals[j]
		if ("{:.4f}".format(i) == "0.0000"):
			failed = True
			break
	if failed:
		continue

	values.append(vals)

df = pd.DataFrame(np.array(values), columns=labels)
df.to_csv("HiSeqV2_filtered.csv", index=False)