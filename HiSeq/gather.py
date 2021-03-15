#import tensorflow as tf 
import pandas as pd
import numpy as np

"""types = {
	"Sample ID": np.str,
	"Cluster ID": np.int,
	"Age (Median)": np.int, #filter ["Normal"]
	"ER Status": np.str, #filter [nan]
	"PR Status": np.str, #filter [nan]
	"Her2 Status": np.str, #filter [nan]
	"PAM50": np.str, #filter ["Normal"]
	"Pathologic stage": np.str, #filter [nan]
	"Histological type": np.str, #filter [nan]
}"""

seq_info = pd.read_csv("../BRCA/BRCA_clinicalMatrix_filtered.csv",sep=";")
seq_labels = pd.read_csv("../PAM50/PAM50_filtered.csv",sep=';')
seq_values = pd.read_csv("HiSeqV2",sep='\t')

"""
>>> len(seq_info["sampleID"])
1229
>>> len(seq_labels["Sample ID"])
3989
>>> len(seq_values.columns.values)
1219
"""

values = []
labels = []

labels.append("sampleID")
for i in seq_values["sample"]:
	labels.append(i)
labels.append("PAM50_Label")

for index, row in seq_info.iterrows():
	data_id = row["sampleID"]

	if not data_id in seq_values:
		continue

	tbl = []
	tbl.append(data_id)

	#if 0 in seq_values[data_id]:
	#	continue

	for i in seq_values[data_id]:
		tbl.append(i)

	try:
		tbl.append(seq_labels.loc[seq_labels["Sample ID"] == data_id,"PAM50"].values[0])
	except:
		print(data_id)
		continue
	#tbl.append(row["PAM50Call_RNAseq"])

	values.append(tbl)

df = pd.DataFrame(np.array(values), columns=labels)
df.to_csv("master.csv", index=False, sep=";")

