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

seq_info = pd.read_csv("BRCA_clinicalMatrix",sep='\t')
seq_labels = pd.read_csv("PAM50_filtered.csv",sep=';')
seq_values = pd.read_csv("HiSeqV2",sep='\t')

print(seq_labels.dtypes)

for index, row in seq_labels.iterrows():
	print("label: ",row)
	data_id = row["Sample ID"]
	info = seq_info.loc[seq_info["sampleID"] == data_id]
	print("info: ",info)
	print("genes: ",seq_values["sample"])
	print("gene values: ",seq_values[data_id])

	input("dance break")
