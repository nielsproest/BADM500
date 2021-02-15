import pandas as pd
import numpy as np


types = {
	"Sample ID": np.str,
	"Cluster ID": np.int,
	"Age (Median)": np.int, #filter ["Normal"]
	"ER Status": np.str, #filter [nan]
	"PR Status": np.str, #filter [nan]
	"Her2 Status": np.str, #filter [nan]
	"PAM50": np.str, #filter ["Normal"]
	"Pathologic stage": np.str, #filter [nan]
	"Histological type": np.str, #filter [nan]
}

pd.options.display.max_colwidth = 200
#filter by male, metastatic, unknown tissue, normal
seq_data = pd.read_csv("BRCA_clinicalMatrix",sep='\t')

seq1 = pd.read_csv("PAM50_labels-RNA-Seq_1148.csv",sep=';')
seq2 = pd.read_csv("PAM50_labels-RNA-Seq_737.csv",sep=';')
seq3 = pd.read_csv("PAM50_labels-RNA-Seq_534.csv",sep=';')#.pop("LumA-R1/2") #FIXME
seq4 = pd.read_csv("PAM50_labels-Meth450_679.csv",sep=';')
seq5 = pd.read_csv("PAM50_labels-Meth450_513.csv",sep=';')
seq6 = pd.read_csv("PAM50_labels-Meth450_378.csv",sep=';')#.pop("LumA-M1/2/3") #FIXME

print(seq6.dtypes)

def add(f, seq):
	for index, row in seq.iterrows():
		if "Normal" in str(row["Age (Median)"]):
			continue
		if "nan" in str(row["ER Status"]):
			continue
		if "nan" in str(row["PR Status"]):
			continue
		if "nan" in str(row["Her2 Status"]):
			continue
		if "nan" in str(row["Pathologic stage"]):
			continue
		if "nan" in str(row["Histological type"]):
			continue

		who = seq_data.loc[seq_data["sampleID"] == row["Sample ID"]]
		if "FEMALE" != who["gender"].item():
			continue
		if "Metastatic" == who["sample_type"].item():
			continue
		if "Breast" != who["tumor_tissue_site"].item():
			continue

		#print(who)
		#print(who.values)
		#input("dance break")

		f.write(";".join([str(row[i]) for i in types]) + "\n")

def gen():
	with open("PAM50_filtered.csv", "w") as f:
		f.write("Sample ID;Cluster ID;Age (Median);ER Status;PR Status;Her2 Status;PAM50;Pathologic stage;Histological type\n")
		add(f,seq1)
		add(f,seq2)
		add(f,seq3)
		add(f,seq4)
		add(f,seq5)
		add(f,seq6)

gen()