import pandas as pd
import numpy as np

dt = pd.read_csv("BRCA_clinicalMatrix",sep='\t')

types = dt.columns.values

f = open("BRCA_clinicalMatrix_filtered.csv", "w")
f2 = open("BRCA_clinicalMatrix_removed.csv", "w")
f.write(";".join(types) + ";\n")
f2.write(";".join(types) + ";\n")

def lul(thing):
	what = str(thing)
	if (what == "nan"):
		return ""
	if (what.endswith(".0")):
		what = what[:-2]

	return what

seen = []
for index, row in dt.iterrows():
	failed = False

	if str(row["gender"]) == "MALE":
		failed = True
	if str(row["Gender_nature2012"]) == "MALE":
		failed = True
	if str(row["sampleID"]) in seen:
		failed = True
	if str(row["sample_type"]) == "YES" or str(row["sample_type"]) == "NO":
		failed = True
	if str(row["tumor_tissue_site"]) != "Breast":
		failed = True

	if (failed):
		f2.write(";".join([lul(row[i]) for i in types]) + "\n")
		continue

	f.write(";".join([lul(row[i]) for i in types]) + "\n")
	seen.append(row["sampleID"])

f.close()