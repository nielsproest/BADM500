
#from gradientboost max depth 5
ourmap = [['DEGS2', 0.1264972], ['ADAMTS5', 0.10701432], ['TFF3', 0.04287165], ['IL34', 0.025782468], ['TIMELESS', 0.024747847], ['MPHOSPH6', 0.020023478], ['LOC145837', 0.017804982], ['SPRY2', 0.01736538], ['GATA3', 0.016528653], ['PTGER3', 0.0155141335], ['GSG2', 0.013249909], ['HOXC13', 0.011829336], ['FABP7', 0.009907588], ['NKAIN1', 0.009564138], ['KRT5', 0.009445746], ['MCM10', 0.0086531695], ['PRSS33', 0.008623586], ['TFCP2L1', 0.008525479], ['RDH16', 0.008360593], ['RUNX1', 0.007924087], ['CCDC8', 0.00789772], ['CEP55', 0.007315912], ['PPAP2B', 0.007138676], ['PRPF3', 0.0071124053], ['USP36', 0.00635462], ['DLX6', 0.006306232], ['TGFBR3', 0.0062688114], ['AACSL', 0.006155054], ['KSR2', 0.0058592097], ['DEPDC1B', 0.0057979217], ['CDH3', 0.0057302047], ['C6orf211', 0.0056741354], ['FANCA', 0.005529149], ['F10', 0.0054974896], ['DBNDD2', 0.005380933], ['WDR72', 0.00532611], ['UBE2T', 0.0042625405], ['LPO', 0.0041979933], ['PGAP3', 0.0041516013], ['TRIM2', 0.0041230093], ['PXDNL', 0.0041004797], ['NCAPG', 0.00405359], ['MLPH', 0.004023359], ['LEPREL1', 0.0039622267], ['CA12', 0.003720466], ['FGD3', 0.003665869], ['F2R', 0.003645767], ['RAB9A', 0.0036168785], ['SUSD2', 0.0036067679], ['TMEM108', 0.0035040022]]
ourmap = [k[0] for k in ourmap]

pam50_map = ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1', 'CDC20', 'CDC6', 'CDCA1', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2', 'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C', 'KNTC2', 'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67', 'MLPH', 'MMP11', 'MYBL2', 'MYC', 'NAT1', 'ORC6L', 'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B', 'TYMS', 'UBE2C', 'UBE2T']

import pandas as pd
import numpy as np
if False:
	print("Load data")
	dt = pd.read_csv("pearson.csv",sep=";")

	print("Scanning")
	arr = dt.columns.values
	columns_array = [i for i in arr if not i in ourmap]
	row_array = [i for i in arr if not i in pam50_map]

	print("Working")
	new_dt = dt
	new_dt.index = new_dt[new_dt.columns.values[0]] #Fix row labels
	new_dt = new_dt.drop("Unnamed: 0", axis=1) #Fix row labels
	new_dt = new_dt.drop(columns_array, axis=1, errors="ignore")
	new_dt = new_dt.drop(row_array, axis=0, errors="ignore")

	print("Save")
	new_dt.to_csv("pearson_heatmap.csv", sep=";")

import seaborn as sns
import matplotlib.pyplot as plt
if True:
	dt = pd.read_csv("pearson_heatmap.csv", sep=";")
	#plt.figure(figsize=(6,6))
	dt.index = dt[dt.columns.values[0]]
	dt = dt.drop("Unnamed: 0", axis=1)

	dt = dt[sorted(dt)]
	dt = dt.sort_index(ascending=False)
	dt = dt.abs()
	for i in dt.columns.values:
		idx = dt[i].idxmax()
		print(i, idx, dt[i][idx])


	print(dt)
	ax = sns.heatmap(dt, annot=False, vmin=0, vmax=1, xticklabels=True, yticklabels=True)

	plt.xlabel("from_gb")
	plt.ylabel("pam50")
	plt.show()
	#plt.savefig("heatmap.png",dpi=600)
