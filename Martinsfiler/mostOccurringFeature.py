from numpy import bincount
from numpy.core.fromnumeric import sort
import seaborn as sns

#list of 40 most important features, concatinated over 5 different runs.

features5run = ['GP2', 'ABCC11', 'CCNO', 'CYP4F8', 'CYP4Z2P', 'HSD3B2', 'PGLYRP4', 'CXCL5', 'SERPINB5', 'CEACAM5', 'BPI', 'CYP4B1', 'OLFM4', 'KRT6B', 'SERHL2', 'GJB3', 'AR', 'HORMAD1', 'FGFBP1', 'PNMT', 'LOC145837', 'KRT5', 'GDF5', 'HEPHL1', 'ABCC12', 'ERBB2', 'SFRP1', 'FOLR1', 'MAT1A', 'GATA5', 'FOXI1', 'OLAH', 'SLC9A2', 'MUCL1', 'BCAS1', 'C2orf54', 'SOSTDC1', 'AQP5', 'FOXC1', 'GSTA1','CEACAM5', 'CYP4Z2P', 'LOC643719', 'PNMT', 'CCNO', 'CYP4F8', 'TFAP2B', 'GP2', 'AMY1A', 'MAT1A', 'SDR16C5', 'FOLR1', 'SFRP1', 'SERHL2', 'C2orf54', 'AKR7A3', 'DHRS2', 'HSD3B2', 'PRSS21', 'ABCC11', 'BCAS1', 'CLEC3A', 'CYP4Z1', 'PADI3', 'ERBB2', 'ALOX15B', 'CYP4B1', 'FBN3', 'FGFBP1', 'SAA2', 'HS6ST3', 'FOXI1', 'ABCC12', 'SLC6A14', 'FOXC1', 'UGT2B11', 'BPI', 'TCAP', 'CXCL5', 'MUCL1','CEACAM5', 'GP2', 'MUCL1', 'ABCC11', 'TFF1', 'CYP4Z2P', 'PADI3', 'BCAS1', 'PNMT', 'SFRP1', 'AQP5', 'AR', 'C20orf114', 'CAPN13', 'SCGB3A1', 'HS6ST3', 'CYP4Z1', 'CPA4', 'ERBB2', 'GDF5', 'LOC145837', 'SDR16C5', 'FOXC1', 'C2orf54', 'ROPN1B', 'CCNO', 'PRSS21', 'SOSTDC1', 'FGB', 'GSTA1', 'CAPN6', 'ROPN1', 'ABCA12', 'IL22RA2', 'CXCL5', 'SERHL2', 'MYO3B', 'ALOX15B', 'CYP4B1', 'LRRC31','GP2', 'ABCC11', 'AQP5', 'OLFM4', 'DMBT1', 'ZIC1', 'GDF5', 'MUCL1', 'CEACAM5', 'CYP4B1', 'SCGB3A1', 'S100A2', 'HORMAD1', 'SDR16C5', 'FOXC1', 'NXPH1', 'LOC145837', 'CXCL5', 'CYP4F8', 'ROPN1B', 'ROPN1', 'MSLN', 'ABCC12', 'AR', 'TCAM1P', 'SCGB2A2', 'ERBB2', 'LEMD1', 'PTPRZ1', 'PNMT', 'LRRC31', 'HS6ST3', 'CHI3L2', 'HMGCS2', 'C4BPA', 'DMRTA2', 'CCNO', 'BMP5', 'TFF3', 'DHRS2','GP2', 'ABCC11', 'CEACAM5', 'FGB', 'MUCL1', 'SDR16C5', 'HMGCS2', 'AQP5', 'PRSS21', 'DHRS2', 'TCAM1P', 'OLFM4', 'CYP4Z2P', 'LOC145837', 'ABCC12', 'TFAP2B', 'CYP4F8', 'SLC44A5', 'CAPN13', 'KRT5', 'FGFBP1', 'LRP2', 'FOXC1', 'CCNO', 'PGC', 'EPO', 'FGG', 'ROPN1', 'SAA2', 'AMY1A', 'GDF5', 'SCGB3A1', 'PAX7', 'KCNS1', 'ERBB2', 'HBA1', 'ROPN1B', 'SAA1', 'CPB1', 'TFF3']


unique = []
double = []
final = []
for number in features5run:
    if number in unique:
        double.append(number)
    else:
        unique.append(number)
    if number in double:
        final.append(number)

""" unique = []
for number in final:
    if number in unique:
        continue
    else:
        unique.append(number) """
        
print(unique)
print(final)
finish = final + unique
print(sort(finish))

sns_plot = sns.histplot(data=finish, shrink=.8)
sns_plot.set_xticklabels(labels= finish, rotation=90, fontsize=7)
sns_plot.set_title('Extracting most important features - SVM \n 5 runs - treshhold 2 occurences \n PAM 51 ! ', fontsize=10)
sns_plot.figure.savefig('features5run-2.png', bbox_inches='tight', dpi=500 )

