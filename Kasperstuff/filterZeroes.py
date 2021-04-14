import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#Load data
print("Load data")
df = pd.read_csv("../HiSeq/master.csv",sep=';')

zerosarr = []

# Adding the amount of zeroes from each gene into a list
# Takes a while to run. I think df.drop is slow 
for column in df.columns[1:]:
    y = df[column].to_numpy()
    num_zeros = (y == 0).sum()
    if(num_zeros > 1066):                   # Drop from dataframe if # of zeroes is over the threshold (1144 - 78 = 1066)
        zerosarr.append(num_zeros)
        df.drop(column, axis=1, inplace=True)

print(len(df.columns))
print(df)

df.to_csv("filterZeroes.csv", index=False, sep=";")