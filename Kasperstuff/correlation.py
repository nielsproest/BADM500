import numpy as np
import pandas as pd

#Load data
# KØR dimensionality_reduction.py først for at få master123
print("Load data")
dt = pd.read_csv("master1234.csv")
#print(dt.head())


print(pd.DataFrame(np.sort(dt.corr())[:,::-1]))
