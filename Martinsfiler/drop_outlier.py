#drop outlier
from operator import index
import pandas as pd
import numpy as np
from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("matricen05.csv")
#ny_df = df.drop(['TCGA-A7-A56D-01', 'TCGA-AC-A2QJ-01'])
#ny_df = df.drop('TCGA-A7-A56D-01', axis=0)
ny_df = df.drop(df.index[[269]])
ny_df.to_csv("matricen05-out.csv", index=False)