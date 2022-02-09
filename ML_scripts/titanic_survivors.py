import pandas as pd
import numpy as np
import seaborn as sns

# data import
path = "../competitions/titanic.zip"
df = pd.read_csv(path, index_col='PassengerId')

# print(df.head())
# print(df.describe())
print(df.info())

# data analysis
