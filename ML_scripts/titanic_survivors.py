import pandas as pd
import numpy as np
import seaborn as sns
from zipfile import ZipFile
import os

# data download & extraction
path_download = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'helpers', 'competition_downloader.py')
path_zip = "../competitions/titanic.zip"
path_out = "../competitions/titanic"

if not os.path.exists(path_zip):
    os.system(f'python \"{path_download}\" titanic')
    while not os.path.exists(path_zip):
        pass

if not os.path.exists(path_out):
    with ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_out)

all_files = os.listdir(path_out)
print(all_files)

# data import

y_test = pd.read_csv(os.path.join(path_out,all_files[0]), index_col='PassengerId')
X_test = pd.read_csv(os.path.join(path_out,all_files[1]), index_col='PassengerId')
X_train = pd.read_csv(os.path.join(path_out,all_files[2]), index_col='PassengerId')
y_train = pd.read_csv(os.path.join(path_out,all_files[2]), index_col='PassengerId', usecols=['PassengerId','Survived'])

X_train.drop(columns='Survived', inplace=True)

print(X_train.head())
print(y_train.head())

# data analysis

