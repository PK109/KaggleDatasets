import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import shutil

from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

# pandas setup for output display
pd.options.display.width = 0
pd.set_option('display.max_columns', 500)

# Seaborn setup
sns.set_theme(style="whitegrid")

# data download & extraction
# if data is missing, it will be downloaded automatically using Kaggle API
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_download = os.path.join(base_path, 'helpers', 'dataset_downloader.py')
path_out = os.path.join(base_path, 'datasets', 'pumpkin')

if not os.path.exists(path_out):
    os.system(f'python \"{path_download}\" mkoklu42/pumpkin-seeds-dataset \"{path_out}\"')



if os.path.exists(os.path.join(path_out, 'output')):
    shutil.rmtree(os.path.join(path_out, 'output'))

all_files = os.listdir(path_out)
logger.info(f"Imported files: {all_files}")

# data import
df = pd.read_excel(io= os.path.join(path_out, all_files[-1]), sheet_name=0)
# X.drop(['Class'], inplace= True, axis= 1)

logger.info(df.head())
logger.info(df.describe())
logger.info(df.info())

lb = LabelBinarizer()
df['Class_labels'] = pd.DataFrame(lb.fit_transform(df['Class']))

# data analysis

# general overview - it takes much time to draw that plot
# sns.pairplot(df.drop(columns=['Class_labels']) , hue= 'Class')

# checking correlation between features
# sns.heatmap(df.corr(), annot= True)

#due to high correlation of several features, some of them needs to be dropped.
df.drop(columns= ['Perimeter', 'Major_Axis_Length', 'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Aspect_Ration', 'Compactness'], inplace=True)

# simplified pairplot
sns.pairplot(df.drop(columns=['Class_labels']) , hue= 'Class')

plt.show()
