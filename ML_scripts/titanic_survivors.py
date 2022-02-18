import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
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
sns.set_theme(style= "whitegrid")


# data download & extraction
## if data is missing, it will be downloaded automatically using Kaggle API
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
logger.info(f"Imported files: {all_files}")


# data import
# Initial columns of X ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
y_test = pd.read_csv(os.path.join(path_out,all_files[0]), index_col='PassengerId',)
X_test = pd.read_csv(os.path.join(path_out,all_files[1]), index_col='PassengerId')
X_train = pd.read_csv(os.path.join(path_out,all_files[2]), index_col='PassengerId')
y_train = pd.read_csv(os.path.join(path_out,all_files[2]), index_col='PassengerId', usecols=['PassengerId','Survived'])

## check for data
# print(X_train.describe())
# print(X_train.info())
# print(X_train.head())

## creating categorized column based on strings
le_emb = LabelEncoder()
le_emb.fit(X_train['Embarked'])
X_train['Embarked_labels']= le_emb.transform(X_train['Embarked'])
X_test['Embarked_labels']= le_emb.transform(X_test['Embarked'])
logger.info(f"'Embarked' column encoded by labels: {le_emb.classes_}")

le_sex = LabelEncoder()
le_sex.fit(X_train['Sex'])
X_train['Sex_labels']= le_sex.transform(X_train['Sex'])
X_test['Sex_labels']= le_sex.transform(X_test['Sex'])
logger.info(f"'Sex' column encoded by labels: {le_sex.classes_}")

## drop unnecessary columns
X_train.drop(['Survived'], axis=1, inplace=True)
X_train.drop(['Embarked'], axis=1, inplace=True)
X_test.drop(['Embarked'], axis=1, inplace=True)
X_train.drop(['Name'], axis=1, inplace=True)
X_test.drop(['Name'], axis=1, inplace=True)
X_train.drop(['Ticket'], axis=1, inplace=True)
X_test.drop(['Ticket'], axis=1, inplace=True)
X_train.drop(['Cabin'], axis=1, inplace=True)
X_test.drop(['Cabin'], axis=1, inplace=True)

# print(X_train.head(10))


# data analysis

## general overview
# sns.pairplot(data= X_train.drop('Sex_labels', axis=1), hue= 'Sex')
# plt.figure(1)

# plt.figure(2)
# sns.heatmap(data= X_train.corr(), annot=True)
## heatmap have shown that Age is mostly correlated with Pclass

## updating missing Age values based on the Passenger's ticket class
fill_table = {}
for Pclass in X_train['Pclass'].unique():
    mean_age = int(X_train[X_train['Pclass']==Pclass]['Age'].median())
    fill_table.update({Pclass: mean_age})

## for loop output:
#  [(3, 24), (1, 37), (2, 29)]

def fill_age(cols, fill):
    """
    :param cols: columns to be retrieved - Pclass, Age
    :return: current age or median value if the age is NaN
    """
    if pd.isnull(cols[1]):
        return fill[cols[0]]
    else:
        return cols[1]

X_train['Age']=X_train[['Pclass', 'Age']].apply(fill_age, axis= 1, fill= fill_table)
X_test['Age']=X_test[['Pclass', 'Age']].apply(fill_age, axis= 1, fill= fill_table)

## fill all remaining NaN's
X_train['Fare'].fillna(X_train['Fare'].median(), inplace= True)
X_test['Fare'].fillna(X_test['Fare'].median(), inplace= True)

## confirm that all values are set
# print(X_train.describe())
# print(X_test.describe())

plt.figure(3)
sns.violinplot(data= X_train, hue= 'Sex', x= 'Pclass', y='Age', split=True)

## combination of these two lines causes script to draw graph without stopping whole script
plt.draw()
plt.pause(0.001)

# creating model
## finding the best estimator with using Grid Search
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train.drop('Sex', axis=1))
X_test_scaled = scale.fit_transform(X_test.drop('Sex', axis=1))
param_grid = dict(
    C=[0.01, 0.1, 1, 10, 100],
    kernel = ['linear', 'poly', 'rbf', 'sigmoid'],
    gamma=['scale', 'auto']
)

gscv = GridSearchCV(SVC(), param_grid=param_grid, n_jobs=-1, verbose=3)
gscv.fit(X_train_scaled,y_train['Survived'] )

## Saving the best estimator
clf = gscv.best_estimator_

y_pred =clf.predict(X_test_scaled)

tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
logger.info(f"\nBest selected estimator for this dataset is:\n\t\t{clf}\n")
logger.info("Confusion matrix:"
            f"\n\tTrue positives: {tp}"
            f"\n\tTrue negatives: {tn}"
            f"\n\tFalse positives: {fp}"
            f"\n\tFalse negatives: {fn}")
logger.info(classification_report(y_test,y_pred))

## show() prevents graph from closing at the end of script
## close graph(s) in order to exit
plt.show()
