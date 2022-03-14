import os
import shutil
import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
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

# logger.info(df.head())
# logger.info(df.describe())
# logger.info(df.info())

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
plt.title("Reduced Pair Plot")

# data preparation
X = df.drop(columns= ['Class', 'Class_labels'])
y = df['Class_labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Classifier
rf_clf = RandomForestClassifier( n_jobs=-1)# max_depth=3, min_samples_leaf=5,
rf_clf.fit(X_train,y_train)
y_rf_proba = [y[1] for y in rf_clf.predict_proba(X_test)]
y_rf_deviation = abs(y_test-y_rf_proba)
y_rf_pred = rf_clf.predict(X_test)

# Logistic Regression
lr_clf = LogisticRegression(solver='liblinear', n_jobs= -1 )
lr_clf.fit(X_train_scaled,y_train)
y_lr_proba= [y[1] for y in lr_clf.predict_proba(X_test_scaled)]
y_lr_deviation = abs(y_test-y_lr_proba)
y_lr_pred = lr_clf.predict(X_test_scaled)

# Grid search on loss functions for Stochastic Gradient Descent CLF
params ={'loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron')}
grid_clf = GridSearchCV(SGDClassifier(), param_grid=params, n_jobs=-1, verbose=3)
grid_clf.fit(X_train_scaled, y_train)
y_grid_pred = grid_clf.predict(X_test_scaled)

#result analysis

# this plot shows the deviation between real and predicted values
# if value on y axis is bigger than 0.5, it means wrong prediction
fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('Deviation representation')
ax1.scatter(x= y_test.index, y=y_rf_deviation, marker= '.', c= 'blue', label= "Random Forest")
ax1.axhline(y=0.5, label= 'Confusion Threshold')
ax2.scatter(x= y_test.index, y=y_lr_deviation, marker= '.', c= 'red', label= "Logistic Regression")
ax2.axhline(y =0.5)
fig.legend(loc = 'upper right')


# This figure presents feature importances
# higher value means bigger impact on final calculation
plt.figure()
sns.barplot(y=rf_clf.feature_importances_,x = X.columns)
plt.title("Feature importances")
# It is visible that 'Roundness' column has the biggest importance as long as it shows the biggest separation
# among all features - can be noticed also on the pairplot.

logger.info("Random forest predictions:")
logger.info(confusion_matrix(y_pred=y_rf_pred, y_true=y_test))
logger.info(classification_report(y_pred=y_rf_pred, y_true=y_test)+"\n\n")

logger.info("Logistic regression predictions:")
logger.info(confusion_matrix(y_pred=y_lr_pred, y_true=y_test))
logger.info(classification_report(y_pred=y_lr_pred, y_true=y_test)+"\n\n")

logger.info("Stochastic Gradient Descent predictions:")
logger.info(confusion_matrix(y_pred=y_grid_pred, y_true=y_test))
logger.info(classification_report(y_pred=y_grid_pred, y_true=y_test))
plt.show()
