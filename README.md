# Kaggle Datasets Playground

This project is dedicated to educational purposes.
It contains files that are used for different machine learning algorithms,
especially datasets and scripts that operates on them.
 
__In order to manipulate on Kaggle datasets from this application:__ 
1. [provide Kaggle API token](https://www.kaggle.com/docs/api#authentication) 
and place`kaggle.json` in root folder
2. Run in terminal `pip install -r requirements.txt`

## How to search for datasets/competitions
1. Provide searching parameters inside `.\helpers\kaggle_explorer.py` script
2. Run script
3. Look for results in the terminal

## How to download dataset
1. Obtain unique identifier name of dataset (contained in _ref_ column of searching output)
2. As default, `.\helpers\dataset_downloader.py` script downloads sample dataset to `.\datasets` folder unzipped
3. Parameter modification is possible in two ways:
   1. Changing parameters inside script
   2. Invoking script with syntax: `python .\helpers\dataset_downloader.py` `dataset_name` `download_location` `unzip?`
   All parameters are optional. When missing, default values are obtained from script

## How to download competition
1. Obtain unique identifier name of the competition (i.e. from competition web page)
2. As default, `.\helpers\competition_downloader.py` script downloads competition files to `.\competitions` folder
3. Parameter modification is possible in two ways:
   1. Changing parameters inside script
   2. Invoking script with syntax: `python .\helpers\competition_downloader.py` `competition_name` `download_location`
   All parameters are optional. When missing, default values are obtained from script
   
## All ML algorithms can be invoked directly from `ML_scripts` folder


