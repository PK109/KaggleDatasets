import os
import sys
import json
import logging
from kaggle_credential import setup_kaggle_credentials

# downloading path
default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'competitions')

#sample data to be downloaded
predefined_values = dict(
    COMPETITION = "titanic",
    PATH =        default_path,
)

command = 'kaggle competitions download'

try:
    competition = sys.argv[1]
except:
    competition = predefined_values['COMPETITION']

try:
    path = sys.argv[2]
except:
    path = predefined_values['PATH']

command += f" {competition}"
command += f" -p \"{path}\""


setup_kaggle_credentials()
print("Command:\n" + command)
os.system(command)