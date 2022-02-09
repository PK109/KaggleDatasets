import os
import sys
import json
import logging
from kaggle_credential import setup_kaggle_credentials

default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'datasets')

#sample data to be downloaded
predefined_values = dict(
    DATASET = "johnharshith/world-happiness-report-2021-worldwide-mortality",
    PATH =    default_path,
    UNZIP =   True
)

command = 'kaggle datasets download'

try:
    dataset = sys.argv[1]
except:
    dataset = predefined_values['DATASET']

try:
    path = sys.argv[2]
except:
    path = predefined_values['PATH']

try:
    unzip = sys.argv[3].lower() == 'true'
except:
    unzip = predefined_values['UNZIP']

command += f" {dataset}"
command += f" -p \"{path}\""
if unzip:
    command += " --unzip"


setup_kaggle_credentials()
os.system(command)