import os
import sys
import json
import logging
from kaggle_credential import setup_kaggle_credentials

default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'datasets')
setup_kaggle_credentials()

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
    unzip = sys.argv[3] == 'true'
except:
    unzip = predefined_values['PATH']

command += f" {dataset}"
command += f" -p \"{path}\""
if unzip:
    command += " --unzip"

print(command)
os.system(command)