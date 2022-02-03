import os
import json
import logging
import pdb
from io import StringIO

#kaggle searching setup - update used fields by inserting values
#(defaults noted in comment)
"""
  -h, --help            show this help message and exit
  --sort-by SORT_BY     Sort list results. Default is 'hottest'. Valid options are 'hottest', 'votes', 'updated', and 'active'
  --size SIZE           Search for datasets of a specific size. Default is 'all'. Valid options are 'all', 'small', 'medium', and 'large'
  --file-type FILE_TYPE Search for datasets with a specific file type. Default is 'all'. Valid options are 'all', 'csv', 'sqlite', 'json', and 'bigQuery'. Please note that bigQuery datasets cannot be downloaded
  --license LICENSE_NAME 
                        Search for datasets with a specific license. Default is 'all'. Valid options are 'all', 'cc', 'gpl', 'odb', and 'other'
  --tags TAG_IDS        Search for datasets that have specific tags. Tag list should be comma separated                      
  -s SEARCH, --search SEARCH
                        Term(s) to search for
  -m, --mine            Display only my items
  --user USER           Find public datasets owned by a specific user or organization
  -p PAGE, --page PAGE  Page number for results paging. Page size is 20 by default
  -v, --csv  
  """

help = None # True
sort_by = 'votes'
size = None # 'all'
file_type = None # 'all'
specific_license = None # 'all'
tags = None # ''
search = 'titanic'
mine = None # True
user = None # 'username'
page = None # '20'
csv  =  None #True

#logger setup
logging.basicConfig(level=logging.DEBUG )
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

#variable setup for Kaggle API
os.environ['KAGGLE_CONFIG_DIR']= os.path.abspath(os.pardir)
logger.info(f"API token should be located in the folder: {os.environ['KAGGLE_CONFIG_DIR']}")

kaggle_cred = {}
with open(os.path.join(os.getenv('KAGGLE_CONFIG_DIR'),'kaggle.json')) as file:
    json_file = file.readline()
    kaggle_cred.update(json.loads(json_file))
    logger.info(f'Successfully imported API credentials for user: \'{kaggle_cred["username"]}\'')

command = 'kaggle datasets list'
if help: command += ' -h'
if sort_by: command += f' --sort-by {sort_by}'
if size: command += f' --size {size}'
if file_type : command += f' --file-type {file_type}'
if specific_license: command += f' --license {specific_license}'
if tags: command += f' --tags {tags}'
if search: command += f' -s {search}'
if mine: command += f' -m'
if user: command += f' --user {user}'
if page: command += f' -p {page}'
if csv: command += f' -v'

logger.debug(command)

#check terminal for results
os.system(command)
