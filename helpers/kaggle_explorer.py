import os

from kaggle_credential import setup_kaggle_credentials

# kaggle searching setup - update used fields by inserting values
# (defaults noted in comment)

# Common parameters
data_type = "datasets"  # available also "competitions"
help = None  # True
sort_by = None  # 'latestDeadline' / 'hottest'
search = 'Pumpkin'
page = None  # '20'
csv = None  # True

""" COMPETITION EXPLORER

  -h, --help            show this help message and exit
  --group GROUP         Search for competitions in a specific group. Default is 'general'. Valid options are 'general', 'entered', and 'inClass'
  --category CATEGORY   Search for competitions of a specific category. Default is 'all'. Valid options are 'all', 'featured', 'research', 'recruitment', 'gettingStarted', 'masters', and 'playground'
  --sort-by SORT_BY     Sort list results. Default is 'latestDeadline'. Valid options are 'grouped', 'prize', 'earliestDeadline', 'latestDeadline', 'numberOfTeams', and 'recentlyCreated'
  -p PAGE, --page PAGE  Page number for results paging. Page size is 20 by default 
  -s SEARCH, --search SEARCH
                        Term(s) to search for
  -v, --csv             Print results in CSV format
                        (if not set print in table format)
"""  # COMPETITION EXPLORER
group = None  # 'general'
category = None  # 'all'

"""DATASET EXPLORER

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
  """  # DATASET EXPLORER
size = None  # 'all'
file_type = None  # 'all'
specific_license = None  # 'all'
tags = None  # ''
mine = None  # True
user = None  # 'username'

command = f'kaggle {data_type} list'
if help is not None:
    command += ' -h'
if sort_by is not None:
    command += f' --sort-by {sort_by}'
if search is not None:
    command += f" -s {search.replace(' ', '-')}"
if page is not None:
    command += f' -p {page}'
if csv is not None:
    command += f' -v'

# optional parameters
if size is not None:
    command += f' --size {size}'
if file_type is not None:
    command += f' --file-type {file_type}'
if specific_license is not None:
    command += f' --license {specific_license}'
if tags is not None:
    command += f' --tags {tags}'
if mine is not None:
    command += f' -m'
if user is not None:
    command += f' --user {user}'

setup_kaggle_credentials()
# check terminal for results
os.system(command)
