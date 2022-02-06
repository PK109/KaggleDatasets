import logging
import os
import json

def setup_kaggle_credentials(location = None):
    #logger setup
    logging.basicConfig(level=logging.DEBUG )
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    #variable setup for Kaggle API
    if location is None:
        os.environ['KAGGLE_CONFIG_DIR']= os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    else:
        os.environ['KAGGLE_CONFIG_DIR']= os.path.abspath(location)

    logger.info(f"API token should be located in the folder: {os.environ['KAGGLE_CONFIG_DIR']}")

    kaggle_cred = {}
    with open(os.path.join(os.getenv('KAGGLE_CONFIG_DIR'),'kaggle.json')) as file:
        json_file = file.readline()
        kaggle_cred.update(json.loads(json_file))
        logger.info(f'Successfully imported API credentials for user: \'{kaggle_cred["username"]}\'')
