import os
import logging
import pandas as pd
import pytest  
from churn_library import import_data, perform_eda
from constants import PATH

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture
def perform_eda():
    """Fixture returns the perform_eda function."""
    from churn_library import perform_eda as _perform_eda
    return _perform_eda

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data(PATH)
		logging.info("TEST import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("TEST import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("TEST import_data: The file doesn't appear to have rows and columns")
		raise err

def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df = import_data(PATH)
        perform_eda(df)
        logging.info("TEST perform_eda: SUCCESS")
    except Exception as err:
        logging.error("TEST perform_eda: An error occurred")
        raise err



# def test_encoder_helper(encoder_helper):
# 	'''
# 	test encoder helper
# 	'''


# def test_perform_feature_engineering(perform_feature_engineering):
# 	'''
# 	test perform_feature_engineering
# 	'''


# def test_train_models(train_models):
# 	'''
# 	test train_models
# 	'''


if __name__ == "__main__":
	test_import()
	test_eda(perform_eda)







