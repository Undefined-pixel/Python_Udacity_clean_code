import os
import logging
import pandas as pd
import pytest
from churn_library import (
    import_data,
    perform_eda,
    encoder_helper,
    perform_feature_engineering,
)
from constants import PATH

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


@pytest.fixture
def perform_eda():
    """Fixture returns the perform_eda function."""
    from churn_library import perform_eda as _perform_eda

    return _perform_eda


@pytest.fixture
def encoder_helper():
    """Fixture returns the encoder_helper function."""
    from churn_library import encoder_helper as _encoder_helper

    return _encoder_helper


@pytest.fixture
def perform_feature_engineering():
    """Fixture returns the perform feature engineering function."""
    from churn_library import (
        perform_feature_engineering as _perform_feature_engineering,
    )

    return _perform_feature_engineering


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
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
        logging.error(
            "TEST import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_eda(perform_eda):
    """
    test perform eda function
    """
    try:
        df = import_data(PATH)
        perform_eda(df)
        logging.info("TEST perform_eda: SUCCESS")
    except Exception as err:
        logging.error("TEST perform_eda: An error occurred")
        raise err


def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """
    try:
        df = import_data(PATH)
        category_lst = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]
        df = encoder_helper(df, category_lst, "Churn")
        assert "Gender_Churn" in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: An error occurred")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, encoder_helper):
    """
    test perform_feature_engineering
    """
    try:
        df = import_data(PATH)
        category_lst = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]
        df = encoder_helper(df, category_lst, "Churn")
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        assert X_train.shape[1] == X_test.shape[1]
        logging.info("TEST perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error("TEST perform_feature_engineering: An error occurred")
        raise err


# def test_train_models(train_models):
# 	'''
# 	test train_models
# 	'''


if __name__ == "__main__":
    test_import()
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering, encoder_helper)

