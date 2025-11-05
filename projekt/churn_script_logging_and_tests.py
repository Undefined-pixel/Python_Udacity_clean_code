"""
test function in churm library.
Date: 2025-11-04
"""
import os
import warnings
import logging
import pytest

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from constants import PATH
from churn_library import (
    import_data,
    perform_eda,
    encoder_helper,
    perform_feature_engineering,
    train_models,
    classification_report_image
)


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


@pytest.fixture
def train_models():
    """Fixture returns the perform feature engineering function."""
    from churn_library import train_models as _train_models

    return _train_models


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


def test_eda(perform_eda_data):
    """
    test perform eda function
    """
    try:
        df = import_data(PATH)
        perform_eda_data(df)
        logging.info("TEST perform_eda: SUCCESS")
    except Exception as err:
        logging.error("TEST perform_eda: An error occurred")
        raise err


def test_encoder_helper(encoder_helper_data):
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
        df = encoder_helper_data(df, category_lst, "Churn")
        assert "Gender_Churn" in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: An error occurred")
        raise err


def test_perform_feature_engineering(
        perform_feature_engineering,
        encoder_helper):
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
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, "Churn")
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        assert x_train.shape[1] == x_test.shape[1]
        logging.info("TEST perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error("TEST perform_feature_engineering: An error occurred")
        raise err


def test_classification_report_image(tmp_path):
    """
    test classification_report_image
    """

    x, y = make_classification(n_samples=200, n_features=5, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier()
    lr.fit(x_train, y_train)
    rf.fit(x_train, y_train)

    y_train_preds_lr = lr.predict(x_train)
    y_test_preds_lr = lr.predict(x_test)
    y_train_preds_rf = rf.predict(x_train)
    y_test_preds_rf = rf.predict(x_test)

    os.makedirs("images", exist_ok=True)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    assert os.path.exists("images/classification_report.png")
    logging.info("TEST classification_report_image: SUCCESS")


def test_train_models(
        train_models_data,
        encoder_helper_data,
        perform_feature_engineering_data):
    """
    test train_models
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    try:
        df = import_data(PATH)
        category_lst = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]
        df = encoder_helper_data(df, category_lst, "Churn")
        x_train, x_test, y_train, y_test = perform_feature_engineering_data(
            df, "Churn")
        train_models_data(x_train, x_test, y_train, y_test)

        # Check if model files and images are created
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")
        assert os.path.exists("images/feature_importance.png")
        assert os.path.exists("images/ROC_Curve.png")
        assert os.path.exists("images/classification_report.png")
        print("TEST train_models: SUCCESS")
    except Exception as err:
        print("TEST train_models: FAILED")
        raise err

if __name__ == "__main__":
    test_import()
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(
    perform_feature_engineering, encoder_helper)
    train_models(train_models, encoder_helper, perform_feature_engineering)
