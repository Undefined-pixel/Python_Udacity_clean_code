# import libraries
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from constants import PATH

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def print_df(df):
    print(df.head())


def import_data(pth: str):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    return df


def perform_eda(df):
    """
    perform eda (Exploratory Data Analysis) on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    os.makedirs("images/eda", exist_ok=True)

    # Histogram of Churn Distribution
    plt.figure(figsize=(20, 10))
    plt.title("Churn_Distribution")
    df["Churn"].hist()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("images/eda/churn_distribution.png")
    plt.close()

    # Histogram of Customer Age
    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.title("Distribution of Customer Age")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.savefig("images/eda/customer_age_distribution.png")
    plt.close()

    # Bar plot of Marital Status
    plt.figure(figsize=(20, 10))
    df["Marital_Status"].value_counts(normalize=True).plot(kind="bar")
    plt.title("Marital Status Distribution")
    plt.xlabel("Marital Status")
    plt.ylabel("Proportion")
    plt.savefig("images/eda/marital_status_distribution.png")
    plt.close()

    # Bar plot of Normalized Marital Status
    plt.figure(figsize=(20, 10))
    df["Marital_Status"].value_counts(normalize=True).plot(kind="bar")
    plt.title("Normalized Marital Status Distribution")
    plt.xlabel("Marital Status")
    plt.ylabel("Proportion")
    plt.savefig("images/eda/normalized_marital_status_distribution.png")
    plt.close()

    # Density plot of Total_Trans_Ct
    plt.figure(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.title("Density Plot of Total Transaction Count")
    plt.xlabel("Total Transaction Count")
    plt.ylabel("Density")
    plt.savefig("images/eda/total_transaction_count_density.png")
    plt.close()

    # Heatmap of Correlation Matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df.select_dtypes(include=["number"]).corr(),
        annot=False,
        cmap="Dark2_r",
        linewidths=2,
    )
    plt.title("Correlation Matrix")
    plt.savefig("images/eda/correlation_matrix.png")
    plt.close()


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    if not pd.api.types.is_numeric_dtype(df[response]):
        raise ValueError(
            f"Die Zielvariable '{response}' muss numerisch sein (z. B. 0/1)."
        )

    for column in category_lst:
        if column not in df.columns:
            print(f"⚠️ Spalte '{column}' nicht im DataFrame – wird übersprungen.")
            continue

        mean_churn = df.groupby(column)[response].mean()

        df[column + "_" + response] = df[column].map(mean_churn)

    return df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    X = pd.DataFrame()
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]
    X[keep_cols] = df[keep_cols]
    y = df[response]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return x_train, x_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """

    reports = {
        "LR Train": classification_report(
            y_train, y_train_preds_lr, output_dict=True, zero_division=0
        ),
        "LR Test": classification_report(
            y_test, y_test_preds_lr, output_dict=True, zero_division=0
        ),
        "RF Train": classification_report(
            y_train, y_train_preds_rf, output_dict=True, zero_division=0
        ),
        "RF Test": classification_report(
            y_test, y_test_preds_rf, output_dict=True, zero_division=0
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    titles = [
        "Logistic Regression - Train",
        "Logistic Regression - Test",
        "Random Forest - Train",
        "Random Forest - Test",
    ]

    for ax, (key, report), title in zip(axes.flat, reports.items(), titles):
        df_report = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'accuracy' row
        sns.heatmap(
            df_report, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=False
        )
        ax.set_title(title)
        ax.set_ylabel("Classes")
        ax.set_xlabel("Metrics")

    plt.tight_layout()
    plt.savefig("images/classification_report.png")
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["sqrt", "log2"],  # 'auto' entfernen!
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc = LogisticRegression(solver="lbfgs", max_iter=5000)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )
    feature_importance_plot(
        cv_rfc.best_estimator_, X_train, "images/feature_importance.png"
    )

    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test, ax=ax, alpha=0.8)
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8
    )
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.tight_layout()
    plt.savefig("images/ROC_Curve.png")
    plt.close()


if __name__ == "__main__":
    df = import_data(PATH)
    perform_eda(df)
    category_lst = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    df = encoder_helper(df, category_lst, "Churn")
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")

    train_models(X_train, X_test, y_train, y_test)

    y_train_preds_rf = joblib.load("./models/rfc_model.pkl").predict(X_train)
    y_test_preds_rf = joblib.load("./models/rfc_model.pkl").predict(X_test)
    y_train_preds_lr = joblib.load("./models/logistic_model.pkl").predict(X_train)
    y_test_preds_lr = joblib.load("./models/logistic_model.pkl").predict(X_test)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )
