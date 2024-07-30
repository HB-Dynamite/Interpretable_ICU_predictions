from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import copy
from utils.logger import log


class Imputer(BaseEstimator, TransformerMixin):
    """
    Class which imputs missing values for the given splits.

    Methods:
    --------
    - fit_data()
    - transform_data()

    Note:
    -----
    Intended to be used through a Dataset Object!
    """

    def __init__(self, X_train, numerical_cols, categorical_cols, params):
        """
        Intializes the class.
        Parameters:
        X_train = X_train split used for fitting the imputer
        numerical_cols = list of column names of numerical features
        categorical_cols = list of column names of categorical features
        imputation_method = can either be one of "mean", "median", "knn" or "mice"
        flag_missing = (True/False) create dummy column which indicates if original values was missing for each imputed variable
        """
        self.X_train = X_train
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.imputation_method = params["imputation_method"]
        self.flag_missing = params["flag_missing"]
        self.imputer = None

        self.imputers = {
            "mean": SimpleImputer(strategy="mean"),
            "median": SimpleImputer(strategy="median"),
            "knn": KNNImputer(n_neighbors=10),
            "iterative_RF": IterativeImputer(
                estimator=RandomForestRegressor(), max_iter=10, random_state=1991
            ),
            "iterative_LR": IterativeImputer(
                estimator=LinearRegression(), max_iter=10, random_state=1991
            ),
        }

    def fit(self, X=None):
        """
        Function which fits the chosen imputer to X_train
        """
        X = copy.deepcopy(self.X_train) if X is None else X
        # create flagged columns if flag_missing True
        if self.flag_missing:
            for col in X:
                flag = col + "_missing"
                X[flag] = np.where(X[col].isnull(), 1, 0)
                self.categorical_cols.append(flag)

        if self.imputation_method not in self.imputers:
            raise KeyError(
                "Unknown form of imputation! Can be one of 'mean', 'median', 'knn' or 'mice'!"
            )
        # select the imputer according to the given imputation method
        self.imputer = self.imputers[self.imputation_method]
        # fit the imputer using the X_train split
        self.imputer.fit(X)

        return self

    def transform(self, X=None):
        """
        Function which transforms the given split using the fitted imputer.
        """
        X = self.X_train if X is None else X
        if self.imputer is None:
            raise RuntimeError("You must call fit_data before calling transform_data")
        # Create missing flags if needed
        if self.flag_missing:
            for col in X:
                flag = col + "_missing"
                X[flag] = np.where(X[col].isnull(), 1, 0)
        # transform the given X_split
        X = self.imputer.transform(X)
        # recreate DataFrame after transformation
        X = pd.DataFrame(X, columns=self.X_train.columns)

        # Identify columns with missing values
        missing_columns = X.columns[X.isna().any()].tolist()

        if len(missing_columns) > 0:
            log.warning(
                f"DataFrame still has missing values after imputation in columns: {missing_columns}"
            )
        else:
            log.info("No missing values found after imputation.")

        assert (
            len(missing_columns) == 0
        ), f"DataFrame has still missing values after Impuation, Missing values found in columns:{missing_columns}"

        return X
