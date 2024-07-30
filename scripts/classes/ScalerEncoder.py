import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import copy


class ScalerEncoder(BaseEstimator, TransformerMixin):
    """
    Class which encodes and scales certain columns of the given splits.

    Methods:
    --------
    - fit_data()
    - transform_data()
    - inverse_transform_X()
    - inverse_transform_feature()

    Note:
    -----
    Intended to be used through a Dataset Object!
    """

    def __init__(self, X_train, numerical_cols, categorical_cols):
        """
        Intializes the class.
        Parameters:
        X_train = X_train splits to be used
        numerical_cols = None or list of numeric column names
        categorical_cols = None or list of categoric column names
        """
        self.X_train = X_train
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.column_transformer = None
        self.X_scaled_dict = {"X_train": None, "X_test": None}
        self.count = 0
        self.YScaler = None

    def fit(self, X=None):
        """
        Function which encodes cateogrical features and fits a standard scaler for numerical features.
        """
        X = self.X_train if X is None else X

        # create a transfer which applies standard scaling to all numeric columns
        numerical_transformers = [
            (f"num{i}", StandardScaler(), [col])
            for i, col in enumerate(self.numerical_cols)
        ]

        def custom_naming_ohe(feature, category):
            return str(feature) + "__" + str(category)

        # transformer list which contains both the numerical_transformer as well as a onehotencoder for all categorical features.
        transformers = [
            *numerical_transformers,
            (
                "ohe",
                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                    # drop="if_binary", #This is easyier to handle those feature in the plotting and interpretation also it seams no Problem for Performance
                    drop=None,
                    feature_name_combiner=custom_naming_ohe,
                ),
                self.categorical_cols,
            ),
        ]

        # create a ColumnTransformer using the list of transformers
        self.column_transformer = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )

        # fit the column transformer using the X data
        self.column_transformer.fit(X)

        return self

    def transform(self, X=None):
        """
        Function which transforms the given data using the fitted column transformer.
        """
        X = self.X_train if X is None else X
        # store index to recreate dataframe
        idx = X.index
        if self.column_transformer is None:
            raise RuntimeError("You must call fit_data before calling transform_data")
        # i hope i do not need that but safety first
        X = copy.deepcopy(X)
        # transform the given data
        X = self.column_transformer.transform(X)

        # recreate data frame
        X = pd.DataFrame(
            X, columns=self.column_transformer.get_feature_names_out(), index=idx
        )

        # store scaled data
        if self.count == 0:
            self.X_scaled_dict["X_train"] = X
        elif self.count == 1:
            self.X_scaled_dict["X_test"] = X
        else:
            pass

        self.count += 1

        return X

    def fit_y(self, y_train):
        self.YScaler = StandardScaler().fit(np.reshape(y_train, (-1, 1)))
        return self

    def transform_y(self, y):
        transformed_y = self.YScaler.transform(np.reshape(y, (-1, 1))).flatten()
        return pd.Series(transformed_y, index=y.index)

    def inverse_transform_y(self, y_scaled):
        return self.YScaler.inverse_transform(np.reshape(y_scaled, (-1, 1))).flatten()

    def inverse_transform_X(self, X=None):
        """
        Function which reverses the transformation given a split for all features of the split.
        """
        X = self.X_scaled_dict["X_train"] if X is None else X
        # Inverse transform the scaled/encoded data to original form
        # X_inversed = self.column_transformer.inverse_transform(X)
        X_inversed = X

        # find names of numeric features
        num_feats = [col for col in X.columns if "num" in col]
        for feature in num_feats:
            X_inversed[feature] = self.inverse_transform_feature(
                feature_name=feature, X=X[feature]
            )
        return X_inversed

    def inverse_transform_feature(self, feature_name: str, X=None):
        """
        Functions which allows for reverse transformation of a specific feature of a given split.
        """
        X = self.X_scaled_dict["X_train"][feature_name] if X is None else X

        # convert pd.Series or pd.DataFrames to numpy array for reshape
        if isinstance(X, pd.core.series.Series) or isinstance(
            X, pd.core.frame.DataFrame
        ):
            X_numpy = X.to_numpy()
        else:
            X_numpy = X
        # Get the transformer for the specified feature
        try:
            feature = feature_name.split("__")[1]
        except IndexError:
            feature = feature_name

        transformer = None
        for name, trans, columns in self.column_transformer.transformers_:
            if feature in columns:
                transformer = trans
                break

        if transformer is None:
            raise ValueError(f"No transformer found for feature '{feature_name}'")

        # Inverse transform the specified feature
        X_inversed_feature = transformer.inverse_transform(
            X_numpy.reshape(-1, 1)
        ).flatten()

        return X_inversed_feature

   