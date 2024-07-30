import pandas as pd
import numpy as np
import json

# from utils.params import *
from utils.config import UTILS_DIR
from utils.logger import log


class Preprocessor:
    """
    Class which applies all neceassary pre processing steps to the given splits.

    Methods:
    --------
    - height_weight_switch()
    - outlier_removal()
    - remove_minors()
    - remove_missing_data()
    - balance()
    - preprocess()


    Note:
    -----
    Intended to be used through a Dataset Object!
    """

    def __init__(
        self,
        data_name,
        X_train,
        X_test,
        y_train,
        y_test,
        prepro_params: {},
        cat_cols=None,
    ):
        """
        Intializes the class.
        Parameters:
        data_name = either 'tudd' or 'mimic'
        X_train, X_test, y_train, y_test = specific splits to be pre processed.
        cat_cols = None or list of categorical column names
        """
        self.data_name = data_name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cat_cols = cat_cols
        self.prepro_params = prepro_params

    def outlier_removal(self, df):
        """'
        Function that removes outliers as specified in limits.json.
        """
        ##### we do not use this for now as we remove aoutlier before feature aggregation#####

        # load outlier json

        limits_path = UTILS_DIR / self.prepro_params["outlier_limits"]

        with open(limits_path, "rb") as file:
            outlier_dict = json.load(file)
            print(limits_path)
        # for each object in outlier json set upper and lower bounds and delete observations if exceed values
        for i in outlier_dict:
            try:
                upper = outlier_dict[i]["upper_bound"]
                lower = outlier_dict[i]["lower_bound"]
                log.info(
                    f"Removing outlieser for feature > {i} <. Limits are {upper} and {lower}."
                )
                index_list = df.loc[
                    (df.loc[:, i] < lower) | (df.loc[:, i] > upper), :
                ].index

                df.loc[index_list, i] = np.nan
            except Exception as e:
                log.info(e)
        return df

    def remove_minors(self, df):
        """
        Functions which removes all observations with age less than the min_age.
        """
        ##### this function does nothing as no minors are in the dataset (just to make sure #####
        # Set a minimal age to be included to the dataset
        min_age = self.prepro_params["min_age"]
        df.drop(df.loc[df["Age"] < min_age].index, axis="index", inplace=True)
        return df

    def remove_missing_data(self, df):
        """
        Function which removes all observations if missing rate is larger than minrate.
        """
        # set the minimal rate of missing feature for an observation to be included in the dataset
        minrate = self.prepro_params["missing_rate"]
        # calculate fraction of missings for each observation
        missing_rate = df.isna().sum(axis=1) / len(df.columns)
        # retrive all indices which have a missing rate higher than minrate
        to_drop = df[missing_rate > minrate].index
        # drop observations from data frame
        df.drop(to_drop, axis=0, inplace=True)
        return df

    def adapt_max_age(self, df, max_age=90):
        """
        Set ages larger than max_age to max_age.
        """
        ##### this function does nothing as no people above 90 are in the dataset (just to make sure) #####
        max_age = self.prepro_params["max_age"]
        df["Age"] = df["Age"].where(df["Age"] < max_age, max_age)
        return df

    def preprocess(self):
        """
        Apply all of the functions as named in steps to both the train and test data seperately.
        """
        # create data pairs (.copy() mighte not be nessary)
        data_pairs = [
            (self.X_train.copy(), self.y_train.copy()),
            (self.X_test.copy(), self.y_test.copy()),
        ]

        data_pair_names = ["train", "test"]
        steps = self.prepro_params["steps"]

        prepro_methods = {
            "remove_minors": self.remove_minors,
            "adapt_max_age": self.adapt_max_age,
            "remove_missing_data": self.remove_missing_data,
            "outlier_removal": self.outlier_removal,
        }

        if not isinstance(steps, list) or not all(
            isinstance(step, str) for step in steps
        ):
            log.error(
                "'steps' should be a list of strings representing preprocessing functions."
            )
            raise ValueError(
                "'steps' should be a list of strings representing preprocessing functions."
            )

        for i, (X, y) in enumerate(data_pairs):
            log.debug(f"Processing {data_pair_names[i]} data pair...")
            # Combine X and y along the column axis.
            # To ensure all targets are removed with the obersvations
            data = pd.concat([X, y], axis=1)

            for step in steps:
                if step not in prepro_methods:
                    log.warning(f"Unknown preprocessing function: {step}. Skipping...")
                    continue
                try:
                    log.info(f"Applying {step}...")
                    data = prepro_methods[step](data)
                    log.info(f"{step} Done...")
                except Exception as e:
                    log.error(f"Failed to do step {step}. Reason: {str(e)}")

            # Split `data` back into X and y.
            X, y = data.iloc[:, :-1], data.iloc[:, -1]

            # Store back the processed data
            data_pairs[i] = (X, y)

        # Extract the data back from the list
        self.X_train, self.y_train = data_pairs[0]
        self.X_test, self.y_test = data_pairs[1]

        return self.X_train, self.X_test, self.y_train, self.y_test
