# this is just to test my git
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from classes.Preprocessor import Preprocessor
from classes.ScalerEncoder import ScalerEncoder
from classes.Imputer import Imputer
from classes.Balancer import Balancer

from classes.FeatureSelector import FeatureSelector
import matplotlib.pyplot as plt
import json
import datetime
from utils.logger import log
import os
from utils.config import DATA_DIR
import sys
import copy
import dill as pickle


class DataSet:
    """
    A DataSet class used to handle data preprocessing, scaling, encoding, and imputation tasks.
    Usage is in accordance with sklearn standard.

    Methods:
    --------
    - setup()
    - set_df_types()
    - create_X_y()
    - init_splits()
    - split_shapes
    - describe
    - histograms
    - split_describe(name = 'initial', split_type = 'X_train')
    - split_histograms(name = 'initial', split_type = 'X_train)
    - add_splits(split_name , X_train, X_test, y_train, y_test)
    - get_splits(split_name)
    - set_preprocessor(split_name = 'initial')
    - preprocess(data_name = 'preprocessed',
                steps=["remove_minors",
                        "adapt_max_age",
                        "remove_missing_data",
                        "outlier_removal"]
                        )
    - set_scaler_encoder(split_name = 'preprocessed')
    - scaler_encoder_fit()
    - scaler_encoder_transform(split_name = 'preprocessed')
    - scaler_encoder_fit_transform()
    - set_imputer(split_name = 'scaled_encoded', imputation_method = 'mean', flag_missing = False)
    - imputer_fit()
    - imputer_transform(split_name = 'scaled_encoded')
    - imputer_fit_transform(split_name = 'scaled_encoded')
    - YScaler (object, optional): Scaler object for the target variable, used in regression tasks.

    Examples:
    ---------

    # setup Dataset
    TUDD = Dataset(
        df=tudd_extracted,
        name="tudd",
        target = 'exitus',
        classification = True,
        cat_cols=["gender"],
        num_cols=tudd_extracted.columns.drop(["gender", "exitus"]),
    )

    # setup preprocessor and preprocess
    TUDD.set_preprocessor()
    TUDD.preprocess()

    # setup scaler_encoder and scale and encode
    TUDD.set_scaler_encoder("preprocessed")
    TUDD.scaler_encoder_fit_transform()

    # setup imputer and impute
    TUDD.set_imputer(
        split_name="scaled_encoded", imputation_method="mean", flag_missing=False
    )
    TUDD.imputer_fit_transform()
    """

    def __init__(
        self,
        name,
        target: str,
        classification: bool,
        dataset_params=None,
        cat_cols=None,
        num_cols=None,
        df=None,
        custom=False,
        load_from_json=False,
        random_state=None,
        YScaler=None,
        scaler_encoder=None,
    ):
        """
        Initializes the DataSet class and automatically creates stratified train-test-splits (70-30).

        Parameters:
        - df (pd.DataFrame): Initial data frame.
        - name (str): Name to identify the dataset for later use.
        - cat_cols (list): List of categorical columns of the data frame.
        - num_cols (list): List of numeric columns of the data frame.
        """
        self.df = df
        self.target = target
        self.classification = classification
        self.dataset_params = dataset_params
        self.custom = custom
        self.load_from_json = load_from_json
        self.random_state = (
            random_state if random_state is not None else dataset_params["random_state"]
        )
        self.YScaler = YScaler
        self.ScalerEncoder = scaler_encoder
        if custom == False:
            self.setup(name, cat_cols, num_cols)
            if load_from_json == False:
                self.set_df_types()
                self.init_splits()

    def setup(self, name, cat_cols, num_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        # TODO check datatype of num cols
        # and check why it fails the encoder if num_cols is list

        self.cols = list(num_cols) + (cat_cols)
        self.name = name
        self.Preprocessor = None
        self.Imputer = None
        self.split_dict = {}

    def save_split_as_json(
        self, filename=None, split_name=None, path=DATA_DIR / "prepared_for_training"
    ):
        """
        Save the processed dataset splits to a JSON file.

        Args:
            filename (str, optional): The name of the file to save the data to.
            split_name (str, optional): The name of the data split to save.
        """
        # Retrieve dataset splits and metadata
        (X_train, X_test, y_train, y_test, selected_split_name) = self.get_splits(
            split_name=split_name, log_split_name=True, return_split_name=True
        )

        #  deepcopy
        X_train_copy = copy.deepcopy(X_train)
        X_test_copy = copy.deepcopy(X_test)
        y_train_copy = copy.deepcopy(y_train)
        y_test_copy = copy.deepcopy(y_test)

        y_train_copy = pd.DataFrame(y_train_copy)
        y_test_copy = pd.DataFrame(y_test_copy)

        if isinstance(y_train_copy, pd.Series):
            y_train_copy = y_train_copy.to_frame()
        if isinstance(y_test_copy, pd.Series):
            y_test_copy = y_test_copy.to_frame()

        # write data and metadata into a dictionary
        data_splits = {
            "X_train": X_train_copy.to_dict(orient="index"),
            "X_test": X_test_copy.to_dict(orient="index"),
            "y_train": y_train_copy.to_dict(orient="index"),
            "y_test": y_test_copy.to_dict(orient="index"),
        }

        data_splits["metadata"] = self.get_metadata()
        data_splits["metadata"]["split_name"] = selected_split_name

        # include YScaler parameters if available
        if self.ScalerEncoder.YScaler is not None:
            y_scaler_params = {
                "mean": self.ScalerEncoder.YScaler.mean_[0].tolist(),
                "scale": self.ScalerEncoder.YScaler.scale_[0].tolist(),
            }
        else:
            y_scaler_params = None
        data_splits["metadata"]["YScaler"] = y_scaler_params

        # Determine the name of the dataset if not specified
        if filename is None:
            filename = data_splits["metadata"]["name"]
            filename = "data_" + filename + ".json"
            filename = filename.lower()

        filename_pkl = data_splits["metadata"]["name"]
        filename_pkl = "Scaler_Encoder_" + filename_pkl + ".pkl"
        filename_pkl = filename_pkl.lower()

        # Create a subdirectory for the dataset
        dataset_dir = os.path.join(path, data_splits["metadata"]["name"])
        os.makedirs(dataset_dir, exist_ok=True)

        # Save to JSON file
        path_json = os.path.join(dataset_dir, filename)
        with open(path_json, "w") as file:
            json.dump(data_splits, file)
            log.info(f"Dataset JSONn saved to {path_json}")

        path_pkl = os.path.join(dataset_dir, filename_pkl)
        with open(path_pkl, "wb") as file:
            pickle.dump(self.ScalerEncoder, file)
            log.info(f"Pickled ScalerEncoder class saved to {path_pkl}")

    def set_params(self, dataset_params):
        self.dataset_params = dataset_params

    def set_df_types(self):
        for feature in self.cat_cols:
            self.df[feature] = self.df[feature].astype("category")

    def create_X_y(self, df):
        """
        Prepare the data by setting up X and y.
        """

        X = df.drop(self.target, axis=1)

        # Convert values to integer
        if self.classification:
            y = df[self.target].astype(int, errors="ignore")
        else:
            y = df[self.target].astype(float, errors="ignore")

        return X, y

    def init_splits(self, random_state=1991):
        """
        Function to create and store the inital train and test splits.
        """
        try:
            log.info("Initializing train-test splits for the dataset.")
            X, y = self.create_X_y(self.df)

            # set stratification
            stratify = y if self.classification else None
            #
            train_size = self.dataset_params.get("train_size", 0.8)
            # split data
            # if full_cv is True, the full dataset is used for training and testing
            if self.dataset_params.get("full_cv", False):
                X_train, X_test, y_train, y_test = X, X, y, y
            # othewise, the data is split into train and test sets
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    train_size=train_size,
                    random_state=random_state,
                    shuffle=True,
                    stratify=stratify,
                )
            # create dictionary of splits
            self.add_splits(
                "initial",
                X_train,
                X_test,
                y_train,
                y_test,
            )
            log.info("Train-test splits created successfully.")
        except Exception as e:
            log.error(f"Error in init_splits: {e}")
            raise

    @property
    def split_shapes(self, split=None):
        """print the shape of the split"""
        X_train, X_test, y_train, y_test = self.get_splits(split_name=split)
        print(
            "Shape of '{}': X_train: {} X_test: {} y_train: {} y_test: {}".format(
                split, X_train.shape, X_test.shape, y_train.shape, y_test.shape
            )
        )

    @property
    def describe(self):
        """
        Retrieve summary statistic of the input data
        """
        print("Summary Statistics of Input Data")
        return self.df.describe()

    @property
    def histograms(self):
        """
        Plot a histogram for each column of the input data.
        """
        print("Histograms of Input Data")
        for column in self.df.columns:
            if self.df[column].dtype != "category":
                bins = (
                    self.df[column].nunique()
                    if self.df[column].nunique() < 100
                    else 100
                )
                self.df[column].plot.hist(bins=bins)
                plt.title(f"Histogram of {column}")
                plt.show()
            else:
                print(f"{column} is not numeric and thus can not be plotted")

    def split_describe(self, name="initial", split_type="X_train"):
        """
        Returns the summary statistics of the chosen split.
        """
        return self.split_dict[name][split_type].describe()

    def split_histograms(self, name="initial", split_type="X_train", rescale=False):
        """
        #TODO: check if needed
        Plots the histograms of all columns of the chosen split.
        """
        split = self.split_dict[name][split_type]
        if (rescale == True) and (type(split) == pd.DataFrame):
            split = self.ScalerEncoder.inverse_transform_X(split)
        if (type(split) == pd.Series) or (type(split) == np.ndarray):
            if rescale:
                split = self.ScalerEncoder.inverse_transform_y(split)

            if len(np.unique(split)) == 2:
                plt.hist(split, bins=2)
            else:
                plt.hist(split, bins=len(np.unique(split)))
            if split_type.lower().startswith("y"):
                plt.title(f"Histogram of {split_type} of {name} {split_type}")
            else:
                plt.title(f"Histogram of {split.name} of {name} {split_type}")
            plt.show()
        else:
            for column in split:
                if split[column].dtype != "category":
                    bins = (
                        split[column].nunique()
                        if split[column].nunique() < 100
                        else 100
                    )
                    print(bins)
                    split[column].plot.hist(bins=bins)
                    plt.title(f"Histogram of {column} of {name} {split_type}")
                    plt.show()
                else:
                    print(f"{column} is not numeric and thus can not be plotted")

    def add_splits(self, split_name, X_train, X_test, y_train, y_test):
        """
        Creates a dictionary of the train-test-splits.
        Allows for multiple splits by changing the name of the key (split_name).
        """
        split_counter = len(list(self.split_dict.keys()))
        self.split_dict[split_counter] = {
            "name": split_name,
            "split": {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            },
        }

    def get_splits(
        self, split_name=None, log_split_name=False, return_split_name=False
    ):
        """
        Function to retrieve the stored splits given a split_name or the latest split if no split name is provided
        """
        if split_name:
            for _, value in self.split_dict.items():  # value is nested dict
                if value["name"] == split_name:
                    splits = value["split"]
                    selected_split_name = value["name"]
                    if log_split_name:
                        log.debug(f"{selected_split_name} split selected")
                    return (
                        (
                            splits["X_train"],
                            splits["X_test"],
                            splits["y_train"],
                            splits["y_test"],
                            selected_split_name,
                        )
                        if return_split_name
                        else (
                            splits["X_train"],
                            splits["X_test"],
                            splits["y_train"],
                            splits["y_test"],
                        )
                    )
            log.critical(f"No split found with name {split_name}.")
            sys.exit(1)
        else:
            if self.split_dict:
                highest_int = max(self.split_dict.keys())
                splits = self.split_dict[highest_int]["split"]
                if log_split_name:
                    selected_split_name = self.split_dict.get(highest_int, {}).get(
                        "name", {}
                    )

                    log.debug(f"{selected_split_name} split selected")
                return (
                    (
                        splits["X_train"],
                        splits["X_test"],
                        splits["y_train"],
                        splits["y_test"],
                        selected_split_name,
                    )
                    if return_split_name
                    else (
                        splits["X_train"],
                        splits["X_test"],
                        splits["y_train"],
                        splits["y_test"],
                    )
                )
            else:
                log.critical("No split available.")
                sys.exit(1)

    def save_splits(self, file_path):
        """
        Saves all the data splits in the split_dict to a JSON file.

        Parameters:
        - file_path (str): The directory path where the file will be saved.
        """
        # Format the current time for the file name
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_name = f"{file_path}/{self.name}_splits_{current_time}.json"

        # Convert DataFrames in split_dict to JSON strings
        json_splits = {
            split_name: {
                key: value.to_json(orient="split") for key, value in split.items()
            }
            for split_name, split in self.split_dict.items()
        }

        # Save to JSON file
        with open(file_name, "w") as file:
            json.dump(json_splits, file, indent=4)

        print(f"Splits saved to {file_name}")

    def set_preprocessor(self, split_name="initial"):
        """
        Function to initialize the Preprocessor given the splits.
        """
        prepro_params = self.dataset_params["Preprocessor"]
        # retrieve splits
        splits = self.get_splits(split_name)
        # unlist splits
        X_train, X_test, y_train, y_test = splits
        # intialize Preprocessor
        # if self.reproduce_study:
        #     print("true in dataset")
        self.Preprocessor = Preprocessor(
            self.name,
            X_train,
            X_test,
            y_train,
            y_test,
            prepro_params,
            self.cat_cols,
        )

    def preprocessing(
        self,
        data_name="preprocessed",
    ):
        try:
            # apply preprocessing
            X_train, X_test, y_train, y_test = self.Preprocessor.preprocess()
            # store preprocessed splits
            self.add_splits(data_name, X_train, X_test, y_train, y_test)
        except Exception as e:
            log.error(f"Error in preprocessing: {e}")
            raise

    def set_scaler_encoder(self, split_name="preprocessed"):
        """
        Function which intializes the ScalerEncoder for scaling and encoding of the splits.
        """
        try:
            # retrieve stored splits
            splits = self.get_splits(split_name)
            # unlist splits
            X_train, X_test, y_train, y_test = splits
            # intialize ScalerEncoder
            self.ScalerEncoder = ScalerEncoder(X_train, self.num_cols, self.cat_cols)
        except Exception as e:
            log.error(f"Error in scaling: {e}")
            raise

    def scaler_encoder_fit(self, split_name="preprocessed"):
        """
        Function which fits scaler_encoder to splits.
        """
        try:
            # retrieve splits
            splits = self.get_splits(split_name)
            # unlist splits
            X_train, X_test, y_train, y_test = splits
            # fit scaler and encoder to the X_train data
            self.ScalerEncoder.fit()

            # TODO: this might needs a rework since we now have the task variable
            if self.classification == False:
                self.ScalerEncoder.fit_y(y_train)
        except Exception as e:
            log.error(f"Error in scaling: {e}")
            raise

    def scaler_encoder_transform(self, split_name="preprocessed"):
        """
        Function which transforms the X-splits using the fitted X_train data.
        """
        try:
            # retrieve splits
            splits = self.get_splits(split_name)
            # unlist splits
            X_train, X_test, y_train, y_test = splits
            # transform X_train split
            X_train = self.ScalerEncoder.transform(X_train)
            # transform X_test splits
            X_test = self.ScalerEncoder.transform(X_test)

            if self.classification == False:
                # transfrom y_train split
                y_train = self.ScalerEncoder.transform_y(y_train)
                # transfrom y_test split
                y_test = self.ScalerEncoder.transform_y(y_test)

            # store the transformed splits and regular y-splits to dictionary
            self.add_splits("scaled_encoded", X_train, X_test, y_train, y_test)
        except Exception as e:
            log.error(f"Error in scaling: {e}")
            raise

    def scaler_encoder_fit_transform(self, split_name="preprocessed"):
        """
        Function which fits the ScalerEncoder to X_train and transforms both X-splits.
        """
        try:
            # retrieve splits
            splits = self.get_splits(split_name)
            # unlist splits
            X_train, X_test, y_train, y_test = splits
            # fit ScalerEncoder and Transform X_train
            X_train = self.ScalerEncoder.fit_transform(X_train)
            # Transform X_test as well
            X_test = self.ScalerEncoder.transform(X_test)

            if self.classification == False:
                # fit ScalerEncoder to transform target
                self.ScalerEncoder.fit_y(y_train)
                # transfrom y_train split
                y_train = self.ScalerEncoder.transform_y(y_train)
                # transfrom y_test split
                y_test = self.ScalerEncoder.transform_y(y_test)

            # store the transformed splits and regular y-splits to dictionary
            self.add_splits("scaled_encoded", X_train, X_test, y_train, y_test)
        except Exception as e:
            log.error(f"Error in scaling: {e}")
            raise

    def set_imputer(
        self, split_name="scaled_encoded", imputation_method="mean", flag_missing=False
    ):
        """
        Function which intializes Imputer.
        Parameters:
        split_name = name of the splits used for storing
        imputation_method = can be either mean, median, knn or mice
        flag_missing = (True/False) create dummy columns which flag if observation was originally missing.
        """
        try:
            #
            imputer_params = self.dataset_params["Imputer"]
            # retrieve splits
            splits = self.get_splits(split_name)
            # unlist splits
            X_train, X_test, y_train, y_test = splits
            # intialize imputer
            self.Imputer = Imputer(
                X_train, self.num_cols, self.cat_cols, imputer_params
            )
        except Exception as e:
            log.error(f"Error in imputing: {e}")
            raise

    def imputer_fit(self):
        # def imputer_fit(self, split_name="scaled_encoded"):
        """
        Function to fit the imputer using the X-train split.
        """
        try:
            # # retrieve splits
            # splits = self.get_splits(split_name)
            # # unlist splits
            # X_train, X_test, y_train, y_test = splits
            # # self.Imputer.fit_data(X_train)
            self.Imputer.fit()
        except Exception as e:
            log.error(f"Error in imputing: {e}")
            raise

    def imputer_transform(self, split_name="scaled_encoded"):
        """
        Function which applies the imputation and transform the X-splits.
        """

        # retrieve splits
        splits = self.get_splits(split_name)
        # unlist splits
        X_train, X_test, y_train, y_test = splits
        # transform X_train using fitted imputer
        X_train = self.Imputer.transform(X_train)
        # transform X_test using fitted imputer
        X_test = self.Imputer.transform(X_test)
        # retrieve imputation name
        imputation_method = self.Imputer.imputation_method
        # save imputed splits to dictionary (key = split_name)
        self.add_splits("imputed", X_train, X_test, y_train, y_test)

    def imputer_fit_transform(self, split_name="scaled_encoded"):
        """
        Function which fits the Imputer to X_train and transforms both X-splits.
        """

        # retrieve splits
        splits = self.get_splits(split_name)
        # unlist splits
        X_train, X_test, y_train, y_test = splits
        # fit the imputer to X_train and automatically transform it
        X_train = self.Imputer.fit_transform(X_train)
        # additonally transform X_test as well
        X_test = self.Imputer.transform(X_test)
        # retrieve imputation name
        imputation_method = self.Imputer.imputation_method
        # save imputed splits to dictionary (key = split_name)
        self.add_splits("imputed", X_train, X_test, y_train, y_test)

    def set_balancer(
        self, split_name="imputed", oversampling_rate=0.5, undersampling_rate=0.5
    ):
        """
        Sets up the balancer with specified oversampling and undersampling rates.
        """
        splits = self.get_splits(split_name)
        X_train, _, y_train, _ = splits
        self.balancer = Balancer(
            X_train, y_train, oversampling_rate, undersampling_rate
        )

    def balancer_resample(self, split_name="imputed"):
        """
        Applies the balancer to the training data using the data set in the balancer.
        """
        _, X_test, _, y_test = self.get_splits(split_name)
        X_train_balanced, y_train_balanced = self.balancer.fit_resample()
        log.debug(
            f"Class distribution after balancing: {np.bincount(y_train_balanced)}"
        )
        self.add_splits("balanced", X_train_balanced, X_test, y_train_balanced, y_test)

    def set_selector(self, split_name=None, selector_params=None):
        """
        Function to initialize the FeatureSelector.

        Parameters:
        split_name (str): The name of the split to apply feature selection to.
        strategy (str): The strategy for feature selection (e.g., 'filter', 'wrapper', 'embedded').
        selector_params (dict): Additional parameters for the feature selector.
        """
        try:
            # Retrieve splits
            splits = self.get_splits(split_name)
            X_train, X_test, y_train, y_test = splits
            # Initialize FeatureSelector
            self.FeatureSelector = FeatureSelector(
                X_train,
                y_train,
                selector_params,
                num_cols=self.num_cols,
                cat_cols=self.cat_cols,
            )
        except Exception as e:
            log.error(f"Error in setting selector: {e}")
            raise

    def selector_fit(self):
        """
        Function to fit the FeatureSelector.
        """
        try:
            self.FeatureSelector.fit()
        except Exception as e:
            log.error(f"Error in fitting selector: {e}")
            raise

    def selector_transform(self, split_name="imputed"):
        """
        Function to transform the data using the fitted FeatureSelector.

        Parameters:
        split_name (str): The name of the split to apply feature selection to.
        """
        try:
            # Retrieve splits
            splits = self.get_splits(split_name)
            X_train, X_test, y_train, y_test = splits
            # Transform X_train and X_test using the fitted FeatureSelector
            X_train = self.FeatureSelector.transform(X_train)
            X_test = self.FeatureSelector.transform(X_test)
            # update num and cat cols:
            cols = self.FeatureSelector.selected_features_indices
            self.cols = [self.cols[i] for i in cols]
            print(self.cols)
            self.num_cols = self.FeatureSelector.get_num_cols()
            self.cat_cols = self.FeatureSelector.get_cat_cols()
            # Store the transformed splits to dictionary
            self.add_splits("selected", X_train, X_test, y_train, y_test)
        except Exception as e:
            log.error(f"Error in transforming selector: {e}")
            raise

    def preprocess(self):
        log.info("Starting preprocessing...")
        self.set_preprocessor()
        self.preprocessing(data_name="preprocessed")
        log.info("Preprocessing completed.")

    def scale_encode(self):
        log.info("Starting scaling and encoding...")
        self.set_scaler_encoder("preprocessed")
        self.scaler_encoder_fit()
        self.scaler_encoder_transform()
        log.info("Scaling and encoding completed.")

    def impute(self):
        log.info("Starting imputation...")
        log.info(
            f"Using imputation method:{self.dataset_params['Imputer']['imputation_method']}"
        )
        self.set_imputer(
            split_name="scaled_encoded",
            imputation_method=self.dataset_params["Imputer"]["imputation_method"],
            flag_missing=self.dataset_params["Imputer"]["flag_missing"],
        )
        self.imputer_fit()
        self.imputer_transform()
        log.info("Imputation completed.")

    def balance(self):
        log.info("Starting data balancing...")

        self.set_balancer(
            oversampling_rate=self.dataset_params["Balancer"]["oversampling_rate"],
            undersampling_rate=self.dataset_params["Balancer"]["undersampling_rate"],
        )
        self.balancer_resample(split_name="imputed")
        log.info("Data balancing completed.")

    def select(self):
        self.set_selector(
            # split_name="imputed",
            strategy=self.dataset_params["Selector"]["strategy"],
            selector_params=self.dataset_params["Selector"]["params"],
        )
        self.selector_fit()
        self.selector_transform()

    def prepare_dataset(self):
        self.preprocess()
        self.scale_encode()
        self.impute()
        if self.classification:
            self.balance()
        self.split_shapes
        # self.select()

    def select_features(self):
        self.set_selector(
            selector_params=self.dataset_params["Selector"],
        )
        self.selector_fit()
        self.selector_transform()

    def get_metadata(self):
        """Returns meatadata of the object"""
        return {
            "target": self.target,
            "classification": self.classification,
            "cat_cols": self.cat_cols,
            "name": self.name,
            "num_cols": self.num_cols,
            "custom": self.custom,
            "random_state": self.random_state,
            "YScaler": self.YScaler,
        }

    def set_metadata(self, metadata):
        """Sets metadata of the object"""
        self.classification = metadata["classification"]
        self.target = metadata["target"]
        self.cat_cols = metadata["cat_cols"]
        self.name = metadata["name"]
        self.num_cols = metadata["num_cols"]
        self.custom = metadata["custom"]
        self.cols = list(self.num_cols) + (self.cat_cols)

    # function to import YScaler, column_transformer and X_scaled_dict into ScalerEncoder Class
    def initialize_scaler_encoder(self):
        if self.column_transformer and self.X_scaled_dict:

            self.ScalerEncoder = ScalerEncoder(
                X_train=self.X_scaled_dict["X_train"],
                numerical_cols=self.num_cols,
                categorical_cols=self.cat_cols,
                YScaler=self.YScaler,
            )

            self.ScalerEncoder.column_transformer = self.column_transformer
            self.ScalerEncoder.X_scaled_dict = self.X_scaled_dict
        else:
            log.warning("column_transformer or X_scaled_dict not available.")
