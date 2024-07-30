# %% Setup #
import traceback
import os
import pandas as pd
import numpy as np
import json
import pickle
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# from igann import IGANN
from utils.sklearn_igann import IGANNClassifier, IGANNRegressor
import xgboost
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer, r2_score
from utils.sklearn_pygam import PYGAMClassifier, PyGAMRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from utils.config import HPO_DIR
from sklearn.base import clone
from pprint import pprint
from utils.logger import log


class Trainer:
    """
    This class is intendet to train differen ML models for a givn data set.
    It loads hpo for simple fitting or CV with HPO fitting from the json in the hpo_congfigs folder.

    Methods:
    --------
    - load_train_data()
    - load_hpo_config()
    - train_model()
    - train_EBM()
    - train_XGB()
    - train_IGANN()
    - train_LR()
    - train_PYGAM()
    - train_MLP()
    - train_models()
    - get_model_dict()
    - get_model_params()

    Example:
    --------
    trainer_obj = Trainer(Dataset_obj)
    fitted_models = trainer_objs.train_models(["LR", "MLP"])
    """

    def __init__(
        self,
        dataset,
        hpo_file,
        reproduce_study=False,
        random_state=1991,
        n_cv_folds=5,
        split_name=None,  # if None, get_splits selects the latest split
    ):
        """
        Intializes the class.
        Parameters:
        data_folder = None or path to folder where splits are stored
        random_state = 1991 or some other number to specify fixed state
        X_train = None or data frame of train features
        y_train = None or list of target values
        hpo = (True/False) whether hpo should be performed
        n_cv_folds = (5 or int) number of splits used in cross validation
        """
        self.random_state = random_state
        self.reproduce_study = reproduce_study
        self.classification = dataset.classification
        self.hpo_file = hpo_file
        self.n_cv_folds = n_cv_folds
        self.hpo_dict = {}
        self.model_dict = {}  # create an empty dictionary to store trained models
        self.load_hpo_config()
        self.dataset = dataset
        self.X_train, _, self.y_train, _ = self.dataset.get_splits(
            split_name, log_split_name=True
        )

    def load_hpo_config(self):
        file_path = HPO_DIR / self.hpo_file or HPO_DIR / "default_hpo.json"

        with open(file_path, "r") as hpo:
            self.hpo_dict = json.load(hpo)

    def preprocess_hpos(self, parameters):
        # delete hpo _comments from JSON
        if "_comments" in parameters:
            del parameters["_comments"]

        # insert none values if applicable
        if "max_depth" in parameters:
            parameters["max_depth"] = [
                None if i == -1 else i for i in parameters["max_depth"]
            ]

        if "max_leaf_nodes" in parameters:
            parameters["max_leaf_nodes"] = [
                None if i == -1 else i for i in parameters["max_leaf_nodes"]
            ]

        if "class_weight" in parameters:
            parameters["class_weight"] = [
                None if i == "None" else i for i in parameters["class_weight"]
            ]
        return parameters

    def train_model(self, model_name, model):
        """
        Train a given model using cross-validation and optional HPO.
        """
        if not isinstance(self.y_train, pd.DataFrame):
            y_train = self.y_train.ravel()
        # assign hpo dict to variable
        parameters = self.hpo_dict[model_name]
        parameters = self.preprocess_hpos(parameters)

        # check if the model has the parameters
        valid_params = model.get_params().keys()
        parameters = {
            param: values
            for param, values in parameters.items()
            if param in valid_params
        }

        # Create a custom ROC AUC scorer that explicitly uses predict_proba
        # or create scorer for regression
        if self.classification:
            scorer = make_scorer(
                roc_auc_score, greater_is_better=True, needs_proba=True
            )
        else:
            # use r2 score for regression
            scorer = make_scorer(r2_score, greater_is_better=True)

        # create a KFold object for cv

        if self.classification:
            kf = StratifiedKFold(
                n_splits=self.n_cv_folds, shuffle=True, random_state=1991
            )
        else:
            kf = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=1991)

        # TODO: scoring for regression
        clf = GridSearchCV(model, parameters, cv=kf, n_jobs=-1, scoring=scorer)
        clf.fit(self.X_train, self.y_train)
        log.info(f"Best parameters for {model_name}: {clf.best_params_}")
        best_model = clf.best_estimator_
        best_params = clf.best_params_

        # create Submodels for fold of new CV_fold
        submodels = []

        # create a new KFold object for submodels to evaluate the model on the different folds
        kf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=1337)

        if self.classification:
            for fold_index, _ in kf.split(self.X_train, self.y_train):
                X_train_fold, y_train_fold = (
                    self.X_train.iloc[fold_index],
                    self.y_train.iloc[fold_index],
                )
                submodel = clone(model)
                submodel.set_params(**best_params)
                submodel.fit(X_train_fold, y_train_fold)
                submodels.append(submodel)
        else:
            y_train = pd.Series(y_train)
            for fold_index, _ in kf.split(self.X_train):
                X_train_fold, y_train_fold = (
                    self.X_train.iloc[fold_index],
                    y_train.iloc[fold_index],
                )
                submodel = clone(model)
                submodel.set_params(**best_params)
                submodel.fit(X_train_fold, y_train_fold)
                submodels.append(submodel)

        model_info = {
            "best_model": best_model,
            "best_params": best_params,
            "cv_submodels": submodels,
        }

        self.model_dict[model_name] = model_info

        return model

    def train_EBM(self):
        """
        Train the EBM model on the given training data.
        """
        if self.classification:
            model = ExplainableBoostingClassifier()
        else:
            model = ExplainableBoostingRegressor()
        self.train_model("EBM", model)
        return model

    def train_XGB(self):
        """
        Train the XGBoost model on the given training data.
        """
        if self.classification:
            model = xgboost.XGBClassifier()
        else:
            model = xgboost.XGBRegressor()
        model = self.train_model("XGB", model)
        return model

    def train_IGANN(self):
        """
        Train the IGANN model on the given training data.
        """

        # TODO: Add option for regression cases
        if self.classification:
            model = IGANNClassifier()
        else:
            model = IGANNRegressor()
        model = self.train_model("IGANN", model)
        return model

    def train_LR(self):
        """
        Train Logistic Regresssion on the given data
        """
        if self.classification:
            model = LogisticRegression()
        else:
            model = LinearRegression()

        model = self.train_model("LR", model)
        return model

    def train_PYGAM(self):
        """
        Train PYGAM on the given data
        """

        # Define the terms based on the features in your data
        # For example, if the first two features are continuous and the rest are categorical
        # tms = terms.TermList(
        #     *[
        #         f(i)
        #         if any(
        #             check_str in self.X_train.columns[i]
        #             for check_str in self.dataset.cat_cols
        #         )
        #         else s(i)
        #         for i in range(len(self.X_train.columns))
        #     ]
        # )
        def create_feature_types_string(X_train, cat_cols):
            feature_types = []
            for i in range(len(X_train.columns)):
                if any(check_str in self.X_train.columns[i] for check_str in cat_cols):
                    feature_types.append("f")  # 'f' for factor (categorical)
                else:
                    feature_types.append("s")  # 's' for spline (continuous)
            return ",".join(feature_types)

        # print(self.dataset.cat_cols)
        tms = create_feature_types_string(self.X_train, self.dataset.cat_cols)
        # print(tms)
        # print(type(tms))
        if self.classification:
            model = PYGAMClassifier(terms_string=tms)
            model = clone(model)
        else:
            model = PyGAMRegressor(terms=tms)
        model = self.train_model("PYGAM", model)
        return model

    def train_MLP(self):
        """
        Train Multi-layer Perceptron on the given data
        """

        if self.classification:
            model = MLPClassifier()
        else:
            model = MLPRegressor()
        model = self.train_model("MLP", model)

        return model

    def train_RF(self):
        """
        Train Random Forest on given data.
        """
        if self.classification:
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

        model = self.train_model("RF", model)
        return model

    def train_DT(self):
        """
        Train Decision Tree on given data.
        """
        if self.classification:
            model = DecisionTreeClassifier()
        else:
            model = DecisionTreeRegressor()
        model = self.train_model("DT", model)
        return model

    def train_models(self, model_list):
        """
        Train multiple models by list
        models: list of models
        """
        model_methods = {
            "EBM": self.train_EBM,
            "XGB": self.train_XGB,
            "IGANN": self.train_IGANN,
            "PYGAM": self.train_PYGAM,
            "LR": self.train_LR,
            "MLP": self.train_MLP,
            "RF": self.train_RF,
            "DT": self.train_DT,
        }
        if not isinstance(model_list, list):
            log.error("'models' should be a list of model names.")
            raise AssertionError("'models' should be a list of model names.")
        if not all(isinstance(model, str) for model in model_list):
            log.error("All elements in 'models' should be strings.")
            raise AssertionError("All elements in 'models' should be strings.")
        if self.X_train is None or self.y_train is None:
            log.error("Training data (X_train, y_train) must not be None.")
            raise AssertionError("Training data (X_train, y_train) must not be None.")

        for model in model_list:
            if model not in model_methods:
                log.warning(f"Unknown model: {model}. Skipping...")
                continue

            try:
                log.info(f"Training {model}...")
                model_methods[model]()
                log.info(f"Done with training {model}")
            except Exception as e:
                log.error(f"Failed to train {model}. Reason: {e}")
                traceback.print_exc()

        # pprint(self.model_dict)
        return self.model_dict

    def get_model_dict(self):
        return self.model_dict
