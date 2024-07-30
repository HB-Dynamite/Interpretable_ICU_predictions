# %%
import numpy as np
import sklearn
import os
import pandas as pd
import json
import hashlib

# TODO: check this from utils.params import *
from classes.Trainer import Trainer
from classes.Preprocessor import Preprocessor
from classes.Dataset import DataSet
from classes.Trainer import Trainer
from classes.Evaluator import Evaluator
from classes.Plotter_v2 import Plotter
from classes.Memorizer import Memorizer
from sklearn.model_selection import train_test_split
from utils.logger import log
from sklearn.preprocessing import StandardScaler

from pprint import pprint


class Pipeline:
    def __init__(
        self,
        # method: str,
        dataset: DataSet,
        pipe_params: dict,
        test_size=0.3,
        # data_path="data/",  # os.path.sep.join(os.getcwd().split(os.path.sep)[:-1]) + "/data/" this is only needed for the data extraction.
        # calc_differences=False,
        # prepro_steps=None,
    ):
        """
        Initializes Pipeline class.

        Paremeters:
        - method (str): Name of the training method which is used for the dataset. Options include:
            - 'simple': train and test on the same datset.
            - 'transfer': Custom dataset, train on source and test on target dataset
            - 'fractional': Custom dataset, train on a fraction of source, test on full source test set
            - 'combination': Custom dataset, train and test on the combination of both datasets
            - 'frac_comb': Custom dataset, train on the full source data and a fraction of the target set.
                            Tests on the non-train target test set
        - models (str, optional): Specifies the model which is trained. If not specified, all svaliable models are trained. Options include:
            - 'EBM'
            - 'XGB'
            - 'IGANN'
            - 'PYGAM'
            - 'LR' #TODO: this is the LinReg or Logistik regression?
            - 'LinReg' # this is redundant?
            - 'MLP'
        - test_size (str, optional): The size of the test dataset. Default ist set to 0.3.
        - classification (bool, optional): Boolean specification if the pipeline is used for Regression or
            clasification task
        - data_path (str, optional): The path to the data directory
        - simple_dataset (str, optional): The dataset to be used when the 'simple' method is chosen.
            Must be either 'tudd' or 'mimic'. Only required and used if the method is 'simple'.
        - source_dataset (str, optional): The source dataset for Custom dataset methods.
             Must be either 'tudd' or 'mimic'.
        - target_dataset (str, optional): The target dataset for Custom dataset methods.
            Must be either 'tudd' or 'mimic'.
        - reproduce study (bool, optional): Default = False. Aims for reproducing Interpretable ML and ICUs study
        - target (str, optional): Default = 'exitus'.Target variable for classification, needs to be changed only for study reproduction. Options include:
            - 'exitus'
            - 'LOS3': length of stay > 3 days
            - 'LOS7'_ length of stay > 7 days
        - save_extracted_features (bool, optional): Default = False. Saves extracted feature table to csv.
        - hpo (bool, optional): Default = False. hyperparameter tuning.
        - feature_selection (str, optional): Default = 'manual'. Specifies Feature selection method for study reproduction. Options include:
            - 'manual: applies manually selected features (mean values)
            - 'automated': applies automated (SFFS) selected features
        - include_sensitive_features (bool, optional): Default = True. Specifies if sensitive features (gender, age, ethnicity) are included

        Note:
        - The 'simple_dataset' parameter is exclusively used when the method is set to 'simple'.
        - The 'source_dataset' and 'target_dataset' parameters are required when the method is
        set to either 'transfer' or 'frac'. These are not used with the 'simple' method.
        """
        log.info("Initializing the pipeline...")
        self.pipe_params = pipe_params
        self.Trainer_params = self.pipe_params["Trainer"]
        self.Evaluator_params = self.pipe_params["Evaluator"]
        self.Plotter_params = self.pipe_params["Plotter"]
        self.Memorizer_params = self.pipe_params["Memorizer"]

        # self.calc_differences = calc_differences
        self.dataset = dataset
        self.classification = self.dataset.classification
        # self.method = method
        self.test_size = test_size
        # self.differences = None

        self.fitted_models = {}
        self.results = {}
        self.plots = {}

        # self.validate_inputs()

        self.logger = log

    def train(self):
        try:
            self.logger.info("Starting model training...")
            trainer = Trainer(
                dataset=self.dataset,
                hpo_file=self.Trainer_params["hpo_file"],
                split_name=self.Trainer_params["input_split"],
            )
            models = self.Trainer_params["models"]
            self.fitted_models = trainer.train_models(models)
            self.logger.info("Model training completed.")
        except Exception as e:
            self.logger.error(f"Error in training: {e}")
            raise

    def evaluate(self):
        self.logger.info("Starting model evaluation...")

        try:
            # TODO: check if it would be better to chose the splits in the Evaluator itself so its hard coded from the imputations method
            # _, X_test, _, y_test = self.dataset.get_splits("imputed")
            _, X_test, _, y_test = self.dataset.get_splits()
            YScaler = self.reconstruct_yscaler()
            if YScaler is None:
                YScaler = self.dataset.ScalerEncoder.YScaler
            evaluator = Evaluator(
                model_dict=self.fitted_models,
                classification=self.dataset.classification,
                target=self.dataset.target,
                # This should be handled elsewhere memorzier? Evaluator should just return results. So I see the problems with the plots
                # run_dir=self.run_dir,
                X_test=X_test,
                y_test=y_test,
                YScaler=YScaler,
                full_cv=self.dataset.dataset_params["full_cv"],
            )

            self.results = evaluator.test_models()
            if self.pipe_params["Evaluator"].get("plot_metrics", False):
                self.plots["metrics"] = evaluator.plot_metrics()
            if (
                self.pipe_params["Evaluator"].get("plot_confusion_matrix", False)
                and self.dataset.classification
            ):
                self.plots["confusion_matrix"] = evaluator.plot_confusion_matrices()
            self.logger.info("Model evaluation completed.")
            return self.results
        except Exception as e:
            self.logger.error(f"Error in evaluating: {e}")
            raise

    def plot(self):
        plotter = Plotter(
            dataset=self.dataset,
            params=self.Plotter_params,
            # model_dict=self.fitted_models,
            # run_dir=self.run_dir,
        )
        for model_name, model_details in self.fitted_models.items():
            # pprint(model_details)
            if model_name in self.Plotter_params["models_to_plot"]:
                best_model = model_details["best_model"]
                plotter.plot_model(model_name, best_model)
                # lets do not use this for now
                # plotter.plot_shap(model)

    def memorize(self):
        if self.Memorizer_params["save"] == False:
            self.logger.info("Memorization is turned off.")
            return
        self.logger.info("Starting memorizer...")
        try:
            memorizer = Memorizer(
                self.dataset, self.pipe_params, self.results, self.plots
            )
            memorizer.save_results_to_csv()
            memorizer.save_params_and_results_as_json()
            memorizer.save_plots_as_jpg()
            self.logger.info("Memorization complete...")
        except Exception as e:
            self.logger.error(f"Error in memorizing: {e}")
            raise

    def reconstruct_yscaler(self):
        yScaler = StandardScaler()
        if self.dataset.YScaler is not None:
            yscaler_params = self.dataset.YScaler

            yScaler.mean_ = np.array(yscaler_params["mean"])
            yScaler.scale_ = np.array(yscaler_params["scale"])

            return yScaler
        return None

    def run(self):
        log.info("Running the pipeline...")

        self.train()
        self.evaluate()
        self.plot()
        self.memorize()

        # print(self.results)
        log.info("Pipeline execution completed.")
        return self.results


# %%
