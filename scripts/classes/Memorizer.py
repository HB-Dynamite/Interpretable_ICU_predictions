import os
import numpy as np
import json
import hashlib
import pandas as pd
from matplotlib import pyplot as plt
import datetime
from pandas import Index
from utils.config import RESULTS_DIR
from utils.logger import log
import logging


# TODO:save plotter plots
class Memorizer:
    def __init__(
        self, dataset_object, pipe_params={}, results={}, plots={}, logfile_name=None
    ):
        """
        Initialize the Memorizer class with dataset parameters, pipeline parameters, results, and plots.

        Parameters:
        - dataset_object: An object representing the dataset, containing attributes like name, target, etc.
        - pipe_params (dict, optional): Parameters of the pipeline used in processing the dataset.
        - results (dict, optional): The results of the model(s) after processing the dataset.
        - plots (dict, optional): Plot figures generated during the analysis.

        Returns:
        - None
        """
        self.dataset_params = {
            "name": dataset_object.name,
            "target": dataset_object.target,
            "classification": dataset_object.classification,
            "cat_cols": dataset_object.cat_cols,
            "num_cols": dataset_object.num_cols,
            "custom": dataset_object.custom,
        }
        self.dataset_classname = type(dataset_object).__name__
        self.id = pipe_params["Memorizer"]["id"]
        self.pipe_params = self.flatten_dict(pipe_params)
        self.results = results
        self.plots = plots
        self.base_dir = RESULTS_DIR
        self.csv_file = RESULTS_DIR / "results_summary.csv"
        self.current_datetime = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        if log:
            for handler in log.handlers:
                if isinstance(handler, logging.FileHandler):
                    logfile_path = handler.baseFilename
                    logfile_name = os.path.basename(logfile_path)
                    break

        if logfile_name is not None:
            self.logfile_name = logfile_name

    def save_results_to_csv(self):
        """
        Save the results of the analysis to a CSV file. Combines dataset and pipeline parameters,
        and appends results for each model. Updates existing CSV or creates a new one if not present.

        Parameters:
        - None

        Returns:
        - None
        """
        # make config by combining the dicts
        complete_config = {**self.dataset_params, **self.pipe_params}
        complete_config["DataSetClassName"] = self.dataset_classname

        data_for_csv = []

        # nake one entry in the csv per model
        for model, metrics in self.results.items():
            model_config = complete_config.copy()  # per entry all configs are written
            model_config["model"] = model
            # this is not a bad thing but mybe a date and time would make also a good identifier.
            model_hash = self.config_hash(
                model_config
            )  # hash to create unique identifier

            # make the actual row in csv
            row = {
                "Datetime": self.current_datetime,
                "ModelConfigHash": model_hash,
                "LogfileName": self.logfile_name,
                "ID": self.id,
            }
            row.update(model_config)

            # row.update(metrics)

            # Add individual model metrics
            for metric, value in self.results[model].items():
                if metric == "cv_metrics":
                    continue
                # Avoiding nested cv_metrics for now
                if isinstance(
                    value, np.ndarray
                ):  # Special handling for arrays like confusion matrix
                    value = value.tolist()  # Convert numpy arrays to list
                row[metric] = value

            cv_metrics = self.results[model].get("cv_metrics", None)
            if cv_metrics:
                for metric, values in cv_metrics.items():
                    if metric != "Confusion Matrix":
                        row[f"{metric}_CV_Mean"] = values["mean"]
                        row[f"{metric}_CV_Std"] = values["std"]
            else:
                expected_cv_metrics = [
                    "AUROC",
                    "AUPRC",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-Score",
                ]
                for metric in expected_cv_metrics:
                    row[f"{metric}_CV_Mean"] = None
                    row[f"{metric}_CV_Std"] = None

            # write differences if present
            if "differences" in self.results and model in self.results["differences"]:
                row.update(
                    {
                        "DiffAUROC": self.results["differences"][model]["AUROC"],
                        "DiffAUPRC": self.results["differences"][model]["AUPRC"],
                    }
                )

            data_for_csv.append(row)

        # load or create csv
        if os.path.exists(self.csv_file):
            df_existing = pd.read_csv(self.csv_file)

            # if there are new columns due to a change in the params dict, write them to the existing df
            new_columns = set(data_for_csv[0].keys()) - set(df_existing.columns)
            for new_column in new_columns:
                df_existing[new_column] = None

            # cancat new entries into df
            df_combined = pd.concat(
                [df_existing, pd.DataFrame(data_for_csv)], ignore_index=True, sort=False
            )
        else:
            # elsewise just make a new df
            df_combined = pd.DataFrame(data_for_csv)

        df_combined.drop_duplicates(subset=["ModelConfigHash"], inplace=True)
        df_combined.to_csv(self.csv_file, index=False)
        log.info(f"CSV updated: {self.csv_file}")

    def save_params_and_results_as_json(self):
        """
        Save the parameters and results of the analysis to a JSON file. Creates a directory based on
        a hash of the configuration and saves the data in a readable format.

        Parameters:
        - None

        Returns:
        - None
        """
        run_dir = os.path.join(self.base_dir, self.current_datetime)
        os.makedirs(run_dir, exist_ok=True)

        # params and results to save
        data_to_save = {
            "parameters": {
                "datetime": self.current_datetime,
                "dataset_classname": self.dataset_classname,
                "dataset_params": self.dataset_params,
                "pipe_params": self.pipe_params,
            },
            "results": self.make_json_serializable(self.results),
        }
        # nwrite to json
        with open(os.path.join(run_dir, "params_results.json"), "w") as file:
            json.dump(data_to_save, file, indent=4, default=self.make_json_serializable)

        log.info(f"Parameters and results saved to {run_dir}")

    def save_plots_as_jpg(self):
        """
        Save plot figures as JPG files. Creates directories based on the configuration hash and saves each plot.

        Parameters:
        - None

        Returns:
        - None
        """
        run_dir = os.path.join(self.base_dir, self.current_datetime)
        os.makedirs(run_dir, exist_ok=True)

        for plot_name, plot_figure in self.plots.items():
            plot_path = os.path.join(run_dir, f"{plot_name}.jpg")
            plot_figure.savefig(plot_path)
            plt.close(plot_figure)

        log.info(f"Plots saved to {run_dir}")

    def flatten_dict(self, dictionary, parent_key="", sep="_"):
        """
        Flatten a nested dictionary to create a single-level dictionary with concatenated keys.

        Parameters:
        - dictionary (dict): The dictionary to flatten.
        - parent_key (str, optional): A base key to prepend to each key in the flattened dictionary.
        - sep (str, optional): Separator to use between concatenated keys.

        Returns:
        - dict: A flattened dictionary.
        """
        items = []
        for i, a in dictionary.items():
            new_key = f"{parent_key}{sep}{i}" if parent_key else i
            if isinstance(a, dict):
                items.extend(self.flatten_dict(a, new_key, sep=sep).items())
            else:
                items.append((new_key, a))
        return dict(items)

    def config_hash(self, config):
        """
        Generate a hash for a given configuration. This hash is used as a unique identifier for directories and files.

        Parameters:
        - config (dict): Configuration dictionary to hash.

        Returns:
        - str: The generated MD5 hash as a hexadecimal string.
        """
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()

    def make_json_serializable(self, data):
        """
        Convert data into a format that can be serialized into JSON. This includes converting numpy arrays, pandas Index, Series and DataFrame objects to lists.

        Parameters:
        - data: The data to be converted into a JSON-serializable format.

        Returns:
        - The data in a JSON-serializable format.
        """
        if isinstance(data, np.ndarray) or isinstance(data, pd.Index):
            return data.tolist()
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict("records")
        elif isinstance(data, dict):
            return {
                key: self.make_json_serializable(value) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self.make_json_serializable(item) for item in data]
        else:
            return data
