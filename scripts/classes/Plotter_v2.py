import sys
import os
import warnings
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go


from interpret import show

from matplotlib import pyplot as plt


from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import torch
from scipy.stats.mstats import winsorize
from typing import Dict
from dataclasses import dataclass, field
import shap
from pprint import pprint


class Plotter:
    """
    A Plotter class used to produce shape plots (feature effects) and SHAP plots.

    Methods:
    --------
    - create_directories()
    - winsorize_col(col, upper_limit, lower_limit)
    - map_y_range(a,b,Y)
    - get_original_feature_name(feature_name:str)
    - plot_model(model_name: str, rescale = False)
    - plot_shape(model_name: str, rescale = False)
    - make_plot_data(X,Y, feature_name, model_name)
    - make_plot(X,Y, feature_name, model_name, displot=False, title=True, x_label=True,y_label=True, ex_distplot=False, save_plot_data=True)
    - create_dist_plot(X, feature_name, dataset_name)
    - make_one_hot_plot(class_zero, class_one, feature_name, model_name, distplot=False, title=True, x_label=True, y_label=True, ex_distplot=False)
    - create_one_hot_dist_plot(feature_name)
    - make_one_hot_multi_plot(model_name, distribution_plot=False, ex_distribution_plot=False)
    - make_one_hot_multi_distribution_plot(feature_name: str, dataset_name: str, x_list: list)
    - EBM(EBM)
    - PYGAM(PYGAM)
    - LR(LR)
    - IGANN(IGANN)
    - shap_scatter(model)
    - log_odds_wrapper(model)
    - shap_pd(model)

    Examples:
    ---------
    plotter = Plotter(dataset_object, fitted_models)

    plotter.plot_model("MLP", rescale=False)

    plotter.plot_shap("LR", rescale=True)
    """

    def __init__(self, dataset, params: {}):
        self.dataset = dataset
        self.save_plot_data = params["save_plot_data"]
        self.save_plots = params["save_plots"]
        self.dist_plots = params["dist_plots"]
        self.show_plots = params["show_plots"]
        self.path = params["path"]
        self.plot_data = PlotData()

        # this could be more intuive to handle outside the ploter
        self.scale_back = True
        # self.model_dict = model_dict
        # TODO: add a get x original to Dataset would make this cleaner!
        # we might not want to soley access the the train data for plotting
        self.X_original = dataset.df
        self.create_directories()

    def create_directories(self):
        """
        Creates directories, including the base path if necessary, to store plots correctly.
        """
        if self.path is None:
            self.path = "."

        # Automatically create the base path and any required intermediate directories
        os.makedirs(self.path, exist_ok=True)

        # Paths for plots and plot_data directories
        self.plots_path = os.path.join(self.path, "plots")
        self.plot_data_path = os.path.join(self.path, "plot_data")
        self.dataset_plots_path = os.path.join(self.plots_path, self.dataset.name)
        self.dataset_plot_data_path = os.path.join(
            self.plot_data_path, self.dataset.name
        )
        # print(self.plots_path)
        # print(self.dataset_plots_path)

        # Create the directories if they do not exist
        os.makedirs(self.plots_path, exist_ok=True)
        os.makedirs(self.plot_data_path, exist_ok=True)
        os.makedirs(self.dataset_plots_path, exist_ok=True)
        os.makedirs(self.dataset_plot_data_path, exist_ok=True)

    def get_original_feature_name(self, feature_name: str):
        """
        Returns the original feature name
        """
        return feature_name.split("__")[1]

    def find_column_containing_substring(self, df, substring):
        """
        Finds the name of the first column in the DataFrame that contains the given substring.
        """
        matching_columns = df.filter(like=substring).columns
        n_matching = len(matching_columns)
        if n_matching == 1:
            return matching_columns[0]  # return value
        elif n_matching > 1:
            Warning(f"found multiple matching columns")
            return matching_columns  # return list of values
        else:
            Warning(f"no math found in dataset")
            return None  # Or raise an error, depending on how you want to handle no matches

    def get_feature_type(self, feature_name):
        """
        Determine the type of a feature: 'numeric', 'categorical'.

        Args:
        feature_name (str): The name of the feature to determine the type for.

        Returns:
        str: The type of the feature ('numeric', 'categorical').
        """
        if feature_name in self.dataset.cat_cols:
            return "categorical"
        else:
            # If not in categorical columns, consider it numeric
            return "numeric"

    def plot_model(self, model_name: str, model):
        """
        Creates the shape plots for each feature of a given model

        Parameters:
        - model_name (str): Name of the model to plot. Can be 'EBM', 'LR', 'PYGAM', 'XGB' or 'IGANN'
        - resacel (bool): Indicating whether to rescale values for improved visualization.
        """
        model_method = {
            "LinReg": self.LR,
            "LR": self.LR,
            "EBM": self.EBM,
            "PYGAM": self.PYGAM,
            "IGANN": self.IGANN,
            "XGB": self.shap_scatter,
            "MLP": self.shap_scatter,
            "RF": self.shap_scatter,
            # TODO add more models
        }
        # run plot method
        method = model_method[model_name]
        if model_name in ["XGB", "RF"]:
            method(model, model_name)
        else:
            method(model)

    def plot_shap(self, model_name: str, rescale=False):
        """
        Generate and display SHAP plots to interpret and visualize feature effects on predictions.

        Parameters:
        - model_name (str): The name of the machine learning model for which SHAP plots are generated.
        - rescale (bool): A flag indicating whether to rescale SHAP values for improved visualization.
        """

        model = self.model_dict[model_name]

        if rescale == True:
            self.rescale_shap = True
        else:
            self.rescale_shap = False

        self.shap_scatter(model)

    def make_shape_plot_data(self, X, Y, feature_name, model_name):
        """
        Prepare and plot_data.

        Parameters:
        - X: The feature values for which plot data is generated.
        - Y: The corresponding model's output values.
        - feature_name (str): The name of the feature being analyzed.
        - model_name (str): The name of the machine learning model associated with the data.
        """

        X = np.array(X)

        if (feature_name in self.dataset.num_cols) and (self.scale_back is True):

            X = self.dataset.ScalerEncoder.inverse_transform_feature(feature_name, X)
        else:
            print("Feature values are standardized!")

        if (self.dataset.classification is False) and (self.scale_back is True):
            Y = self.dataset.ScalerEncoder.inverse_transform_y(Y)

        if isinstance(Y, list):
            Y = np.array(Y)
        elif torch.is_tensor(Y):
            Y = Y.cpu().numpy()

        if isinstance(X, list):
            X = np.array(X)
        elif torch.is_tensor(X):
            X = X.cpu().numpy()

        # Ensure one dimension
        if len(Y.shape) > 1:
            Y = Y.flatten()

        if len(X.shape) > 1:
            X = X.flatten()

        # cast to list for better use
        X = X.tolist()
        Y = Y.tolist()

        plot_data = {
            "dataset": self.dataset.name,
            "model": model_name,
            "feature": feature_name,
            "X": X,
            "Y": Y,
        }
        return plot_data

    def save_shape_plot_data(self, plot_data, path=None):
        if path is None:
            path = ".."
        with open(
            self.dataset_plot_data_path
            + "/"
            + plot_data["model"]
            + "_"
            + plot_data["feature"]
            + ".json",
            "w",
        ) as f:
            json.dump(plot_data, f)

    def make_shape_plot(
        self,
        plot_data,
        # scale_back=False, # we might want to use this again
        titel=True,
        x_label=True,
        y_label=True,
    ):
        X = plot_data["X"]
        Y = plot_data["Y"]
        feature_name = plot_data["feature"]
        model_name = plot_data["model"]

        # this may also needs a rework in order to function correctly with the make_plot_data
        if self.dist_plots:
            # create a fig with subplots
            fig, ax1 = plt.subplots(
                2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [2, 1]}
            )
            main_ax = ax1[0]  # Use the first subplot for the main plot
            dist_ax = ax1[1]  # Use the second subplot for the distribution plot
        else:
            fig, main_ax = plt.subplots(figsize=(5, 4))

        fig.set_dpi(100)

        # this works also for EBM since preprocessing the data points in EBM function!
        main_ax.plot(X, Y, color="black", alpha=1)

        if titel:
            main_ax.set_title(f"Feature:{feature_name}")
        if x_label:
            main_ax.set_xlabel(f"Feature value")
        if y_label:
            main_ax.set_ylabel("Feature effect on model output")

        if self.dist_plots:
            self.create_distribution_plot(dist_ax, feature_name)
            # Synchronize the x-axis limits
            combined_xlim = [
                min(X),
                max(X),
            ]
            # print(combined_xlim)
            main_ax.set_xlim(combined_xlim)
            dist_ax.set_xlim(combined_xlim)
        fig.tight_layout()

        if self.save_plots:
            plot_filename = f"{model_name}_shape_{feature_name}.png"
            plot_path = os.path.join(self.dataset_plots_path, plot_filename)
            plt.savefig(plot_path)

        if self.save_plot_data:
            self.save_shape_plot_data(plot_data)
        if self.show_plots:
            plt.show()
        plt.close(fig)

    def create_distribution_plot(
        self,
        ax,
        feature_name,
    ):
        """
        Generate and save a distribution plot for any given feature.

        Parameters:
        - feature_name (str): The name of the feature for which the distribution plot is generated.
        """
        # Determine if the feature is numeric or categorical
        feature_type = self.get_feature_type(feature_name)
        # Numeric feature
        if feature_type == "numeric":
            ##### this code will show the distribtion of the final training data.
            train_data, _, _, _ = self.dataset.get_splits()
            training_feature_name = self.find_column_containing_substring(
                train_data, feature_name
            )
            print(training_feature_name)
            data = train_data[training_feature_name]
            data = self.dataset.ScalerEncoder.inverse_transform_feature(
                training_feature_name, data
            )

            ax.hist(data, bins=30, color="grey")
            # ax.set_title(f"Distribution of {feature_name}")
            # ax.set_xlabel(feature_name)
            ax.set_ylabel("Frequency")

        # Categorical feature
        elif feature_type is "categorical":
            train_data, _, _, _ = self.dataset.get_splits()  # selctes the final split
            training_feature_names = self.find_column_containing_substring(
                train_data, feature_name
            )
            counts = train_data[
                training_feature_names
            ].sum()  # this works because one hot encoding
            ax.bar(counts.index, counts.values, color="grey")
            # ax.set_title(f"Distribution of {feature_name}")
            # ax.set_xlabel(feature_name)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)  # Rotate labels if necessary

    def make_shape_cat_plot_data(
        self,
        classes: list,
        values: list,
        feature_name,
        model_name,
    ):
        if (self.dataset.classification is False) and (self.scale_back is True):
            values = self.dataset.ScalerEncoder.inverse_transform_y(values)

        plot_data = {
            "dataset": self.dataset.name,
            "model": model_name,
            "feature": feature_name,
            "classes": classes,
            "values": values,
        }
        return plot_data

    def make_shape_cat_plot(self, plot_data, title=True, y_label=True):
        model_name = plot_data["model"]
        feature = plot_data["feature"]
        classes = plot_data["classes"]
        values = plot_data["values"]

        if self.dist_plots:
            fig, ax1 = plt.subplots(
                2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [2, 1]}
            )
            main_ax = ax1[0]
            dist_ax = ax1[1]
        else:
            fig, main_ax = plt.subplots(1, 1, figsize=(5, 4))

        main_ax.bar(
            classes,
            values,
            color="gray",
        )

        if self.dist_plots:
            self.create_distribution_plot(dist_ax, feature)

        if title:
            plt.title(f"Feature:{feature}")
        if y_label:
            main_ax.set_ylabel("Feature effect on model output")

        fig.tight_layout()

        if self.save_plots:
            plot_filename = f'{model_name}_onehot_{str(feature).replace("?", "")}.png'
            plot_path = os.path.join(self.dataset_plots_path, plot_filename)
            plt.savefig(plot_path)
        if self.save_plot_data:
            self.save_shape_plot_data(plot_data)
        if self.show_plots:
            plt.show()
        plt.close(fig)

    def make_one_hot_multi_plot(
        self,
        model_name,
    ):
        """
        Generate and save multi-bar plots to visualize the effect of one-hot encoded features on a machine learning model's output for multiple feature categories.

        Parameters:
        - model_name (str): The name of the machine learning model for labeling and saving plots.
        - distribution_plot (bool): A flag indicating whether to include a distribution plot for each feature category.
        - ex_distribution_plot (bool): A flag indicating whether to include an extended distribution plot for each feature category.
        """
        for feature_name in self.plot_data.entries:
            position_list = np.arange(len(self.plot_data.entries[feature_name]))
            y_values = list(self.plot_data.entries[feature_name].values())
            y_list_not_given_class = [
                list(dict_element.values())[0] for dict_element in y_values
            ]
            y_list_given_class = [
                list(dict_element.values())[1] for dict_element in y_values
            ]

            y_plot_value = []
            for i in range(len(y_values)):
                y_not_given_values = sum(
                    [
                        value
                        for index, value in enumerate(y_list_not_given_class)
                        if index != i
                    ]
                )
                y_plot_value.append((y_list_given_class[i] + y_not_given_values).item())

            position_list = position_list.tolist()

            plot_data = self.make_shape_cat_plot_data(
                position_list, y_plot_value, feature_name, model_name
            )

            self.make_shape_cat_plot(plot_data)

    def EBM(self, EBM):
        """
        Create and Plot Shape Plots of a given EBM model.
        """
        ebm_global = EBM.explain_global()

        for i, _ in enumerate(ebm_global.data()["names"]):
            data_names = ebm_global.data()
            feature_name = data_names["names"][i]
            # pprint(data_names)
            # print(feature_name)
            shape_data = ebm_global.data(i)
            if shape_data["type"] == "interaction":
                pass  # no Interactions available yet
            elif shape_data["type"] == "univariate":
                original_feature_name = self.get_original_feature_name(feature_name)
                feature_type = self.get_feature_type(original_feature_name)

                if feature_type is "categorical":
                    # this assumes one hot encoding
                    column_name = original_feature_name
                    class_name = feature_name.split("_")[-1]
                    # print(class_name)
                    not_given_class_score = shape_data["scores"][0]
                    given_class_score = shape_data["scores"][1]
                    self.plot_data.add_entry(
                        column_name,
                        class_name,
                        not_given_class_score,
                        given_class_score,
                    )

                elif feature_type is "numeric":
                    X_values = shape_data["names"].copy()
                    Y_values = shape_data["scores"].copy()
                    Y_values = np.r_[Y_values, Y_values[np.newaxis, -1]]

                    # this code is a to ensure that the plots are correctly displayes even without plt.step
                    # keep in mind that the vales are reverses transformed by standard scaler!
                    epsilon = 1e-10

                    # New lists to hold expanded values
                    new_X_values = []
                    new_Y_values = []

                    # Initialize with the first value
                    new_X_values.append(X_values[0])
                    new_Y_values.append(Y_values[0])

                    for i in range(1, len(X_values)):
                        # Insert x - epsilon for every x and get the previous y value
                        new_X_values.append(X_values[i] - epsilon)
                        new_Y_values.append(Y_values[i - 1])

                        # Append the original x and y values
                        new_X_values.append(X_values[i])
                        new_Y_values.append(Y_values[i])

                    # Convert lists back to numpy arrays
                    X_values = np.array(new_X_values)
                    Y_values = np.array(new_Y_values)

                    # self.make_plot(X_values, Y_values, original_feature_name, "EBM")
                    plot_data = self.make_shape_plot_data(
                        X_values, Y_values, original_feature_name, "EBM"
                    )
                    self.make_shape_plot(plot_data)

            else:
                raise ValueError(f"Unknown type {shape_data['type']}")
        self.make_one_hot_multi_plot("EBM")
        self.plot_data.reset()

        # # local explaination:
        # _, X_test, _, _ = self.dataset.get_splits()
        # isinstance_to_explain = X_test.iloc[0:1]
        # local_explaiantion = EBM.explain_local(isinstance_to_explain)
        # show(local_explaiantion)

    def PYGAM(self, PYGAM):
        """
        Create and Plot Shape Plots of a given PYGAM model.
        """

        # acess the underliying gam model
        PYGAM = PYGAM.gam

        model_name = "PYGAM"
        # TODO: Integrate terms as parameters on model initialization

        X, _, _, _ = self.dataset.get_splits()

        for i, term in enumerate(PYGAM.terms):
            if term.isintercept:
                continue
            X_values = PYGAM.generate_X_grid(term=i)
            pdep, confi = PYGAM.partial_dependence(term=i, X=X_values, width=0.95)
            # this looks odd
            original_feature_name = self.get_original_feature_name(X[X.columns[i]].name)
            feature_type = self.get_feature_type(original_feature_name)
            if feature_type is "categorical":
                column_name = original_feature_name
                class_name = X[X.columns[i]].name.split("_")[-1]
                not_given_class_score = pdep[0]
                given_class_score = pdep[-1]

                self.plot_data.add_entry(
                    column_name,
                    class_name,
                    not_given_class_score,
                    given_class_score,
                )
            else:
                plot_data = self.make_shape_plot_data(
                    X_values[:, i].squeeze(), pdep, original_feature_name, model_name
                )
                self.make_shape_plot(plot_data)

        self.make_one_hot_multi_plot(model_name)
        self.plot_data.reset()

    def LR(self, LR):
        """
        Create and Plot Shape Plots of a given LR model.
        """
        model_name = "LR"

        coefficients = LR.coef_.squeeze()
        # TODO: Feature names aus dem Datensatz ziehen
        X_train, _, _, _ = self.dataset.get_splits()
        feature_names = X_train.columns

        if np.size(coefficients) == 1:
            word_to_coef = {
                feature_names[0]: coefficients.item()
            }  # convert numpy array to scalar
        else:
            word_to_coef = dict(zip(feature_names, coefficients))

        dict(sorted(word_to_coef.items(), key=lambda item: item[1]))
        word_to_coef_df = pd.DataFrame.from_dict(word_to_coef, orient="index")

        for i, feature_name in enumerate(feature_names):
            original_feature_name = self.get_original_feature_name(feature_name)
            feature_type = self.get_feature_type(original_feature_name)
            if feature_type is "categorical":
                column_name = original_feature_name
                class_name = feature_name.split("_")[-1]
                class_score = word_to_coef[feature_name]
                self.plot_data.add_entry(column_name, class_name, 0, class_score)
            else:
                inp = torch.linspace(
                    X_train[feature_name].min(),
                    X_train[feature_name].max(),
                    1000,
                )
                outp = word_to_coef[feature_name] * inp
                # convert back to list before plooting.
                inp = inp.cpu().numpy().tolist()
                outp = outp.cpu().numpy().tolist()
                plot_data = self.make_shape_plot_data(
                    inp, outp, original_feature_name, model_name
                )
                self.make_shape_plot(plot_data)
        self.make_one_hot_multi_plot(model_name)
        self.plot_data.reset()

    def IGANN(self, IGANN):
        """
        Create and Plot Shape Plots of a given IGANN model.
        """
        model_name = "IGANN"

        shape_data = IGANN.get_shape_functions_as_dict()
        for feature in shape_data:
            original_feature_name = self.get_original_feature_name(feature["name"])
            feature_type = self.get_feature_type(original_feature_name)
            if feature_type is "categorical":
                column_name = original_feature_name
                class_name = feature["name"].split("_")[-1]
                not_given_category_value = feature["y"][0]
                if len(feature["y"]) == 2:
                    given_category_value = feature["y"][1]
                elif len(feature["y"]) == 1:
                    given_category_value = 0
                else:
                    raise ValueError(
                        "Feature has neither than 2 nor 1 value. This should not be possible."
                    )
                self.plot_data.add_entry(
                    column_name,
                    class_name,
                    not_given_category_value,
                    given_category_value,
                )

            else:
                plot_data = self.make_shape_plot_data(
                    feature["x"],
                    feature["y"],
                    original_feature_name,
                    model_name,
                )
                self.make_shape_plot(plot_data)

        self.make_one_hot_multi_plot(model_name)
        self.plot_data.reset()

    def shap_scatter(self, model, model_name: str = ""):
        """
        Generate custom scatter plots to visualize SHAP values for each feature
        in relation to model predictions, similar to shap.plots.scatter.

        Parameters:
        - model: A trained machine learning model capable of making predictions.
        """

        X_train, X_test, y_train, y_test = self.dataset.get_splits()

        if self.dataset.classification:
            input_func = self.log_odds_wrapper(model)
        else:
            input_func = model.predict

        if model_name in [
            "XGB",
        ]:
            explainer = shap.TreeExplainer(model, X_train)
        else:
            explainer = shap.explainers.Exact(input_func, X_train)

        shap_values = explainer(X_train)

        # feature_names = self.dataset.cols
        feature_names = X_train.columns
        pprint(X_train.columns)

        # Process and plot SHAP values for each feature
        for i, feature_name in enumerate(feature_names):
            # Extract the SHAP values and corresponding feature values
            shap_vals = shap_values[:, i].values
            feature_vals = X_train[feature_name].values
            org_feature_name = self.get_original_feature_name(feature_name)
            SHAP_plot_data = self.make_shape_plot_data(
                X=feature_vals,
                Y=shap_vals,
                feature_name=org_feature_name,
                model_name="SHAP_" + model_name,
            )
            self.save_shape_plot_data(SHAP_plot_data)

            # Create the scatter plot
            plt.figure(figsize=(6, 4))
            plt.scatter(SHAP_plot_data["X"], SHAP_plot_data["Y"], alpha=0.1)
            plt.title(f"SHAP values vs. {org_feature_name}")
            plt.xlabel(f"{org_feature_name} Value")
            plt.ylabel("SHAP Value")
            plt.grid(False)
            plt.show()
            plt.close()

    def log_odds_wrapper(self, model):
        """
        This function creates a predict log ods function for a model.
        The model must have a predict proba function.
        """

        def log_odds(X):
            # get predictions for proba
            p = model.predict_proba(X)[:, 1]
            # p = model.predict(X)
            # convert proba to log_odds
            return np.log(p / (1 - p))
            # return p

        return log_odds

    def shap_pd(self, model):
        """
        Generate a partial dependence plot overlaid with SHAP (SHapley Additive exPlanations) values for each feature in a given machine learning model.

        Parameters:
        - model: A trained machine learning model capable of making predictions.
        """
        import shap

        # Create object that can calculate shap values
        # we use get split method here could be handeld better.
        explainer = shap.Explainer(
            model.predict, self.dataset.get_splits("imputed_mean")[0]
        )
        # TODO:we need the train set. maybe wee add it to the dataset
        shap_values = explainer(self.dataset.get_splits("imputed_mean")[0])

        # For each feature
        for i in range(len(self.dataset.cols)):
            feature_name = self.dataset.cols[i]
            print(feature_name)

        # make a standard partial dependence plot with a single SHAP value overlaid
        fig, ax = shap.partial_dependence_plot(
            i,
            model.predict,
            model_expected_value=True,
            feature_expected_value=True,
            show=False,
            ice=False,
            shap_values=shap_values,
        )


class PlotData:
    """
    The PlotData class provides a convenient way to organize and manage data related to feature scores for different classes.
    It allows users to add data entries representing feature scores for specific features and classes,
    as well as reset the data when needed.

    Attributes:
        entries (dict): A dictionary that stores data entries for feature scores.
            The keys are feature names, and the values are dictionaries with class names as keys and corresponding scores as values.

    Methods:
        __init__():
            Constructor method to initialize an instance of the PlotData class.

        add_entry(feature_name, class_name, not_given_class_score, given_class_score):
            Add an entry to the PlotData object.
        reset():
            Reset all data entries in the PlotData object.
    """

    def __init__(self):
        """
        Initialize a new PlotData object with an empty data dictionary.
        """
        self.entries = {}

    def add_entry(
        self, feature_name, class_name, not_given_class_score, given_class_score
    ):
        """
        Add an entry to the PlotData object.

        Parameters:
        - feature_name (str): The name of the feature.
        - class_name (str): The name of the class.
        - not_given_class_score (float): The score associated with the feature for the class when not given.
        - given_class_score (float): The score associated with the feature for the class when given.
        """
        if feature_name not in self.entries:
            self.entries[feature_name] = {}
        self.entries[feature_name][class_name] = {
            "not_given_class_score": not_given_class_score,
            "given_class_score": given_class_score,
        }

    def reset(self):
        """
        Reset all data entries in the PlotData object.
        """
        self.entries = {}
