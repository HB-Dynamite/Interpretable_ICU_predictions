import sys
import os
import warnings
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, Ridge
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from interpret import show
from pygam import LogisticGAM, LinearGAM
from pygam import terms, s, f
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
)
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import torch
from scipy.stats.mstats import winsorize


from typing import Dict
from dataclasses import dataclass, field
import shap


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

    def __init__(
        self, dataset, model_dict, random_state=1, save_plots=False, run_dir=None
    ):
        self.dataset = dataset
        self.random_state = random_state
        self.save_plots = save_plots
        self.run_dir = run_dir
        self.plot_data = PlotData()
        # might be benficial to save the trainer object here.
        self.model_dict = model_dict
        # TODO: add a get x original to Dataset would make this cleaner!
        # we might not want to soley access the the train data for plotting
        self.X_original = dataset.df
        self.create_directories()

    def create_directories(self):
        """
        Creates directories to store plots correctly.
        """
        if not os.path.isdir("plots"):
            os.mkdir("plots")
        if not os.path.isdir(f"plots/{self.dataset.name}"):
            os.mkdir(f"plots/{self.dataset.name}")
        if not os.path.isdir("plot_data"):
            os.mkdir("plot_data")
        if not os.path.isdir(f"plot_data/{self.dataset.name}"):
            os.mkdir(f"plot_data/{self.dataset.name}")

    def winsorize_col(self, col, upper_limit, lower_limit):
        """
        Applies winsorization to specific column given the limits.
        """
        return winsorize(col, limits=[upper_limit, lower_limit])

    def map_y_range(self, a, b, Y):
        (a1, a2), (b1, b2) = a, b
        return [b1 + ((y - a1) * (b2 - b1) / (a2 - a1)) for y in Y]

    # this function needs to be adapted to naming process of the scalerImputerObj
    def get_original_feature_name(self, feature_name: str):
        """
        Returns the original feature name
        """
        return feature_name.split("__")[1]

    def plot_model(self, model_name: str, rescale=False):
        """
        Creates the shape plots for each feature of a given model

        Parameters:
        - model_name (str): Name of the model to plot. Can be 'EBM', 'LR', 'PYGAM', 'XGB' or 'IGANN'
        - resacel (bool): Indicating whether to rescale values for improved visualization.
        """
        if rescale == True:
            self.scale_back_shape = True
        else:
            self.scale_back_shape = False

        if model_name == "XGB" or model_name == "MLP":
            if rescale == True:
                self.rescale_shap = True
            else:
                self.rescale_shap = False

        model = self.model_dict[model_name]
        model_method = {
            "EBM": self.EBM,
            "LR": self.LR,
            "PYGAM": self.PYGAM,
            "XGB": self.shap_scatter,
            "IGANN": self.IGANN,
            "MLP": self.shap_scatter,
            "LinReg": self.LR
            # TODO add more models
        }
        # run plot method
        model_method[model_name](model)

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

    def make_plot_data(self, X, Y, feature_name, model_name):
        """
        Prepare and save plot data.

        Parameters:
        - X: The feature values for which plot data is generated.
        - Y: The corresponding model's output values.
        - feature_name (str): The name of the feature being analyzed.
        - model_name (str): The name of the machine learning model associated with the data.
        """

        # TODO: rescale y for regression cases

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

        # cast to list for json export
        X = X.tolist()
        Y = Y.tolist()

        plot_data = {
            "dataset": self.dataset.name,
            "model": model_name,
            "feature": feature_name,
            "X": X,
            "Y": Y,
        }
        with open(
            "plot_data/"
            + self.dataset.name
            + "/"
            + model_name
            + "_"
            + feature_name
            + ".json",
            "w",
        ) as f:
            json.dump(plot_data, f)

    def make_plot(
        self,
        X,
        Y,
        feature_name,
        model_name,
        # scale_back=False, # we might want to use this again
        distplot=False,  # there are a few bugs in this
        titel=True,
        x_label=True,
        y_label=True,
        ex_distplot=False,
        save_plot_data=True,
    ):
        """
        Generate and save a the shape plot to visualize the relationship between a feature's values and their effect on a model's output.

        Parameters:
        - X (numpy.ndarray): The feature values to be plotted.
        - Y (numpy.ndarray): The model's output values corresponding to the feature values.
        - feature_name (str): The name of the feature being analyzed.
        - model_name (str): The name of the machine learning model for labeling and saving the plot.
        - distplot (bool): A flag indicating whether to include a distribution plot for the feature values.
        - title (bool): A flag indicating whether to include a title in the feature effect plot.
        - x_label (bool): A flag indicating whether to include an x-axis label in the plot.
        - y_label (bool): A flag indicating whether to include a y-axis label in the plot.
        - ex_distplot (bool): A flag indicating whether to include an extended distribution plot for the feature values.
        - save_plot_data (bool): A flag indicating whether to save plot data.
        """

        X = np.array(X)

        if (feature_name in self.dataset.num_cols) and (self.scale_back_shape == True):
            X = self.dataset.ScalerEncoder.inverse_transform_feature(feature_name, X)
        else:
            print("Feature values are standardized!")

        if (self.dataset.classification == False) and (self.scale_back_shape == True):
            Y = self.dataset.ScalerEncoder.inverse_transform_y(Y)

        if save_plot_data:
            self.make_plot_data(X, Y, feature_name, model_name)

        if distplot:
            fig, (ax1, ax2) = plt.subplots(
                nrows=2, gridspec_kw={"height_ratios": [0.8, 0.2]}
            )
            if feature_name in self.dataset.numerical_cols:
                bins_values, _, _ = ax2.hist(
                    self.X_original[feature_name], bins=10, color="grey"
                )
            else:
                bins_values, _, _ = ax2.hist(X[feature_name], bins=10, color="grey")
            ax2.set_xlabel("Distribution")
            ax2.set_xticks([])
            ax2.set_yticks([0, max(bins_values)])
        else:
            fig, ax1 = plt.subplots(nrows=1)

        fig.set_size_inches(5, 4)
        fig.set_dpi(100)

        if model_name != "EBM":
            ax1.plot(X, Y, color="black", alpha=1)
        else:
            ax1.step(X, Y, where="post", color="black")

        if titel:
            ax1.set_title(f"Feature:{feature_name}")
        if x_label:
            ax1.set_xlabel(f"Feature value")
        if y_label:
            ax1.set_ylabel("Feature effect on model output")
        fig.tight_layout()
        if self.save_plots:
            plot_filename = f"{model_name}_shape_{feature_name}.png"
            plot_path = os.path.join(self.run_dir, plot_filename)
            plt.savefig(plot_path)
        # plt.savefig(f"plots/{self.dataset.name}/{model_name}_shape_{feature_name}.png")
        plt.show()
        plt.close(fig)

        if ex_distplot:
            self.create_dist_plot(X, feature_name, self.dataset.name)

    def create_dist_plot(self, X, feature_name, dataset_name):
        """
        Generate and save a histogram-based distribution plot for a feature in a dataset.

        Parameters:
        - X (pd.DataFrame): The dataset containing the feature to plot.
        - feature_name (str): The name of the feature for which the distribution plot is generated.
        - dataset_name (str): The name of the dataset for labeling and saving the plot.
        """

        fig, ax1 = plt.subplots(nrows=1)
        fig.set_size_inches(4, 1)
        fig.set_dpi(100)

        if feature_name in self.dataset.numerical_cols:
            bins_values, _, _ = ax1.hist(
                self.X_original[feature_name], bins=10, color="grey"
            )
        else:
            bins_values, _, _ = ax1.hist(X[feature_name], bins=10, color="grey")
            ax1.set_xticks([])
            ax1.set_yticks([0, max(bins_values)])

        if self.save_plots:
            plot_filename = f"Distribution_shape_{feature_name}.pdf"
            plot_path = os.path.join(self.run_dir, plot_filename)
            plt.savefig(plot_path)
        # fig.savefig(f"plots/{dataset_name}/Distribution_shape_{feature_name}.pdf")
        fig.show()

    def make_one_hot_plot(
        self,
        class_zero,
        class_one,
        feature_name,
        model_name,
        distplot=False,
        title=True,
        x_label=True,
        y_label=True,
        ex_distplot=False,
    ):
        """
        Generate and save bar plots to visualize the effect of a one-hot encoded feature on a model's output for two discrete categories.

        Parameters:
        - self: The instance of a class containing the dataset and utility functions.
        - class_zero: The model's output for the first category of the feature.
        - class_one: The model's output for the second category of the feature.
        - feature_name (str): The name of the one-hot encoded feature for which the bar plot is generated.
        - model_name (str): The name of the machine learning model for labeling and saving plots.
        - distplot (bool): A flag indicating whether to include a distribution plot for the feature.
        - title (bool): A flag indicating whether to include a title in the bar plot.
        - x_label (bool): A flag indicating whether to include an x-axis label in the bar plot.
        - y_label (bool): A flag indicating whether to include a y-axis label in the bar plot.
        - ex_distplot (bool): A flag indicating whether to include an extended distribution plot for the feature.
        """

        if distplot:
            fig, (ax1, ax2) = plt.subplots(
                nrows=2, gridspec_kw={"height_ratios": [0.8, 0.2]}
            )
            if x_label:
                ax2.set_xlabel("Distribution")

            # If data is boolean, convert to integer
            if self.X_original[feature_name].dtype == np.bool:
                histogram_data = self.X_original[feature_name].astype(int)
            # If data is categorical, convert to category codes
            elif self.X_original[feature_name].dtype.name == "category":
                histogram_data = self.X_original[feature_name].cat.codes
            else:
                histogram_data = self.X_original[feature_name]

            # Add an assertion to ensure that the data type is as expected
            assert np.issubdtype(
                histogram_data.dtype, np.number
            ), f"Expected int or float, got {histogram_data.dtype}"
            bins_values, _, _ = ax2.hist(
                histogram_data, bins=2, rwidth=0.9, color="grey"
            )
            ax2.set_xticks([])
            ax2.set_yticks([0, max(bins_values)])
        else:
            fig, ax1 = plt.subplots(nrows=1)

        fig.set_size_inches(5, 4)
        fig.set_dpi(100)

        # if self.dataset.task == "regression" and scale_back:
        #     class_one = self.inverse_transform_y(class_one)
        #     class_zero = self.inverse_transform_y(class_zero)

        category_0 = self.X_original[feature_name].values.categories[0]
        category_1 = self.X_original[feature_name].values.categories[1]
        categories = [category_0, category_1]
        ax1.bar(
            [0, 1],
            [class_zero, class_one],
            color="gray",
            tick_label=[f"{categories[0]}", f"{categories[1]} "],
        )

        if title:
            plt.title(f'Feature:{feature_name.split("_")[0]}')
        if y_label:
            ax1.set_ylabel("Feature effect on model output")

        fig.tight_layout()

        if self.save_plots:
            plot_filename = (
                f'{model_name}_onehot_{str(feature_name).replace("?", "")}.png'
            )
            plot_path = os.path.join(self.run_dir, plot_filename)
            plt.savefig(plot_path)
        # plt.savefig(f'plots/{self.dataset.name}/{model_name}_onehot_{str(feature_name).replace("?", "")}.png')
        # plt.show()

        if ex_distplot:
            self.create_one_hot_dist_plot(feature_name, self.dataset.name)

    def create_one_hot_dist_plot(self, feature_name):
        """
        Generate and save a histogram-based distribution plot for a one-hot encoded feature with two discrete categories.

        Parameters:
        - feature_name (str): The name of the one-hot encoded feature for which the distribution plot is generated.
        """

        original_feature_name = self.get_original_feature_name(feature_name)
        fig, ax1 = plt.subplots(nrows=1)
        fig.set_size_inches(5, 1)
        fig.set_dpi(100)
        bins_values, _, _ = ax1.hist(
            self.X_original[original_feature_name], bins=2, rwidth=0.9, color="grey"
        )
        ax1.set_xticks([])
        ax1.set_yticks([0, max(bins_values)])

        if self.save_plots:
            plot_filename = f"{self}_Distribution_onehot_{feature_name}.png"
            plot_path = os.path.join(self.run_dir, plot_filename)
            plt.savefig(plot_path)
        # fig.savefig(f"plots/{self}/Distribution_onehot_{feature_name}.png")
        fig.show()

    def make_one_hot_multi_plot(
        self, model_name, distribution_plot=False, ex_distribution_plot=False
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

            # if self.task == "regression":
            #     y_list_not_given_class = self.y_scaler.inverse_transform(
            #         np.array(y_list_not_given_class).reshape((-1, 1))
            #     ).squeeze()
            #     y_list_given_class = self.y_scaler.inverse_transform(
            #         np.array(y_list_given_class).reshape((-1, 1))
            #     ).squeeze()

            x_list = list(self.plot_data.entries[feature_name].keys())

            if distribution_plot:
                fig, (ax1, ax2) = plt.subplots(
                    nrows=2, gridspec_kw={"height_ratios": [0.8, 0.2]}
                )
                bins_values, _, _ = ax2.hist(
                    self.X_original[feature_name],
                    bins=len(x_list),
                    rwidth=0.8,
                    color="grey",
                )
                ax2.set_xlabel("Distribution")
                ax2.set_xticks([])
                ax2.set_yticks([0, max(bins_values)])
            else:
                fig, ax1 = plt.subplots()

            fig.set_size_inches(5, 4)
            fig.set_dpi(100)

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

            ax1.bar(position_list, y_plot_value, color="gray", width=0.8)

            ax1.set_ylabel("Feature effect on model output")
            ax1.set_title(f"Feature:{feature_name}")
            ax1.set_xticks(position_list)
            ax1.set_xticklabels(x_list, rotation=90)
            fig.tight_layout()

            plt.show()

            if ex_distribution_plot:
                self.make_one_hot_multi_distribution_plot(
                    feature_name, self.dataset.name, x_list
                )

    def make_one_hot_multi_distribution_plot(
        self, feature_name: str, dataset_name: str, x_list: list
    ):
        """
        Generate and save a histogram-based distribution plot for a specific feature with multiple discrete categories represented as one-hot encoded values.

        Parameters:
        - feature_name (str): The name of the feature for which the distribution plot is generated.
        - dataset_name (str): The name of the dataset for labeling and saving the plot.
        - x_list (list): A list of unique category labels corresponding to the one-hot encoded values.
        """

        fig, ax1 = plt.subplots(nrows=1)
        bins_values, _, _ = ax1.hist(
            self.X_original[feature_name], bins=len(x_list), rwidth=0.8, color="grey"
        )

        fig.set_size_inches(5, 1)
        fig.set_dpi(100)
        ax1.set_xticks([])
        ax1.set_yticks([0, max(bins_values)])

        if self.save_plots:
            plot_filename = f"Distribution_onehot_{feature_name}.png"
            plot_path = os.path.join(self.run_dir, plot_filename)
            plt.savefig(plot_path, bbox_inches="tight")
        # fig.savefig(f"plots/{dataset_name}/Distribution_onehot_{feature_name}.png")
        fig.show()

    def EBM(self, EBM):
        """
        Create and Plot Shape Plots of a given EBM model.
        """
        ebm_global = EBM.explain_global()

        for i, _ in enumerate(ebm_global.data()["names"]):
            data_names = ebm_global.data()
            feature_name = data_names["names"][i]
            shape_data = ebm_global.data(i)
            if shape_data["type"] == "interaction":
                pass
            elif shape_data["type"] == "univariate":
                original_feature_name = self.get_original_feature_name(feature_name)

                if original_feature_name in self.dataset.cat_cols:
                    if self.X_original[original_feature_name].value_counts().size == 2:
                        self.make_one_hot_plot(
                            shape_data["scores"][0],
                            shape_data["scores"][1],
                            original_feature_name,
                            "EBM",
                        )
                    else:
                        column_name = original_feature_name
                        class_name = feature_name.split("_")[-1]
                        not_given_class_score = shape_data["scores"][0]
                        given_class_score = shape_data["scores"][1]
                        self.plot_data.add_entry(
                            column_name,
                            class_name,
                            not_given_class_score,
                            given_class_score,
                        )
                else:
                    X_values = shape_data["names"].copy()
                    Y_values = shape_data["scores"].copy()
                    Y_values = np.r_[Y_values, Y_values[np.newaxis, -1]]

                    self.make_plot(X_values, Y_values, original_feature_name, "EBM")

            else:
                raise ValueError(f"Unknown type {shape_data['type']}")
        self.make_one_hot_multi_plot("EBM")
        self.plot_data.reset()

    def PYGAM(self, PYGAM):
        """
        Create and Plot Shape Plots of a given PYGAM model.
        """

        model_name = "PYGAM"
        # TODO: Integrate terms as parameters on model initialization

        for i, term in enumerate(PYGAM.terms):
            if term.isintercept:
                continue
            X_values = PYGAM.generate_X_grid(term=i)
            pdep, confi = PYGAM.partial_dependence(term=i, X=X_values, width=0.95)
            # this looks odd
            original_feature_name = self.get_original_feature_name(
                self.X[self.X.columns[i]].name
            )
            if original_feature_name in self.dataset.categorical_cols:
                if self.X_original[original_feature_name].value_counts().size == 2:
                    self.make_one_hot_plot(
                        pdep[0], pdep[-1], original_feature_name, model_name
                    )
                else:
                    column_name = original_feature_name
                    class_name = self.X[self.X.columns[i]].name.split("_")[-1]
                    not_given_class_score = pdep[0]
                    given_class_score = pdep[-1]

                    self.plot_data.add_entry(
                        column_name,
                        class_name,
                        not_given_class_score,
                        given_class_score,
                    )
            else:
                self.make_plot(
                    X_values[:, i].squeeze(), pdep, original_feature_name, model_name
                )

        self.make_one_hot_multi_plot(model_name)
        self.plot_data.reset()

    def LR(self, LR):
        """
        Create and Plot Shape Plots of a given LR model.
        """
        model_name = "LR"
        # if self.task == "regression":
        #     LR = Ridge()
        # else:
        #     LR = LogisticRegression()
        # LR.fit(self.X, self.y)
        # check if there is mor than on coeff (simple one feature regression)
        # check if there is more than one coeff (simple one feature regression)

        coefficients = LR.coef_.squeeze()
        # TODO: Feature names aus dem Datensatz ziehen
        feature_names = list(LR.feature_names_in_)
        X_train = self.dataset.split_dict["imputed_mean"]["X_train"]

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
            if original_feature_name in self.dataset.cat_cols:
                if self.X_original[original_feature_name].value_counts().size > 2:
                    column_name = original_feature_name
                    class_name = feature_name.split("_")[-1]
                    class_score = word_to_coef[feature_name]
                    self.plot_data.add_entry(column_name, class_name, 0, class_score)
                else:
                    self.make_one_hot_plot(
                        0, word_to_coef[feature_name], original_feature_name, model_name
                    )  # zero as value for class one correct?
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
                self.make_plot(inp, outp, original_feature_name, model_name)
        self.make_one_hot_multi_plot(model_name)
        self.plot_data.reset()

    def IGANN(self, IGANN):
        """
        Create and Plot Shape Plots of a given IGANN model.
        """
        model_name = "IGANN"
        # igann = IGANN(self.task, n_estimators=1000, device="cpu")
        # igann.fit(self.X, np.array(self.y))

        shape_data = IGANN.get_shape_functions_as_dict()
        for feature in shape_data:
            # print(feature["name"])
            original_feature_name = self.get_original_feature_name(feature["name"])
            # print(original_feature_name)
            if original_feature_name in self.dataset.cat_cols:
                if self.X_original[original_feature_name].value_counts().size > 2:
                    # print(feature)
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
                    self.make_one_hot_plot(
                        feature["y"][0],
                        feature["y"][1],
                        original_feature_name,
                        model_name,
                    )
            else:
                self.make_plot(
                    feature["x"],
                    feature["y"],
                    original_feature_name,
                    model_name,
                )

        self.make_one_hot_multi_plot(model_name)
        self.plot_data.reset()

    def shap_scatter(self, model):
        """
        Generate scatter plots to visualize SHAP (SHapley Additive exPlanations) values for each feature in relation to model predictions.

        Parameters:
        - model: A trained machine learning model capable of making predictions.
        """

        X_train, X_test, y_train, y_test = self.dataset.get_splits("imputed_mean")

        # if self.rescale_shap == True:
        #     X_train = self.dataset.ScalerEncoder.inverse_transform_X(X=X_train)
        #     X_test = self.dataset.ScalerEncoder.inverse_transform_X(X=X_test)

        if self.dataset.classification:
            input = self.log_odds_wrapper(model)
        else:
            input = model.predict

        explainer = shap.explainers.Exact(input, X_train)

        # TODO:we need the train set. maybe wee add it to the dataset
        shap_values = explainer(X_test)

        # For each feature
        for i in range(len(self.dataset.cols)):
            # print(shap_values[:, i])
            feature_name = self.dataset.cols[i]
            # print(f"Feature: {feature_name}")

            # check format of shap vlaues
            shap.plots.scatter(shap_values[:, i])
            shap.plots.scatter(shap_values[:, i], color=shap_values)

            # Debugging print
            # print(f"X_test shape: {X_test.shape}")
            # print(f"Shap values shape: {shap_values.shape}")

            # make a standard partial dependence plot with a single SHAP value overlaid
            shap.partial_dependence_plot(
                i,
                input,
                X_train,
            )

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
