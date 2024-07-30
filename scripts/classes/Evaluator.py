import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import DATA_DIR
from utils.logger import log
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold


from pprint import pprint


class Evaluator:
    """
    This class is intended to evaluate different ML models on a given dataset.

    Methods:
    --------
    - metrics_dict()
    - test_models()
    - get_results_dict()
    - calculate_differences()
    - plot differences()
    - plot_metrics()
    - plot_confusion_matrices()

    Example:
    --------
    # set up evaluator to test models and plot metrics
    evaluator = Evaluator(tudd_fitted_models, X_test, y_test)
    results = evaluator.test_models()

    evaluator.plot_metrics()
    evaluator.plot_confusion_matrices()

    print(results)
    """

    def __init__(
        self,
        model_dict,
        classification: bool,
        target: str,
        run_dir=None,
        X_test=None,
        y_test=None,
        YScaler=None,
        full_cv=False,
    ):
        """
        Intializes the class.
        Parameters:
        model_dict = dictionary of fitted models
        classification = boolian specification if a classification task is evluated
        X_test = None or data frame of feature values
        y_test = None or list of target values
        """

        self.model_dict = model_dict
        self.classification = classification
        self.target = target
        self.run_dir = run_dir
        self.X_test = X_test
        self.y_test = y_test
        self.YScaler = YScaler
        self.differences = {}
        self.results_dict = {}
        log.debug(f"Scaler loaded as {type(self.YScaler)}")
        self.full_cv = full_cv

    def metrics_dict(self, y_true, y_proba, y_pred):
        """
        Function that returns a dict of common evaluation metrics
        """
        # Check if y_pred is in a valid format
        if y_pred is None or np.isnan(y_pred).any():
            print(y_pred)
            raise ValueError("y_pred contains invalid values or is None.")

        # For classification, also check y_proba
        if self.classification:
            if y_proba is None or np.isnan(y_proba).any() or y_proba.shape[1] != 2:
                print(y_proba)
                raise ValueError(
                    "y_proba contains invalid values, is None, or does not have the expected shape."
                )

        if self.classification:
            confu = confusion_matrix(y_true, y_pred)
            auroc = roc_auc_score(y_true, y_proba[:, 1])
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            auprc = auc(recall, precision)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc_curve_values = roc_curve(y_true, y_proba[:, 1])

            scores = {
                "AUROC": auroc,
                "AUPRC": auprc,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Confusion Matrix": confu,
            }
        else:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            scores = {
                "MAE": mae,
                "RMSE": rmse,
            }

        return scores

    def test_models(self):
        """
        Function which tests the models on the given data.
        Returns a dict with common evaluation metrics.
        """
        if self.model_dict is None or len(self.model_dict) == 0:
            raise Exception("No models provided for evaluation.")

        # create a dict with each model as a specific key
        self.results_dict = {model_name: None for model_name in self.model_dict.keys()}

        for model_name, model_info in self.model_dict.items():
            model = model_info["best_model"]
            self.results_dict[model_name] = self.eva_model(model)

            # get CV models to evaluate the stability of the model
            cv_submodels = model_info["cv_submodels"]
            cv_metrics = self.eva_cv_models(cv_submodels)
            print(f"CV metrics for {model_name}: {cv_metrics}")
            metrics_mean_std = {}
            for metric in cv_metrics[0].keys():
                m_values = [m[metric] for m in cv_metrics]
                metrics_mean_std[metric] = {
                    "mean": np.mean(m_values),
                    "std": np.std(m_values),
                }

            self.results_dict[model_name]["cv_metrics"] = metrics_mean_std
            self.results_dict[model_name]["params"] = model_info["best_params"]

        return self.results_dict

    def eva_model(self, model, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        try:
            if self.classification:
                y_proba = model.predict_proba(X_test)
                y_pred = model.predict(X_test)
                y_test = y_test
            else:
                y_proba = None
                # rescale the predictions and original values!
                # reshape to fit in inverse_transform
                y_pred_scaled = model.predict(X_test).reshape(-1, 1)
                if self.YScaler:
                    y_pred = self.YScaler.inverse_transform(y_pred_scaled)
                    y_test_scaled = y_test.to_numpy().reshape(-1, 1)
                    y_test = self.YScaler.inverse_transform(y_test_scaled)
                    print(f"TYPE AFTER INVERSE TRANSFORM: {type(y_test)}")
                    print(y_test)

            # set result dict
            return self.metrics_dict(y_test, y_proba, y_pred)
        except Exception as e:
            log.error(f"An error occurred during model evaluation: {e}")
            # Handle the error or return a default value
            return None

    def eva_cv_models(self, cv_models, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            X_test = self.X_test
            y_test = self.y_test

        cv_folds = len(cv_models)
        if self.classification:
            # create the same folds as for the training (see Trainer)
            Kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=1337)
        else:
            Kf = KFold(n_splits=cv_folds, shuffle=True, random_state=1337)
        cv_results = []

        for i, (train_idx, test_idx) in enumerate(Kf.split(self.X_test, self.y_test)):
            # this might look odd, but it is the correct way to sample fold from the test set only
            if self.full_cv:  # if full_cv is set to True
                print("use full cv for testing")
                X_test = self.X_test.iloc[test_idx]
                y_test = self.y_test.iloc[test_idx]
            else:

                X_test = self.X_test.iloc[train_idx]
                y_test = self.y_test.iloc[train_idx]

            cv_results.append(self.eva_model(cv_models[i], X_test, y_test))
        return cv_results

    def get_result_dict(self):
        return self.results_dict

    def plot_metrics(self):
        """
        Function to plot model performance metrics as bar plot.
        """

        labels = list(self.results_dict.keys())
        n_models = len(labels)
        if self.classification:
            metrics = ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1-Score"]
        else:
            metrics = ["MAE", "RMSE"]

        y = np.arange(len(metrics))  # the label locations
        height = 0.8  # the total height of all bars in a group

        fig, ax = plt.subplots(figsize=(8, 12))
        for i, label in enumerate(labels):
            bar_pos = (
                y - height / 2 + (height / n_models) * (i + 0.5)
            )  # compute bar position
            rects = ax.barh(
                bar_pos,
                [self.results_dict[label][m] for m in metrics],
                height / n_models,
                label=label,
            )
            bar_labels = [f"{round(self.results_dict[label][m], 3)}" for m in metrics]
            ax.bar_label(rects, labels=bar_labels, padding=3)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel("Scores")
        ax.set_title(f"Scores by different metrics and models. Target: {self.target}")
        ax.set_yticks(y)
        ax.set_yticklabels(metrics)
        ax.legend()

        fig.tight_layout()

        plt.show()
        return fig

    def plot_confusion_matrices(self):
        """
        Function to plot confusion matrices as heatmaps.
        """
        labels = list(self.results_dict.keys())
        n_models = len(labels)

        n_cols = 3  # number of columns in the subplot grid
        n_rows = int(np.ceil(n_models / n_cols))  # compute number of rows needed

        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(16, 6 * n_rows)
        )  # adjust size based on number of rows
        axs = axs.flatten()  # get a one-dimensional array

        for i, label in enumerate(labels):
            confusion_matrix = self.results_dict[label]["Confusion Matrix"]
            sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=axs[i])
            axs[i].set_title(f"Confusion Matrix for {label}")
            axs[i].set_xlabel("Predicted")
            axs[i].set_ylabel("Actual")

        # remove unused subplots
        for i in range(n_models, n_rows * n_cols):
            fig.delaxes(axs[i])

        plt.tight_layout()

        plt.show()

        return fig
