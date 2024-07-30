from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from igann import IGANN  # Import your custom IGANN class
from pprint import pprint
import numpy as np


class IGANNClassifier(IGANN, BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        params.pop("task", None)  # Remove 'task' if it exists in params
        super(IGANNClassifier, self).__init__(task="classification", **params)
        # Additional initialization if needed

    def fit(self, X, y):
        # Fit the model using the parent class's fit method
        super(IGANNClassifier, self).fit(X, y)

        # Set the classes_ attribute
        self.classes_ = np.unique(y)
        return self


class IGANNRegressor(IGANN, BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        params.pop("task", None)  # Remove 'task' if it exists in params
        super(IGANNRegressor, self).__init__(task="regression", **params)
        # Additional initialization if needed
