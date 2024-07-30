import numpy as np
from pygam import LogisticGAM, LinearGAM, terms, s, f
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels


class PYGAMClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom wrapper for the PyGAM class to make it compatible with sklearn methods.

    Parameters
    ----------
    n_splines : int, default=10
        The number of splines to use in the logistic GAM.
    lam : float, default=0.6
        The lambda parameter to use in the logistic GAM.
    terms : list of PyGAM terms, default=None
        The terms to include in the model. If None, a spline term will be included for each feature.
    """

    def __init__(self, n_splines=10, lam=0.6, terms_string=""):
        self.n_splines = n_splines
        self.lam = lam
        self.terms_string = terms_string
        self.gam = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # Store the unique classes found in y
        self.classes_ = unique_labels(y)

        self._terms = self.create_terms_from_string(self.terms_string)

        self.gam = LogisticGAM(self._terms, n_splines=self.n_splines, lam=self.lam)
        self.gam.fit(X, y)
        return self

    def predict(self, X):
        # check_is_fitted(self)
        X = check_array(X)
        return self.gam.predict(X)

    def predict_proba(self, X):
        # check_is_fitted(self)
        X = check_array(X)
        # predict_mu gives the probability of positive class for LogisticGAM
        proba = self.gam.predict_mu(X)
        # Return an array with two columns: one for the probabilities of the negative class, and one for the positive class
        return np.column_stack([1 - proba, proba])

    @staticmethod
    def create_terms_from_string(feature_types_str):
        feature_types = feature_types_str.split(",")

        term_list = terms.TermList()
        for i, feature_type in enumerate(feature_types):
            if feature_type == "s":
                term_list += s(i)
            elif feature_type == "f":
                term_list += f(i)
            else:
                raise ValueError("Unsupported feature type")
        return term_list


class PyGAMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_splines=10, lam=0.6, terms=None):
        self.n_splines = n_splines
        self.lam = lam
        self.terms = terms
        self.gam = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.terms is None:
            self.terms = s(0)  # Default to spline term for the first feature
        self.gam = LinearGAM(terms=self.terms, n_splines=self.n_splines, lam=self.lam)

        self.gam.fit(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        return self.gam.predict(X)

    @staticmethod
    def create_terms_from_string(feature_types_str):
        feature_types = feature_types_str.split(",")

        term_list = terms.TermList()
        for i, feature_type in enumerate(feature_types):
            if feature_type == "s":
                term_list += s(i)
            elif feature_type == "f":
                term_list += f(i)
            else:
                raise ValueError("Unsupported feature type")
        return term_list
