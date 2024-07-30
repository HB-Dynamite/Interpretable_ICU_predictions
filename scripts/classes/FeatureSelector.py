from sklearn.base import BaseEstimator, TransformerMixin
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# ... import other feature selection strategies as needed


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Class for performing feature selection on a dataset.

    Methods:
    --------
    - fit(X, y=None): Fit the feature selector to the data.
    - transform(X): Transform the data by selecting the chosen features or returning it unmodified.
    - fit_transform(X, y=None): Fit to data, then transform it.
    """

    def __init__(
        self,
        X_train,
        y_train,
        selector_params=None,
        num_cols=None,
        cat_cols=None,
    ):
        """
        Initialize the feature selector.

        Parameters:
        - X_train: Training data used for feature selection
        - strategy: The feature selection strategy (e.g., 'KBest', 'ModelBased', 'None', etc.)
        - selector_params: Dictionary of parameters for the chosen feature selection strategy
        """
        self.X_train = X_train
        self.y_train = y_train
        self.selector_params = selector_params if selector_params is not None else {}
        self.strategy = selector_params["strategy"]
        self.selector = None
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.selected_features_indices = None
        self._initialize_selector()

    def _initialize_selector(self):
        """
        Internal method to initialize the feature selector based on the chosen strategy.
        """
        if self.strategy == "KBest":
            score_func = self.selector_params.get("score_func", f_classif)
            k = self.selector_params.get("k", 10)
            self.selector = SelectKBest(score_func=score_func, k=k)
        elif self.strategy == "SFFS":
            estimator = self.selector_params.get(
                "estimator",
                LogisticRegression(
                    C=0.01,
                    penalty="l1",
                    class_weight=None,
                    solver="saga",
                    l1_ratio=None,
                    max_iter=1000,
                    n_jobs=-1,
                ),
            )
            k_features = self.selector_params.get(
                "k_features", 11
            )  # Set to select top 60 features
            forward = self.selector_params.get("forward", True)
            floating = self.selector_params.get("floating", True)
            scoring = self.selector_params.get("scoring", "roc_auc")
            cv = self.selector_params.get("cv", 5)
            n_jobs = self.selector_params.get("n_jobs", -1)
            self.selector = SFS(
                estimator=estimator,
                k_features=k_features,
                forward=forward,
                floating=floating,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
            )
        elif self.strategy is None:
            # In this case, no selector is needed
            self.selector = None
        # Additional strategies can be initialized here

    def fit(self, X=None, y=None):
        """
        Fit the selector to the data.
        """
        X = self.X_train if X is None else X
        y = self.y_train if y is None else y

        if self.strategy is not None and self.selector is None:
            raise RuntimeError("Feature selector is not initialized")
        if self.strategy is not None:
            self.selector.fit(X, y)
            if hasattr(self.selector, "get_support"):
                self.selected_features_indices = self.selector.get_support(indices=True)
            elif hasattr(self.selector, "k_feature_idx_"):
                self.selected_features_indices = list(self.selector.k_feature_idx_)
        return self

    def transform(self, X):
        """
        Transform the data by selecting the chosen features or return it unmodified.
        """
        if self.strategy is None:
            # If strategy is None, return the data as is
            return X
        if self.selector is None:
            raise RuntimeError("You must fit the selector before transforming data")
        # set the num and cat cols
        X = self.selector.transform(X)
        print(self.selected_features_indices)
        print(X)
        return X

    def get_selected_columns(self):
        """
        Get the names of the selected columns.
        """
        if self.strategy is None:
            return None
        if self.selector is None:
            raise RuntimeError(
                "You must fit the selector before getting the selected columns"
            )

        return self.X_train.columns[self.selected_features_indices]

    def get_num_cols(self):
        """
        Get the names of the selected columns.
        """
        if self.strategy is None:
            return None
        if self.selector is None:
            raise RuntimeError(
                "You must fit the selector before getting the selected columns"
            )
        return self.X_train.columns

    def get_cat_cols(self):
        """
        Get the names of the selected columns.
        """
        print("###################################")
        print(self.cat_cols)
        print(self.num_cols)
        print(self.X_train.columns)
        print("###################################")
        if self.strategy is None:
            return None
        if self.selector is None:
            raise RuntimeError(
                "You must fit the selector before getting the selected columns"
            )
        return self.X_train.columns
