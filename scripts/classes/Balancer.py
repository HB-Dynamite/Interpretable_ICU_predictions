from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


class Balancer:
    def __init__(
        self, X_train, y_train, oversampling_rate=None, undersampling_rate=None
    ):
        """
        Initializes the Balancer with the training data and specified oversampling and undersampling rates.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.oversampling_rate = oversampling_rate
        self.undersampling_rate = undersampling_rate
        self._build_pipeline()

    def _build_pipeline(self):
        """
        Builds the balancing pipeline based on the specified oversampling and undersampling rates.
        """
        steps = []
        if self.oversampling_rate is not None:
            steps.append(
                (
                    "over",
                    SMOTE(sampling_strategy=self.oversampling_rate, random_state=1991),
                )
            )
        if self.undersampling_rate is not None:
            steps.append(
                (
                    "under",
                    RandomUnderSampler(
                        sampling_strategy=self.undersampling_rate, random_state=1991
                    ),
                )
            )
        self.balance_pipe = Pipeline(steps=steps) if steps else None

    def fit_resample(self, X=None, y=None):
        """
        Applies the balancing pipeline to the given features (X) and target (y), or uses the training data set during initialization if no data is provided.
        Returns the balanced features and target, or the original data if no balancing is required.
        """
        if X is None or y is None:
            X, y = self.X_train, self.y_train

        if self.balance_pipe is not None:
            X_balanced, y_balanced = self.balance_pipe.fit_resample(X, y)
            return X_balanced, y_balanced
        else:
            return X, y  # No balancing is applied, return the original data
