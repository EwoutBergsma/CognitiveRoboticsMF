import numpy as np
from skgarden import MondrianForestClassifier


class MondrianForestClassifierWithALStrategy(MondrianForestClassifier):
    @staticmethod
    def calculate_confidence(probabilities):
        """
        :param probabilities:
        :return confidence:
        """
        sorted_proba = sorted(probabilities, reverse=True)
        return 1 - sorted_proba[1] / sorted_proba[0]

    @staticmethod
    def random_sampling(X_train, y_train, num_of_dataset):
        rand_selected_rows = np.random.choice(X_train.shape[0], num_of_dataset, replace=False)
        return X_train[rand_selected_rows, :], y_train[rand_selected_rows], rand_selected_rows

    def fit_using_al_strategy_thres(self, X, Y, classes=None, inital_dataset_size=300, threshold=0.5):
        classes = np.array(range(51)) if classes is None else classes  # default classes are 0-50

        # first, get the initial data set
        X_initial, Y_initial, inital_idxs = self.random_sampling(X, Y, inital_dataset_size)
        # Partial fit on the initial data
        self.partial_fit(X_initial, Y_initial, classes)

        # Retrieve the yet unused data samples
        remaining_idxs = [idx for idx in range(X.shape[0]) if idx not in inital_idxs]

        # Retrieve the probability distributions for the remaining training samples
        probabilities = self.predict_proba(X[remaining_idxs])
        # Filter the training samples based on the confidence of the mf
        selected_idxs = [idx for probs, idx in zip(probabilities, remaining_idxs) if
                         self.calculate_confidence(probs) < threshold]
        # Fit again on the samples the mf is unsure about
        self.partial_fit(X[selected_idxs], Y[selected_idxs])

        return len(selected_idxs)
