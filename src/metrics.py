import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize


class ThresholdOptimizer:
    def __init__(self, min_score, max_score):
        self.min_score = min_score
        self.max_score = max_score
        self.coef = None

    def _loss(self, coef, y_pred, y_true):

        preds_discrete = np.clip(
            np.digitize(y_pred, np.sort(coef)) + self.min_score,
            self.min_score,
            self.max_score,
        )

        return -cohen_kappa_score(y_true, preds_discrete, weights="quadratic")

    def fit(self, y_pred, y_true):

        init_coef = np.arange(self.min_score + 0.5, self.max_score + 0.5)
        res = minimize(
            self._loss, init_coef, args=(y_pred, y_true), method="Nelder-Mead"
        )
        self.coef = np.sort(res.x)
        return self.coef

    def predict(self, y_pred):
        return np.clip(
            np.digitize(y_pred, self.coef) + self.min_score,
            self.min_score,
            self.max_score,
        )
