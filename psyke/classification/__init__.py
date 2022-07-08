from psyke import Predictor
from sklearn.base import ClassifierMixin


class BaseClassifier(Predictor, ClassifierMixin):
    def train(self, X, y):
        return RuntimeError("Untrainable classifier")
