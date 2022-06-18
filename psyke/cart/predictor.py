from collections import Iterable
from typing import Union, Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from psyke.schema import Value, LessThan, GreaterThan

LeafConstraints = list[tuple[str, Value, bool]]
LeafSequence = Iterable[tuple[LeafConstraints, Any]]


class CartPredictor:
    """
    A wrapper for decision and regression trees of sklearn.
    """

    def __init__(self, predictor: Union[DecisionTreeClassifier, DecisionTreeRegressor]):
        self._predictor = predictor

    def __get_constraints(self, nodes: Iterable[(int, bool)]) -> LeafConstraints:
        return [(self._predictor.feature_names_in_[self._predictor.tree_.feature[i[0]]],
                 LessThan(self._predictor.tree_.threshold[i[0]]) if i[1] else
                 GreaterThan(self._predictor.tree_.threshold[i[0]]), i[1]) for i in nodes]

    def __get_leaves(self) -> Iterable[int]:
        left_orphan = [i for i, v in enumerate(self._predictor.tree_.children_left) if v == -1]
        right_orphan = [i for i, v in enumerate(self._predictor.tree_.children_right) if v == -1]
        return [v for v in left_orphan if v in left_orphan and v in right_orphan]

    def __get_prediction(self, node: int) -> Any:
        if hasattr(self._predictor, 'classes_'):
            return self._predictor.classes_[np.argmax(self._predictor.tree_.value[node])]
        else:
            return self._predictor.tree_.value[node]

    def __path(self, node: int, path=[]) -> Iterable[(int, bool)]:
        if node == 0:
            return path
        else:
            father_left = [(i, True) for i, v in enumerate(self._predictor.tree_.children_left) if v == node]
            father_right = [(i, False) for i, v in enumerate(self._predictor.tree_.children_right) if v == node]
            father: (int, bool) = (father_left + father_right)[0]
            return self.__path(father[0], [father] + path)

    def __iter__(self) -> LeafSequence:
        leaves = self.__get_leaves()
        return ((self.__get_constraints(self.__path(i)), self.__get_prediction(i)) for i in leaves)

    def predict(self, data) -> Iterable:
        return self._predictor.predict(data)

    @property
    def predictor(self) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        return self._predictor

    @property
    def n_leaves(self) -> int:
        return len(list(self.__get_leaves()))

    @predictor.setter
    def predictor(self, predictor: Union[DecisionTreeClassifier, DecisionTreeRegressor]):
        self._predictor = predictor
