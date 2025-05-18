import copy
from typing import Union, Any

from psyke.extraction.cart import FairTreeClassifier, FairTreeRegressor, LeafSequence, LeafConstraints
from psyke.extraction.cart.CartPredictor import CartPredictor
from psyke.schema import LessThan, GreaterThan, SchemaException, Value


class FairTreePredictor(CartPredictor):
    """
    A wrapper for fair decision and regression trees of psyke.
    """

    def __init__(self, predictor: Union[FairTreeClassifier, FairTreeRegressor] = FairTreeClassifier(),
                 discretization=None, normalization=None):
        super().__init__(predictor, discretization, normalization)

    def __iter__(self) -> LeafSequence:
        leaves = [node for node in self.recurse(self._predictor.root, {})]
        return (leaf for leaf in leaves)

    @staticmethod
    def merge_constraints(constraints: LeafConstraints, constraint: Value, feature: str):
        if feature in constraints:
            try:
                constraints[feature][-1] *= constraint
            except SchemaException:
                constraints[feature].append(constraint)
        else:
            constraints[feature] = [constraint]
        return constraints

    def recurse(self, node, constraints) -> Union[LeafSequence, tuple[LeafConstraints, Any]]:
        if node.is_leaf_node():
            return constraints, node.value

        feature = node.feature
        threshold = node.threshold if self.normalization is None else \
            (node.threshold * self.normalization[feature][1] + self.normalization[feature][0])

        left = self.recurse(node.left, self.merge_constraints(copy.deepcopy(constraints), LessThan(threshold), feature))
        right = self.recurse(node.right, self.merge_constraints(copy.deepcopy(constraints),
                                                                GreaterThan(threshold), feature))
        return (left if isinstance(left, list) else [left]) + (right if isinstance(right, list) else [right])

    @property
    def predictor(self) -> Union[FairTreeClassifier, FairTreeRegressor]:
        return self._predictor

    @property
    def n_leaves(self) -> int:
        return self._predictor.n_leaves

    @predictor.setter
    def predictor(self, predictor: Union[FairTreeClassifier, FairTreeRegressor]):
        self._predictor = predictor
