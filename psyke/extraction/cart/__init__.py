from abc import ABC

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from psyke.extraction import PedagogicalExtractor
from psyke import get_default_random_seed
from psyke.extraction.cart.FairTree import FairTreeClassifier, FairTreeRegressor
from psyke.schema import DiscreteFeature, Value
from tuprolog.theory import Theory
from typing import Iterable, Any
import pandas as pd


TREE_SEED = get_default_random_seed()

LeafConstraints = dict[str, list[Value]]
LeafSequence = Iterable[tuple[LeafConstraints, Any]]


class Cart(PedagogicalExtractor, ABC):

    def __init__(self, predictor, max_depth: int = 3, max_leaves: int = None, max_features=None,
                 discretization: Iterable[DiscreteFeature] = None,
                 normalization=None, simplify: bool = True):
        from psyke.extraction.cart.CartPredictor import CartPredictor

        super().__init__(predictor, discretization, normalization)
        self.is_fair = None
        self._cart_predictor = CartPredictor(discretization=discretization, normalization=normalization)
        self.depth = max_depth
        self.leaves = max_leaves
        self.max_features = max_features
        self._simplify = simplify

    def _extract(self, data: pd.DataFrame) -> Theory:
        from psyke.extraction.cart.FairTreePredictor import FairTreePredictor

        if self.is_fair:
            self._cart_predictor = FairTreePredictor(discretization=self.discretization,
                                                     normalization=self.normalization)
            fair_tree = FairTreeClassifier if isinstance(data.iloc[0, -1], str) else FairTreeRegressor
            self._cart_predictor.predictor = fair_tree(max_depth=self.depth, max_leaves=self.leaves,
                                                       protected_attr=self.is_fair)
        else:
            tree = DecisionTreeClassifier if isinstance(data.iloc[0, -1], str) else DecisionTreeRegressor
            self._cart_predictor.predictor = tree(random_state=TREE_SEED, max_depth=self.depth,
                                                  max_leaf_nodes=self.leaves, max_features=self.max_features)
        self._cart_predictor.predictor.fit(data.iloc[:, :-1], data.iloc[:, -1])
        return self._cart_predictor.create_theory(data, self._simplify)

    def make_fair(self, features: Iterable[str]):
        self.is_fair = features

    def _predict(self, dataframe: pd.DataFrame) -> Iterable:
        return self._cart_predictor.predict(dataframe)

    def predict_why(self, data: dict[str, float], verbose=True):
        prediction = None
        conditions = {}
        if self.normalization is not None:
            data = {k: v * self.normalization[k][1] + self.normalization[k][0] if k in self.normalization else v
                    for k, v in data.items()}
        for conditions, prediction in self._cart_predictor:
            if all(all(interval.is_in(data[variable]) for interval in intervals)
                   for variable, intervals in conditions.items()):
                break
        return prediction, conditions

    @property
    def n_rules(self) -> int:
        return self._cart_predictor.n_leaves
