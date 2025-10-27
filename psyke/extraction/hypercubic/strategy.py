from __future__ import annotations

from functools import reduce
from collections.abc import Iterable


class Strategy:
    def __init__(self, partitions = None):
        self._partitions = partitions
        self._no_features = []

    def get(self, feature: str) -> int:
        raise NotImplementedError

    def make_fair(self, features: Iterable[str]):
        self._no_features = features

    def partition_number(self, features: Iterable[str]) -> int:
        return reduce(lambda x, y: x * y, map(self.get, features), 1)

    def equals(self, strategy, features: Iterable[str]) -> bool:
        eq = True
        for f in features:
            eq = eq and self.get(f) == strategy.get(f)
        return eq

    def __str__(self):
        return self._partitions

    def __repr__(self):
        return self.__str__()


class FixedStrategy(Strategy):
    def __init__(self, partitions: int = 2):
        super().__init__(partitions)

    def get(self, feature: str) -> int:
        return 1 if feature in self._no_features else self._partitions

    def __str__(self):
        return "Fixed ({})".format(super().__str__())


class AdaptiveStrategy(Strategy):
    def __init__(self, features: Iterable[(str, float)], partitions: Iterable[tuple[float, float]] | None = None):
        super().__init__(partitions if partitions is not None else [(0.33, 2), (0.67, 3)])
        self.features = features

    def get(self, feature: str) -> int:
        if feature in self._no_features:
            return 1
        importance = next(filter(lambda t: t[0] == feature, self.features))[1]
        n = 1
        for (imp, part) in self._partitions:
            if importance >= imp:
                n = part
            else:
                break
        return n

    def __str__(self):
        return "Adaptive ({})".format(super().__str__())
