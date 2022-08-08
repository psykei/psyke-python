from __future__ import annotations
from itertools import chain
from typing import Iterable, Any
import pandas as pd


class Node:

    def __init__(self, samples: pd.DataFrame, n_examples: int, constraints: Iterable[tuple[str, float]] = None,
                 children: list[Node] = None, depth: int = 0):
        self.samples = samples
        self.n_examples = n_examples
        self.constraints = [] if constraints is None else constraints
        self.children = [] if children is None else children
        self.depth = depth

    def __str__(self):
        name = ''.join(('' if c[1] > 0 else '!') + c[0] + ', ' for c in self.constraints)
        return name[:-2] + ' = ' + str(self.dominant)

    @property
    def priority(self) -> float:
        return -(self.reach * (1 - self.fidelity))

    @property
    def fidelity(self) -> float:
        return 1.0 * self.correct / (self.samples.shape[0] if self.samples.shape[0] > 0 else 1)

    @property
    def reach(self) -> float:
        return 1.0 * self.samples.shape[0] / self.n_examples

    @property
    def correct(self) -> float:
        return sum(self.samples.iloc[:, -1] == self.dominant)

    @property
    def dominant(self) -> Any:
        return self.samples.iloc[:, -1].mode()[0] if self.samples.shape[0] > 0 else ''

    @property
    def n_classes(self) -> int:
        return len(set(self.samples.iloc[:, -1]))

    def __iter__(self) -> Iterable[Node]:
        for child in chain(*map(iter, self.children)):
            yield child
        yield self


class Split:

    # TODO: should be configurable by user
    PRIORITY_BONUS: int = 100
    PRIORITY_PENALTY: int = 200

    def __init__(self, parent: Node, children: tuple[Node, Node]):
        self.parent = parent
        self.children = children

    @property
    def priority(self) -> float:
        return self.__priority(self.parent)

    def __priority(self, parent: Node) -> float:
        true_node, false_node = self.children
        priority = - (true_node.fidelity + false_node.fidelity)
        for node in [true_node, false_node]:
            priority -= self.PRIORITY_BONUS if parent.n_classes > node.n_classes else 0
        priority += self.PRIORITY_PENALTY if true_node.dominant == false_node.dominant else 0
        return priority


class SplitLogic:

    DEFAULT = 1
