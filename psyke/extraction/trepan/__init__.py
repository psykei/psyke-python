import numpy as np

from psyke.extraction.trepan.utils import Node, Split, SplitLogic
from psyke import Extractor, DiscreteFeature
from psyke.utils.logic import create_term, create_variable_list, create_head
from psyke.utils.sorted import SortedList
from tuprolog.core import Var, Struct, clause
from tuprolog.theory import MutableTheory, mutable_theory, Theory
from typing import Iterable, Union, Any
import pandas as pd


class Trepan(Extractor):

    def __init__(self, predictor, discretization: Iterable[DiscreteFeature], min_examples: int = 0, max_depth: int = 3,
                 split_logic: SplitLogic = SplitLogic.DEFAULT):
        super().__init__(predictor, discretization)
        self.min_examples = min_examples
        self.max_depth = max_depth
        self.split_logic = split_logic
        self.__root: Node

    @property
    def n_rules(self):
        return sum(1 for _ in self.__root)

    def __best_split(self, node: Node, names: Iterable[str]) -> Union[tuple[Node, Node], None]:
        if node.samples.shape[0] < self.min_examples:
            raise NotImplementedError()
        if node.n_classes == 1:
            return None
        splits = Trepan.__create_splits(node, names)
        return None if len(splits) == 0 or splits[0].children[0].depth > self.max_depth else splits[0].children

    def __compact(self):
        nodes = [self.__root]
        while len(nodes) > 0:
            node = nodes.pop()
            for item in self.__nodes_to_remove(node, nodes):
                node.children.remove(item)
                node.children += item.children

    def __create_body(self, variables: dict[str, Var], node: Node) -> Iterable[Struct]:
        result = []
        for constraint, value in node.constraints:
            feature: DiscreteFeature = [d for d in self.discretization if constraint in d.admissible_values][0]
            result.append(create_term(variables[feature.name], feature.admissible_values[constraint], value == 1.0))
        return result

    @staticmethod
    def __create_samples(node: Node, column: str, value: float) -> pd.DataFrame:
        return node.samples.loc[node.samples[column] == value]

    @staticmethod
    def __create_split(node: Node, column: str) -> Union[Split, None]:
        true_examples = Trepan.__create_samples(node, column, 1.0)
        false_examples = Trepan.__create_samples(node, column, 0.0)
        true_constrains = list(node.constraints) + [(column, 1.0)]
        false_constrains = list(node.constraints) + [(column, 0.0)]
        true_node = Node(true_examples, node.n_examples, true_constrains, depth=node.depth + 1)\
            if true_examples.shape[0] > 0 else None
        false_node = Node(false_examples, node.n_examples, false_constrains, depth=node.depth + 1)\
            if false_examples.shape[0] > 0 else None
        return None if true_node is None or false_node is None else Split(node, (true_node, false_node))

    @staticmethod
    def __create_splits(node: Node, names: Iterable[str]) -> SortedList[Split]:
        splits, constrains = Trepan.__init_splits(node)
        for column in names:
            if column not in constrains:
                split = Trepan.__create_split(node, column)
                if split is not None:
                    splits.add(split)
        return splits

    def __create_theory(self, name: str) -> MutableTheory:
        theory = mutable_theory()
        for node in self.__root:
            variables = create_variable_list(self.discretization)
            theory.assertZ(
                clause(
                    create_head(name, list(variables.values()), str(node.dominant)),
                    self.__create_body(variables, node)
                )
            )
        return theory

    def __init(self, dateset: pd.DataFrame) -> SortedList[Node]:
        self.__root = Node(dateset, dateset.shape[0])
        queue: SortedList[Node] = SortedList(lambda x, y: int(x.priority - y.priority))
        queue.add(self.__root)
        return queue

    @staticmethod
    def __init_splits(node: Node) -> tuple[SortedList[Split], Iterable[str]]:
        return SortedList(lambda x, y: int(x.priority - y.priority)),\
               set(constraint[0] for constraint in node.constraints)

    @staticmethod
    def __nodes_to_remove(node: Node, nodes: list[Node]) -> list[Node]:
        to_remove = []
        for child in node.children:
            if node.dominant == child.dominant and len(child.children) == 1:
                to_remove.append(child)
                nodes.append(node)
            else:
                nodes.append(child)
        return to_remove

    @staticmethod
    def __predict(x: pd.Series, node: Node, categories: Iterable) -> Any:
        for child in node.children:
            skip = False
            for constraint, value in child.constraints:
                if x[constraint] != value:
                    skip = True
                    continue
            if not skip:
                return Trepan.__predict(x, child, categories)
        return node.dominant  # Alternatively node.dominant index in categories

    def __optimize(self) -> None:
        n, nodes = 0, [self.__root]
        while len(nodes) > 0:
            n += Trepan.__remove_nodes(nodes)
        self.__compact() if n == 0 else self.__optimize()

    @staticmethod
    def __remove_nodes(nodes: list[Node]) -> int:
        node = nodes.pop()
        to_remove = [child for child in node.children if len(child.children) == 0 and node.dominant == child.dominant]
        for child in to_remove:
            node.children.remove(child)
        for child in node.children:
            if len(child.children) > 0:
                nodes.append(child)
        return len(to_remove)

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        dataframe.iloc[:, -1] = self.predictor.predict(dataframe.iloc[:, :-1])
        queue = self.__init(dataframe)
        while len(queue) > 0:
            node = queue.pop()
            if self.split_logic == SplitLogic.DEFAULT:
                best: Union[tuple[Node, Node], None] = self.__best_split(node, dataframe.columns[:-1])
                if best is None:
                    continue
            else:
                raise Exception('Illegal split logic')
            queue.add_all(best)
            node.children += list(best)
        self.__optimize()
        return self.__create_theory(dataframe.columns[-1])

    def predict(self, dataframe: pd.DataFrame) -> Iterable:
        return np.array(
            [Trepan.__predict(sample, self.__root, dataframe.columns[-1]) for _, sample in dataframe.iterrows()]
        )
