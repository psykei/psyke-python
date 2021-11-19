from psyke.classification.trepan.node import Node


class Split:

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
