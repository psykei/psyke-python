from typing import Callable, Any


class SortedList(list):

    def __init__(self, comparator: Callable[[Any, Any], int]):
        super().__init__()
        self.comparator = comparator

    def add(self, item) -> None:
        if len(self) == 0:
            self.insert(0, item)
        else:
            for index, element in enumerate(self):
                if self.comparator(element, item):
                    self.insert(index, item)
                    break

    def add_all(self, other) -> None:
        for item in other:
            self.add(item)
