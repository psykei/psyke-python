from typing import Callable, Any


class SortedList(list):

    def __init__(self, comparator: Callable[[Any, Any], int]):
        super().__init__()
        self.comparator = comparator

    def add(self, item) -> None:
        if len(self) == 0:
            self.insert(0, item)
        else:
            starting_len = len(self)
            for index, element in enumerate(self):
                if self.comparator(element, item) > 0:
                    self.insert(index, item)
                    break
            if len(self) == starting_len:
                self.append(item)

    def add_all(self, other) -> None:
        for item in other:
            self.add(item)
