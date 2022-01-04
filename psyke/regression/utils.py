import math

Dimension = tuple[float, float]
Dimensions = dict[str, Dimension]


class Expansion:

    def __init__(self, cube, feature: str, direction: str, distance: float = math.nan):
        self.cube = cube
        self.feature = feature
        self.direction = direction
        self.distance = distance

    def __getitem__(self, index: int) -> float:
        return self.cube[self.feature][index]


class Limit:

    def __init__(self, feature: str, direction: str):
        self.feature = feature
        self.direction = direction

    def __eq__(self, other):
        return (self.feature == other.feature) and (self.direction == other.direction)

    def __hash__(self):
        return hash(self.feature + self.direction)


class MinUpdate:

    def __init__(self, name, value):
        self.name = name
        self.value = value


class ZippedDimension:

    def __init__(self, name: str, this_dimension: Dimension, other_dimension: Dimension):
        self.name = name
        self.this_dimension = this_dimension
        self.other_dimension = other_dimension
        