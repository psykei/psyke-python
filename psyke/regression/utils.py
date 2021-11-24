import math


class Expansion:

    def __init__(self, cube, feature: str, direction: str, distance: float = math.nan):
        self.cube = cube
        self.feature = feature
        self.direction = direction
        self.distance = distance

    def get(self) -> tuple:
        return self.cube.get(self.feature)


class Limit:

    def __init__(self, feature: str, direction: str):
        self.feature = feature
        self.direction = direction

    def __eq__(self, other):
        return (self.feature == other.feature) & (self.direction == other.direction)

    def __hash__(self):
        return hash(self.feature + self.direction)


class MinUpdate:

    def __init__(self, name, value):
        self.name = name
        self.value = value


class ZippedDimension:

    def __init__(self, dimension: str, this_cube: (float, float), other_cube: (float, float)):
        self.dimension = dimension
        self.this_cube = this_cube
        self.other_cube = other_cube
        