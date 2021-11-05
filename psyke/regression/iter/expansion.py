import math


class Expansion:

    def __init__(self, cube, feature: str, direction: str, distance: float = math.nan):
        self.cube = cube
        self.feature = feature
        self.direction = direction
        self.distance = distance

    def get(self) -> tuple:
        return self.cube.get(self.feature)
