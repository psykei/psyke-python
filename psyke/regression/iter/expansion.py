class Expansion:

    def __init__(self, cube, feature, direction, distance):
        self.cube = cube
        self.feature = feature
        self.direction = direction
        self.distance = distance

    def get(self):
        self.cube.get(self.feature)
