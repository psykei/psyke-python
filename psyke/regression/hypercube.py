from psyke.regression.iter.zipped_dimension import ZippedDimension


class HyperCube:

    def __init__(self):
        self.__dimension = {}
        self.__limits = ()
        self.__output = 0.0
        self.__epsilon = 1.0 / 1000

    def __eq__(self, other):
        all([(abs(dimension.this_cube[0] - dimension.other_cube[0]) < self.__epsilon)
             & (abs(dimension.this_cube[1] - dimension.other_cube[1]) < self.__epsilon)
             for dimension in self.__zip_dimensions(other)])

    # Just for debugging
    def __str__(self):
        text = ''
        for k, v in self.__dimension.items():
            text = text + k + ': ' + str(v[0]) + '; ' + str(v[1]) + '\n'
        return text

    def get(self, feature):
        if feature in self.__dimension.keys():
            return self.__dimension[feature]
        else:
            raise Exception('FeatureNotFoundException: ' + feature)

    def expand(self, expansion, hypercubes):
        feature, direction = expansion.feature, expansion.direction
        a, b = self.get(feature)
        self.__dimension[feature] = (expansion.get()[0], b) \
            if direction == '-' else (a, expansion.get()[1])
        other_cube = self.overlap_all(hypercubes)
        if other_cube is not None:
            self.__dimension[feature] = (other_cube[1], b) if direction == '-' else (other_cube[0], a)

    def overlap(self, hypercube):
        return any([not ((dimension.other_cube[0] >= dimension.this_cube[1]) |
                         (dimension.this_cube[0] >= dimension.other_cube[1]))
                    for dimension in self.__zip_dimensions(hypercube)])

    def overlap_all(self, hypercubes):
        for hypercube in hypercubes:
            if (self != hypercube) & self.overlap(hypercube):
                return hypercube
            else:
                return None

    def has_volume(self):
        all([dimension[1] - dimension[0] > self.__epsilon for dimension in self.__dimension.values()])

    def update_dimension(self, feature, values):
        self.__dimension[feature] = values

    def __zip_dimensions(self, hypercube):
        for dimension in self.__dimension.keys():
            yield ZippedDimension(dimension, self.get(dimension), hypercube.get(dimension))

    def equal_all(self, hypercubes):
        any([self == it for it in hypercubes])
