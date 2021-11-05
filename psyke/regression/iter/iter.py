from functools import reduce

from tuprolog.theory import *
from psyke.extractor import Extractor
from psyke.regression.hypercube import HyperCube
from psyke.regression.iter.expansion import Expansion
from psyke.regression.iter.minupdate import MinUpdate
from psyke.utils.logic_utils import *

DomainProperties = (list[MinUpdate], HyperCube)


class Iter(Extractor):

    def __init__(self, predictor, min_update, n_points, max_iterations, min_examples, threshold, fill_gaps):
        super().__init__(predictor)
        self.predictor = predictor
        self.discretization = []
        self.min_update = min_update
        self.n_points = n_points
        self.max_iterations = max_iterations
        self.min_examples = min_examples
        self.threshold = threshold
        self.fill_gaps = fill_gaps
        self.__fake_dataset = pd.DataFrame()
        self.__hypercubes = []

    def __calculate_min_updates(self, surrounding: HyperCube) -> list[MinUpdate]:
        return [MinUpdate(name, (interval[1] - interval[0]) * self.min_update) for (name, interval) in
                surrounding.dimensions.items()]

    @staticmethod
    def __find_closer_sample(dataset: pd.DataFrame, output: float) -> dict[str, tuple]:
        min_difference = min(abs(dataset.iloc[:, -1] - output))
        indices = [math.isclose(abs(value - output), min_difference) for value in dataset.iloc[:, -1]]
        close_samples = dataset.iloc[indices, :]
        close_sample = close_samples.iloc[0, :].to_dict()
        return close_sample

    def __generate_starting_points(self, dataset: pd.DataFrame) -> list[HyperCube]:
        min_output = min(dataset.iloc[:, -1])
        max_output = max(dataset.iloc[:, -1])
        points: list[float]
        if - math.inf < self.n_points <= 0:
            raise (Exception('InvalidAttributeValueException'))
        else:
            if self.n_points == 1:
                points = [(max_output - min_output) / 2]
            else:
                points = [(min_output + (max_output - min_output) / ((self.n_points - 1) * (index - 1))) for index in
                          range(1, self.n_points)]
        return [HyperCube.cube_from_point(Iter.__find_closer_sample(dataset, point)) for point in points]

    def __init_hypercubes(self, dataset, min_updates, surrounding) -> list[HyperCube]:
        while True:
            hypercubes = self.__generate_starting_points(dataset)
            for hypercube in hypercubes:
                hypercube.expand_all(min_updates, surrounding)
            self.n_points = self.n_points - 1
            if not HyperCube.check_overlap(hypercubes, hypercubes):
                break
        return hypercubes

    def __init(self, dataset: pd.DataFrame) -> (list[HyperCube], DomainProperties):
        self.__fake_dataset = dataset.iloc[:, :-1]
        surrounding = HyperCube.create_surrounding_cube(dataset)
        min_updates = self.__calculate_min_updates(surrounding)
        self.__hypercubes = self.__init_hypercubes(dataset, min_updates, surrounding)
        for hypercube in self.__hypercubes:
            hypercube.update_mean(dataset.iloc[:, :-1], self.predictor)
        return self.__hypercubes, (min_updates, surrounding)

    def __create_fake_samples(self, cube: HyperCube, n: int):
        return pd.DataFrame([cube.create_tuple() for _ in range(n, self.min_examples)])

    def __best_cube(self, dataset: pd.DataFrame, cube: HyperCube, cubes: list[Expansion]) -> Union[Expansion, None]:
        filtered = filter(lambda c: c.cube.count(dataset) is not None, cubes)
        expansions = []
        for limit in filtered:
            fake = self.__create_fake_samples(limit.cube, limit.cube.count(dataset))
            if fake is not None:
                self.__fake_dataset = self.__fake_dataset.append(fake)
            limit.cube.update_mean(self.__fake_dataset, self.predictor)
            expansions.append(Expansion(limit.cube, limit.feature, limit.direction, abs(cube.mean - limit.cube.mean)))
        if len(expansions) > 0:
            index = None
            min_distance = min([e.distance for e in expansions])
            for i, item in enumerate(expansions):
                if math.isclose(item.distance, min_distance):
                    index = i
                    break
            return expansions[index]
        else:
            return None

    @staticmethod
    def __create_range(cube: HyperCube, domain: DomainProperties, feature: str, direction: str) -> (
            HyperCube, (float, float)):
        min_updates, surrounding = domain
        a, b = cube.get(feature)
        size = [min_update for min_update in min_updates if min_update.name == feature][0].value
        return (cube.copy(), (max(a - size, surrounding.get_first(feature)), a) if direction == '-' else
                (b, min(b + size, surrounding.get_second(feature))))

    @staticmethod
    def __resolve_overlap(cube: HyperCube, overlapping_cube: HyperCube, hypercubes: list[HyperCube], feature: str,
                          direction: str) -> HyperCube:
        a, b = cube.get(feature)
        cube.update_dimension(feature, max(overlapping_cube.get_second(feature), a) if direction == '-' else a,
                              min(overlapping_cube.get_first(feature), b) if direction == '+' else b)
        return cube.overlap(hypercubes)

    @staticmethod
    def __create_temp_cube(cube: HyperCube, domain: DomainProperties, hypercubes: list[HyperCube], feature: str,
                           direction: str) -> list[Expansion]:
        result = []
        temp_cube, values = Iter.__create_range(cube, domain, feature, direction)
        temp_cube.update_dimension(feature, values)
        overlap = temp_cube.overlap(hypercubes)
        if overlap is not None:
            overlap = Iter.__resolve_overlap(temp_cube, overlap, hypercubes, feature, direction)
        if (temp_cube.has_volume() & (overlap is None)) & (not temp_cube.equal(hypercubes)):
            result.append(Expansion(temp_cube, feature, direction))
        else:
            cube.add_limit(feature, direction)
        return result

    @staticmethod
    def __create_temp_cubes(dataset: pd.DataFrame, cube: HyperCube, domain: DomainProperties,
                            hypercubes: list[HyperCube]) -> list[Expansion]:
        tmp_cubes = []
        for feature in dataset.columns[:-1]:
            limit = cube.check_limits(feature)
            if limit == '*':
                continue
            for x in {'-', '+'} - {limit}:
                tmp_cubes += Iter.__create_temp_cube(cube, domain, hypercubes, feature, x)
        return tmp_cubes

    def __cubes_to_update(self, dataset: pd.DataFrame, hypercubes: list[HyperCube], domain: DomainProperties) \
            -> list[tuple[HyperCube, Expansion]]:
        results = [(hypercube, self.__best_cube(dataset, hypercube, self.__create_temp_cubes(
            dataset, hypercube, domain, hypercubes))) for hypercube in hypercubes]
        return [result for result in results if result[1] is not None]

    def __iterate(self, dataset: pd.DataFrame, hypercubes: list[HyperCube], domain: DomainProperties,
                  left_interaction: int) -> int:
        interactions = 1
        while ((len(dataset.columns) - 1) * 2 * len(hypercubes)) >= \
                reduce(lambda acc, cube: acc + cube.limit_count, hypercubes, 0):
            if interactions == left_interaction:
                break
            updates = self.__cubes_to_update(dataset, hypercubes, domain)
            if len(updates) > 0:
                min_distance = min([expansion.distance for _, expansion in updates])
                to_update = \
                [(_, expansion) for _, expansion in updates if math.isclose(expansion.distance, min_distance)][0]
                self.__expand_or_create(to_update[0], to_update[1], hypercubes)
            interactions += 1
        return interactions

    def __expand_or_create(self, cube: HyperCube, expansion: Expansion, hypercubes: list[HyperCube]):
        if expansion.distance > self.threshold:
            hypercubes.append(expansion.cube)
        else:
            cube.expand(expansion, hypercubes)

    @staticmethod
    def __create_body(variables: dict[str, Var], dimensions: dict[str, (float, float)]) -> list[Struct]:
        return [create_term(variables[name], Between(values[0], values[1])) for name, values in dimensions.items()]

    def __create_theory(self, dataset: pd.DataFrame) -> Theory:
        new_theory = mutable_theory()
        for cube in self.__hypercubes:
            variables = create_variable_list([], dataset)
            head = create_head(dataset.columns[-1], list(variables.values()), cube.mean)
            body = Iter.__create_body(variables, cube.dimensions)
            new_theory.assertZ(
                clause(
                    head,
                    body
                )
            )
        return new_theory

    def extract(self, dataset: pd.DataFrame) -> Theory:
        hypercubes, domain = self.__init(dataset)
        temp_train = dataset.copy()
        iterations = 0
        while temp_train.shape[0] > 0:
            if iterations >= self.max_iterations:
                break
            iterations += self.__iterate(dataset, hypercubes, domain, self.max_iterations - iterations)
            if self.fill_gaps:
                raise Exception('NotImplementedError')
        self.__hypercubes = hypercubes
        return self.__create_theory(dataset)

    def predict(self, dataset) -> list:
        return [self.predictor.predict(data) for data in dataset]
