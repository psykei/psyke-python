import math
import random
from functools import reduce
from typing import Iterable, Union
import pandas as pd
from tuprolog.core import Var, Struct, clause
from tuprolog.theory import Theory, mutable_theory
from psyke import logger, Extractor
from psyke.regression import HyperCube
from psyke.regression.utils import Expansion, MinUpdate
from psyke.schema import Between
from psyke.utils import get_default_random_seed, get_int_precision
from psyke.utils.logic import create_term, create_variable_list, create_head

DomainProperties = (Iterable[MinUpdate], HyperCube)


class ITER(Extractor):
    """
    Explanator implementing ITER algorithm, doi:10.1007/11823728_26.
    """

    def __init__(self, predictor, min_update, n_points, max_iterations, min_examples, threshold, fill_gaps,
                 seed=get_default_random_seed()):
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
        self.__generator = random.Random(seed)

    def __best_cube(self, dataset: pd.DataFrame, cube: HyperCube, cubes: Iterable[Expansion]) -> Union[Expansion, None]:
        filtered = filter(lambda c: c.cube.count(dataset) > 0, cubes)
        expansions = []
        for limit in sorted(filtered, key=lambda l: l.feature):
            count = limit.cube.count(dataset)
            fake = self.__create_fake_samples(limit.cube, count) if count > 0 else None
            if fake is not None and fake.shape[0] > 0:
                self.__fake_dataset = self.__fake_dataset.append(fake)
            limit.cube.update_mean(self.__fake_dataset, self.predictor)
            expansions.append(Expansion(limit.cube, limit.feature, limit.direction, abs(cube.mean - limit.cube.mean)))
        if len(expansions) > 0:
            index = None
            min_distance = min([e.distance for e in expansions])
            for i, item in enumerate(sorted(expansions, key=lambda e: e.feature)):
                if math.isclose(item.distance, min_distance):
                    index = i
                    break
            return expansions[index]
        else:
            return None

    def __calculate_min_updates(self, surrounding: HyperCube) -> Iterable[MinUpdate]:
        return [MinUpdate(name, (interval[1] - interval[0]) * self.min_update) for (name, interval) in
                surrounding.dimensions.items()]

    @staticmethod
    def __create_body(variables: dict[str, Var], dimensions: dict[str, (float, float)]) -> Iterable[Struct]:
        return [create_term(variables[name], Between(values[0], values[1])) for name, values in dimensions.items()]

    def __create_fake_samples(self, cube: HyperCube, n: int) -> pd.DataFrame:
        return pd.DataFrame([cube.create_tuple(self.__generator) for _ in range(n, self.min_examples)])

    @staticmethod
    def __create_range(cube: HyperCube, domain: DomainProperties, feature: str, direction: str)\
            -> (HyperCube, (float, float)):
        min_updates, surrounding = domain
        a, b = cube.get(feature)
        size = [min_update for min_update in min_updates if min_update.name == feature][0].value
        return (cube.copy(), (max(a - size, surrounding.get_first(feature)), a) if direction == '-' else
                (b, min(b + size, surrounding.get_second(feature))))

    @staticmethod
    def __create_temp_cube(cube: HyperCube, domain: DomainProperties, hypercubes: Iterable[HyperCube], feature: str,
                           direction: str) -> Iterable[Expansion]:
        temp_cube, values = ITER.__create_range(cube, domain, feature, direction)
        temp_cube.update_dimension(feature, values)
        overlap = temp_cube.overlap(hypercubes)
        if overlap is not None:
            overlap = ITER.__resolve_overlap(temp_cube, overlap, hypercubes, feature, direction)
        if (temp_cube.has_volume() & (overlap is None)) & (not temp_cube.equal(hypercubes)):
            yield Expansion(temp_cube, feature, direction)
        else:
            cube.add_limit(feature, direction)

    @staticmethod
    def __create_temp_cubes(dataset: pd.DataFrame, cube: HyperCube, domain: DomainProperties,
                            hypercubes: Iterable[HyperCube]) -> Iterable[Expansion]:
        tmp_cubes = []
        for feature in sorted(dataset.columns[:-1]):
            limit = cube.check_limits(feature)
            if limit == '*':
                continue
            for x in {'-', '+'} - {limit}:
                tmp_cubes += ITER.__create_temp_cube(cube, domain, hypercubes, feature, x)
        # Temporaries hypercubes are sorted based on both feature and direction.
        # This is critical to maintain reproducibility w.r.t. the same inputs.
        return sorted(tmp_cubes, key=lambda y: y.feature + y.direction)

    def __create_theory(self, dataset: pd.DataFrame) -> Theory:
        new_theory = mutable_theory()
        for cube in self.__hypercubes:
            logger.info(cube.mean)
            logger.info(cube.dimensions)
            variables = create_variable_list([], dataset)
            head = create_head(dataset.columns[-1], list(variables.values()), cube.mean)
            body = ITER.__create_body(variables, cube.dimensions)
            new_theory.assertZ(
                clause(
                    head,
                    body
                )
            )
        return new_theory

    def __cubes_to_update(self, dataset: pd.DataFrame, hypercubes: Iterable[HyperCube], domain: DomainProperties) \
            -> Iterable[tuple[HyperCube, Expansion]]:
        results = [(hypercube, self.__best_cube(dataset, hypercube, self.__create_temp_cubes(
            dataset, hypercube, domain, hypercubes))) for hypercube in hypercubes]
        return sorted([result for result in results if result[1] is not None], key=lambda x: x[1].feature)

    def __expand_or_create(self, cube: HyperCube, expansion: Expansion, hypercubes: Iterable[HyperCube]) -> None:
        if expansion.distance > self.threshold:
            hypercubes += [expansion.cube]
        else:
            cube.expand(expansion, hypercubes)

    @staticmethod
    def __find_closer_sample(dataset: pd.DataFrame, output: float) -> dict[str, tuple]:
        min_difference = min(abs(dataset.iloc[:, -1] - output))
        indices = [math.isclose(abs(value - output), min_difference) for value in dataset.iloc[:, -1]]
        close_samples = dataset.iloc[indices, :]
        close_sample = close_samples.iloc[0, :].to_dict()
        return close_sample

    def __generate_starting_points(self, dataset: pd.DataFrame) -> Iterable[HyperCube]:
        min_output = min(dataset.iloc[:, -1])
        max_output = max(dataset.iloc[:, -1])
        points: Iterable[float]
        if - math.inf < self.n_points <= 0:
            raise (Exception('InvalidAttributeValueException'))
        else:
            if self.n_points == 1:
                points = [(max_output - min_output) / 2]
            else:
                points = [(min_output + (max_output - min_output) / ((self.n_points - 1) * index)) for index in
                          range(1, self.n_points)]
        return [HyperCube.cube_from_point(ITER.__find_closer_sample(dataset, point)) for point in points]

    def __init(self, dataset: pd.DataFrame) -> tuple[Iterable[HyperCube], DomainProperties]:
        self.__fake_dataset = dataset.copy()
        surrounding = HyperCube.create_surrounding_cube(dataset)
        min_updates = self.__calculate_min_updates(surrounding)
        self.__hypercubes = self.__init_hypercubes(dataset, min_updates, surrounding)
        for hypercube in self.__hypercubes:
            hypercube.update_mean(dataset, self.predictor)
        return self.__hypercubes, (min_updates, surrounding)

    def __init_hypercubes(self, dataset, min_updates, surrounding) -> Iterable[HyperCube]:
        while True:
            hypercubes = self.__generate_starting_points(dataset)
            for hypercube in hypercubes:
                hypercube.expand_all(min_updates, surrounding)
            self.n_points = self.n_points - 1
            if not HyperCube.check_overlap(hypercubes, hypercubes):
                break
        return hypercubes

    def __iterate(self, dataset: pd.DataFrame, hypercubes: Iterable[HyperCube], domain: DomainProperties,
                  left_iteration: int) -> int:
        iterations = 1
        while ((len(dataset.columns) - 1) * 2 * len(list(hypercubes))) >= \
                reduce(lambda acc, cube: acc + cube.limit_count, hypercubes, 0):
            if iterations == left_iteration:
                break
            updates: Iterable[tuple[HyperCube, Expansion]] = self.__cubes_to_update(dataset, hypercubes, domain)
            if len(list(updates)) > 0:
                min_distance = min([expansion.distance for _, expansion in updates])
                to_update = [(_, expansion) for _, expansion in updates
                             if math.isclose(expansion.distance, min_distance)]
                to_update = sorted(to_update, key=lambda x: x[1].feature)[0]
                self.__expand_or_create(to_update[0], to_update[1], hypercubes)
            iterations += 1
        return iterations

    def __predict(self, data: dict[str, float]) -> float:
        data = {k: round(v, get_int_precision() + 1) for k, v in data.items()}
        for cube in self.__hypercubes:
            if cube.contains(data):
                return cube.mean
        return math.nan

    @staticmethod
    def __resolve_overlap(cube: HyperCube, overlapping_cube: HyperCube, hypercubes: Iterable[HyperCube], feature: str,
                          direction: str) -> HyperCube:
        a, b = cube.get(feature)
        cube.update_dimension(feature, max(overlapping_cube.get_second(feature), a) if direction == '-' else a,
                              min(overlapping_cube.get_first(feature), b) if direction == '+' else b)
        return cube.overlap(hypercubes)

    def extract(self, dataset: pd.DataFrame) -> Theory:
        hypercubes, domain = self.__init(dataset)
        temp_train = dataset.copy()
        iterations = 0
        while temp_train.shape[0] > 0:
            if iterations >= self.max_iterations:
                break
            iterations += self.__iterate(dataset, hypercubes, domain, self.max_iterations - iterations)
            if self.fill_gaps:
                raise NotImplementedError('Feature fill_gaps is not supported yet')
        self.__hypercubes = hypercubes
        return self.__create_theory(dataset)

    def predict(self, dataset: pd.DataFrame) -> Iterable:
        return [self.__predict(dict(row.to_dict())) for _, row in dataset.iterrows()]