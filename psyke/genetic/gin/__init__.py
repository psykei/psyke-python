from statistics import mode

import numpy as np
from deap import base, creator, tools, algorithms
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, f1_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures

from psyke import Target


class GIn:

    def __init__(self, train, valid, features, sigmas, slices, min_rules=1, poly=1, alpha=0.5, indpb=0.5, tournsize=3,
                 metric='R2', output=Target.REGRESSION, warm=False):
        self.X, self.y = train
        self.valid = valid
        self.output = output

        self.features = features
        self.sigmas = sigmas
        self.slices = slices
        self.min_rules = min_rules
        self.poly = PolynomialFeatures(degree=poly, include_bias=False)

        self.alpha = alpha
        self.indpb = indpb
        self.tournsize = tournsize
        self.metric = metric

        self.toolbox = None
        self.stats = None
        self.hof = None

        self.setup(warm)

    def region(self, X, cuts):
        indices = [np.searchsorted(np.array(cut), X[f].to_numpy(), side='right')
                   for cut, f in zip(cuts, self.features)]

        regions = np.zeros(len(X), dtype=int)
        multiplier = 1
        for idx, n in zip(reversed(indices), reversed([len(cut) + 1 for cut in cuts])):
            regions += idx * multiplier
            multiplier *= n

        return regions

    def __output_estimation(self, mask, to_pred):
        if self.output == Target.REGRESSION:
            return LinearRegression().fit(self.poly.fit_transform(self.X)[mask], self.y[mask]).predict(
                self.poly.fit_transform(to_pred))
        if self.output == Target.CONSTANT:
            return np.mean(self.y[mask])
        if self.output == Target.CLASSIFICATION:
            return mode(self.y[mask])
        raise TypeError('Supported outputs are Target.{REGRESSION, CONSTANT, CLASSIFICATION}')

    def __score(self, true, pred):
        if self.metric == 'R2':
            return r2_score(true, pred)
        if self.metric == 'MAE':
            return -mean_absolute_error(true, pred)
        if self.metric == 'MSE':
            return -mean_squared_error(true, pred)
        if self.metric == 'F1':
            return f1_score(true, pred, average='weighted')
        if self.metric == 'ACC':
            return accuracy_score(true, pred)
        raise NameError('Supported metrics are R2, MAE, MSE, F1, ACC')

    def evaluate(self, individual):
        to_pred, true = self.valid or (self.X, self.y)
        boundaries = np.cumsum([0] + list(self.slices))
        cuts = [sorted(individual[boundaries[i]:boundaries[i + 1]]) for i in range(len(self.slices))]

        regions = self.region(to_pred, cuts)
        regionsT = self.region(self.X, cuts)

        # y_pred = np.empty(len(to_pred), dtype=type(self.y.iloc[0]))
        y_pred = np.zeros(len(to_pred))
        valid_regions = 0

        for r in range(np.prod([s + 1 for s in self.slices])):
            mask = regions == r
            maskT = regionsT == r
            if min(mask.sum(), maskT.sum()) < 3:
                y_pred[mask] = mode(self.y[mask]) if self.output == Target.CLASSIFICATION else np.mean(self.y[mask])
                continue
            y_pred[mask] = self.__output_estimation(maskT, to_pred[mask])
            valid_regions += 1

        if valid_regions < self.min_rules:
            return -9999,

        return self.__score(true, y_pred),

    def setup(self, warm=False):
        if not warm:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        for f in self.features:
            self.toolbox.register(f, random.uniform, self.X[f].min(), self.X[f].max())

        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (sum([[getattr(self.toolbox, f) for i in range(s)]
                                    for f, s in zip(self.features, self.slices)], [])), n=1)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxBlend, alpha=self.alpha)
        self.toolbox.register("mutate", tools.mutGaussian, indpb=self.indpb, mu=0,
                              sigma=sum([[sig] * s for sig, s in zip(self.sigmas, self.slices)], []))
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        self.toolbox.register("evaluate", self.evaluate)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        self.stats.register("avg", np.mean)
        # self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        # self.stats.register("std", np.std)

        self.hof = tools.HallOfFame(1)

    def run(self, n_pop=30, cxpb=0.8, mutpb=0.5, n_gen=50, seed=123):
        random.seed(seed)
        pop = self.toolbox.population(n=n_pop)
        result, log = algorithms.eaSimple(pop, self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
                                          stats=self.stats, halloffame=self.hof, verbose=False)
        best = tools.selBest(pop, 1)[0]
        return best, self.evaluate(best)[0], result, log
