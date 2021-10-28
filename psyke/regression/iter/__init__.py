
from psyke.utils.extractor import Extractor


class Iter(Extractor):

    def __init__(self, predictor, dataset, minUpdate, nPoints, maxIterations, minExamples, threshold, fillGaps):
        self.predictor = predictor
        self.discretization = None
        self.dataset = dataset
        self.minUpdate = minUpdate
        self.nPoints = nPoints
        self.maxIterations = maxIterations
        self.minExamples = minExamples
        self.threshold = threshold
        self.fillGaps = fillGaps
        self.seed = 123

