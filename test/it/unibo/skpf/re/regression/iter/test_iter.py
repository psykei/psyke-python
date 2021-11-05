import unittest
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from psyke.extractor import Extractor
from tuprolog.theory import *


class TestIter(unittest.TestCase):

    dataset = pd.DataFrame(pd.read_csv('test/resources/arti.csv'))
    training_set = dataset.sample(frac=0.5, random_state=123)
    test_set = dataset.loc[~dataset.index.isin(training_set.index)]
    kernel = RBF(10, (1e-2, 1e2))
    predictor = GPR(kernel=kernel, random_state=123)
    predictor.fit(dataset.iloc[:, :-1], dataset.iloc[:, -1])
    iter = Extractor.iter(predictor, min_update=1.0/20, threshold=0.19)
    theory = iter.extract(training_set)

    print(theory)




