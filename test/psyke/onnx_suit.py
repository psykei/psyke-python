import unittest
import pandas as pd
import sklearn.gaussian_process

from psyke.utils.predictor import Predictor

resource_dir = '../resources/'


class TestIter(unittest.TestCase):
    dataset = pd.DataFrame(pd.read_csv(resource_dir + 'arti.csv'))
    training_set = pd.read_csv(resource_dir + 'artiTrain50.csv')
    test_set = dataset.loc[~dataset.index.isin(training_set.index)]
    X_train, Y_train = training_set.iloc[:, :-1], training_set.iloc[:, -1]
    X_test, Y_test = test_set.iloc[:, :-1], test_set.iloc[:, -1]
    model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
    model.fit(X_train, Y_train)

    p = model.predict(X_test)
    print(sum(abs(p - Y_test) < 1e-6)/len(p))

    predictor = Predictor(model)
    predictor.save_to_onnx(resource_dir + 'artiKNN3.onnx', Predictor.get_initial_types(X_train))
    predictor = Predictor.load_from_onnx(resource_dir + 'artiKNN3.onnx')

    p = predictor.predict(X_test)
    print(sum(abs(p - Y_test) < 1e-6)/len(p))
