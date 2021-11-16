import pickle
import pandas as pd
from psyke.extractor import Extractor
from test.resources.schemas.iris import iris_features

resource_dir = 'test/resources/'
dataset = pd.read_csv(resource_dir + 'iris.data')
training_set = pd.read_csv(resource_dir + 'irisTrain50.csv')
test_set = dataset.loc[~dataset.index.isin(training_set.index)]
predictor = pickle.load(open(resource_dir + 'irisKNN5.txt', 'rb'))
features = iris_features
real = Extractor.real(predictor, iris_features)
theory = real.extract(training_set)

print(theory)
