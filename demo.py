from psyke import Extractor, DiscreteFeature
from psyke.schema import LessThan, Between, GreaterThan
from psyke.utils.dataframe_utils import get_discrete_dataset
from psyke.utils.logic_utils import pretty_theory
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


iris_features = {
    DiscreteFeature(
        "SepalLength",
        {
            "SepalLength_0": LessThan(5.39),
            "SepalLength_1": Between(5.39, 6.26),
            "SepalLength_2": GreaterThan(6.26)
        }),
    DiscreteFeature(
        "SepalWidth",
        {
            "SepalWidth_0": LessThan(2.87),
            "SepalWidth_1": Between(2.87, 3.2),
            "SepalWidth_2": GreaterThan(3.2)
        }),
    DiscreteFeature(
        "PetalLength",
        {
            "PetalLength_0": LessThan(2.28),
            "PetalLength_1": Between(2.28, 4.87),
            "PetalLength_2": GreaterThan(4.87)
        }),
    DiscreteFeature(
        "PetalWidth",
        {
            "PetalWidth_0": LessThan(0.65),
            "PetalWidth_1": Between(0.65, 1.64),
            "PetalWidth_2": GreaterThan(1.64)
        })
}
x, y = load_iris(return_X_y=True, as_frame=True)
x.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
x = get_discrete_dataset(x, iris_features)
y = pd.DataFrame(y).replace({"target": {0: 'setosa', 1: 'virginica', 2: 'versicolor'}})
dataset = x.join(y)
dataset.columns = [*dataset.columns[:-1], 'iris']
train, test = train_test_split(dataset, test_size=0.5, random_state=0)

predictor = KNeighborsClassifier(n_neighbors=7)
predictor.fit(train.iloc[:, :-1], train.iloc[:, -1])

real = Extractor.real(predictor, iris_features)
theory_from_real = real.extract(train)
print('REAL extracted rules:\n' + pretty_theory(theory_from_real))

trepan = Extractor.trepan(predictor, iris_features)
theory_from_trepan = trepan.extract(train)
print('\nTrepan extracted rules:\n' + pretty_theory(theory_from_trepan))
