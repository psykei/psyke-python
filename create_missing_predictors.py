import ast
from psyke.predictor import Predictor
from psyke.utils import get_default_random_seed
from sklearn.model_selection import train_test_split
from test import REQUIRED_PREDICTORS, get_dataset, get_model
import pandas as pd

from test.resources.predictors import get_predictor_path, PATH

SEPARATOR: str = ';'

"""
Read the required predictors to run the tests:
hash | type | model | model_options | dataset
"""
required_predictors = pd.read_csv(REQUIRED_PREDICTORS, sep=SEPARATOR)

"""
Create missing predictors.
"""
for index, row in required_predictors.iterrows():
    file_name = row['dataset'] + row['model']
    if not get_predictor_path(file_name).is_file():
        dataset = get_dataset(row['dataset'])
        model = get_model(row['model'], ast.literal_eval(row['model_options']))
        training_set, test_set = train_test_split(dataset, test_size=0.5, random_state=get_default_random_seed())
        model.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
        predictor = Predictor(model)
        predictor.save_to_onnx(PATH / file_name, Predictor.get_initial_types(training_set.iloc[:, :-1]))

"""
Update required predictor hash for the new predictors.
"""
required_predictors.to_csv(REQUIRED_PREDICTORS, sep=SEPARATOR, index=False)


