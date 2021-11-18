import ast
from psyke.predictor import Predictor
from psyke.utils import get_default_random_seed
from sklearn.model_selection import train_test_split
from psyke.utils.dataframe_utils import get_discrete_dataset
from test import REQUIRED_PREDICTORS, get_dataset, get_model, get_schema
import pandas as pd

from test.resources.predictors import get_predictor_path, PATH, create_predictor_name

SEPARATOR: str = ';'

"""
Read the required predictors to run the tests:
model | model_options | dataset
"""
required_predictors = pd.read_csv(REQUIRED_PREDICTORS, sep=SEPARATOR)

"""
Create missing predictors.
model | model_options | dataset | schema
"""
for index, row in required_predictors.iterrows():
    options = ast.literal_eval(row['model_options'])
    file_name = create_predictor_name(row['dataset'], row['model'], options)
    if not get_predictor_path(file_name).is_file():
        dataset = get_dataset(row['dataset'])
        schema = get_schema(row['schema'])
        if schema is not None:
            dataset = get_discrete_dataset(dataset.iloc[:, :-1], schema).join(dataset.iloc[:, -1])
        model = get_model(row['model'], options)
        training_set, test_set = train_test_split(dataset, test_size=0.5, random_state=get_default_random_seed())
        model.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
        predictor = Predictor(model)
        predictor.save_to_onnx(PATH / file_name, Predictor.get_initial_types(training_set.iloc[:, :-1]))

"""
Update required predictor hash for the new predictors.
"""
required_predictors.to_csv(REQUIRED_PREDICTORS, sep=SEPARATOR, index=False)


