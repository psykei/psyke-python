from sklearn.model_selection import train_test_split
from psyke.utils import get_default_random_seed
from psyke.utils.dataframe_utils import get_discrete_dataset
from test import get_dataset, get_schema, get_model

dataset = get_dataset('iris')
schema = get_schema('iris')
dataset = get_discrete_dataset(dataset.iloc[:, :-1], schema).join(dataset.iloc[:, -1])
training_set, test_set = train_test_split(dataset, test_size=0.5, random_state=get_default_random_seed())
training_set = training_set.sort_index()
m = get_model('knnc',
              {'n_neighbors': 7, 'n_jobs': 1})  # Predictor.load_from_onnx(str(get_predictor_path(row['predictor'])))
m.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])

predictions = []
for i in range(0, 10):
    predictions.append(m.predict(training_set.iloc[:, :-1]))

for i, p in enumerate(predictions[0]):
    print(i, p)
