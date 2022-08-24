from functools import partial

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score


def mae(expected, predicted):
    """
    Calculates the predictions' MAE w.r.t. the instances given as input.

    :param expected: the expected data .
    :param predicted: the predicted data.
    :return: the mean absolute error (MAE) of the predictions.
    """
    return score(expected, predicted, mean_absolute_error)


def mse(expected, predicted):
    """
    Calculates the predictions' MSE w.r.t. the instances given as input.

    :param expected: the expected data .
    :param predicted: the predicted data.
    :return: the mean squared error (MSE) of the predictions.
    """
    return score(expected, predicted, mean_squared_error)


def r2(expected, predicted):
    """
    Calculates the predictions' R2 w.r.t. the instances given as input.

    :param expected: the expected data .
    :param predicted: the predicted data.
    :return: the R2 score of the predictions.
    """
    return score(expected, predicted, r2_score)


def accuracy(expected, predicted):
    """
    Calculates the predictions' classification accuracy w.r.t. the instances given as input.

    :param expected: the expected data .
    :param predicted: the predicted data.
    :return: the classification accuracy of the predictions.
    """
    return score(expected, predicted, accuracy_score)


def f1(expected, predicted):
    """
    Calculates the predictions' F1 score w.r.t. the instances given as input.

    :param expected: the expected data .
    :param predicted: the predicted data.
    :return: the F1 score of the predictions.
    """
    return score(expected, predicted, partial(f1_score, average='weighted'))


def score(expected, predicted, scoring_function):
    """
    Calculates the predictions' score w.r.t. the instances given as input with the provided scoring function.

    :param expected: the expected data .
    :param predicted: the predicted data.
    :param scoring_function: the scoring function to be used.
    :return: the score of the predictions.
    """
    idx = [prediction is not None for prediction in predicted]
    return scoring_function(expected[idx], predicted[idx])
