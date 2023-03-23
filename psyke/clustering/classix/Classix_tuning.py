import itertools
from ctypes import ArgumentError

from psyke import Clustering
from sklearn import metrics


def tuning_params_classix(X_train, X_test, Y_test, parameters, mode):

    keys = list(parameters.keys())
    if len(keys) != 3:
        raise ArgumentError('Wrong amount of parameters')

    score_values = []

    for minPts, radius, scale in itertools.product(*[parameters[k] for k in keys]):
        estimator = Clustering.classix(minPts=10, radius=.31, group_merging_mode=mode).fit(X_train)
        score_values.append([metrics.adjusted_rand_score(Y_test, estimator.predict(X_test)), minPts, radius, scale])

    return score_values

    
    



        
    

                
