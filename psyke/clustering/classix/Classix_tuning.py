import itertools
import pandas as pd
from psyke import Extractor
from sklearn import metrics



def tuning_params_classix(X_train,X_test,Y_test,parameters,mode):
    """
    params:
        X_train,X_test,Y_test: are the dataset will be used to fit and compute the predict on the Classix cluster
        parameters: is a dictionary that has as first parameter the values of the minPts, as second parameter the values of the radius, and the scale as third
        mode: is the mode of aggregation, that are "density" or "distance"
    output:
        list: first element in the list is the ARI score, the second the value of the minPTS, the third is the radius and the fourth is the scale.
    """
    n_params=len(list(parameters.keys()))
    list_value_params=[]
    list_name_params=list(parameters.keys())

    for k in list(parameters.keys()):
        list_value_params.append(parameters[k])
   
    if n_params==3:
        combinations_values_list=list(itertools.product(list_value_params[0],list_value_params[1],list_value_params[2]))
    else:
        print('Errore numero parametri')
    
    score_values=[]

    for i in range(len(combinations_values_list)):
        new_minPts=combinations_values_list[i][0]
        param_radius=combinations_values_list[i][1]
        param_scale=combinations_values_list[i][2]


        estimator=Extractor.classix(minPts=new_minPts,radius=param_radius,scale=param_scale,group_merging_mode=mode)
        

        
        estimator.fit(X_train)
        
        Y_pred=estimator.predict(X_test)

        score_values.append([metrics.adjusted_rand_score(Y_test,Y_pred),new_minPts,param_radius,param_scale])
    
    return score_values

    
    



        
    

                
