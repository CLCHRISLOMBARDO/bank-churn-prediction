import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

import logging
from time import time
import datetime

import pickle
import json

logger = logging.getLogger(__name__)

output_path = 'outputs/random_forest/'


ganancia_acierto = 780000
costo_estimulo = 20000



def entrenamiento_rf(X:pd.DataFrame|np.ndarray ,y:pd.Series|np.ndarray , best_parameters = dict[str, object])->RandomForestClassifier:
    
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    model_rf = RandomForestClassifier(
        n_estimators=1000,
        #**study.best_params,
        **best_parameters,
        max_samples=0.7,
        random_state=42,
        n_jobs=12,
        oob_score=True )
    model_rf.fit(X,y)
    filename=output_path+f'rf_model_{fecha}.sav'
    pickle.dump(model_rf, open(filename, 'wb'))
    return model_rf
    
