#optimizacion.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed
import optuna
from optuna.study import Study
from time import time
import datetime

import pickle
import json
import logging

from src.config import *

output_path = PATH_OUTPUT_OPTIMIZACION
db_path = output_path + 'db/'
bestparms_path = output_path+'best_params/'

ganancia_acierto = GANANCIA
costo_estimulo = ESTIMULO

logger = logging.getLogger(__name__)

def optim_hiperp_binaria(X:pd.DataFrame|np.ndarray ,y:pd.Series|np.ndarray , n_trials:int)-> Study:
    logger.info("Comienzo optimizacion hiperp binario")
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")

    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 2000)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 200)
        max_features = trial.suggest_float('max_features', 0.05, 0.7)

        model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_samples=0.7,
            random_state=42,
            n_jobs=12,
            oob_score=True
        )

        model.fit(X, y)
        proba_oob = model.oob_decision_function_

        auc_score = roc_auc_score(y, proba_oob[:, 1])

        return auc_score
    
    storage_name = "sqlite:///" + db_path + "optimization_tree.db"
    study_name   = f"rf_binario_auc_{fecha}"   

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    
    # Mejor guardarlo en el main ?
    try:

        with open(bestparms_path+f"best_params_{fecha}.json", "w") as f:
            json.dump(best_params, f, indent=4) 
        
        logger.info("Finalizacion de optimizacion hiperp binario. Best parameters guardado en json")
    except Exception as e:
        logger.error(f"Error al tratar de guardar el json de los best parameters por el error :{e}")
    return study



def _ganancia_prob(y_hat:pd.Series|np.ndarray , y:pd.Series|np.ndarray ,prop=1,class_index:int =1,threshold:int=0.025)->float:
    logger.info("comienzo funcion ganancia con threhold = 0.025")
    @np.vectorize
    def _ganancia_row(predicted , actual , threshold=0.025):
        return (predicted>=threshold) * (ganancia_acierto if actual=="BAJA+2" else -costo_estimulo)
    logger.info("Finalizacion funcion ganancia con threhold = 0.025")
    return _ganancia_row(y_hat[:,class_index] ,y).sum() /prop


def optim_hiperp_ternaria(X:pd.DataFrame|np.ndarray ,y:pd.Series|np.ndarray , n_trials:int)-> Study:
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    logger.info("Inicio de optimizacion hiperp ternario")
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 2000)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 200)
        max_features = trial.suggest_float('max_features', 0.05, 0.7)

        model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_samples=0.7,
            random_state=42,
            n_jobs=12,
            oob_score=True
        )

        model.fit(X, y)

        return _ganancia_prob(model.oob_decision_function_, y)

    storage_name = "sqlite:///" + db_path + "optimization_tree.db"
    study_name = f"rf_ternario_ganancia_{fecha}"  

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    
    try:
        with open(bestparms_path+f"best_params_{fecha}.json", "w") as f:
            json.dump(best_params, f, indent=4) 
        logger.info("Finalizacion de optimizacion hiperp ternario. Best parameters guardado en json")
    except Exception as e:
        logger.error(f"Error al tratar de guardar el json de los best parameters por el error :{e}")


    
    return study


