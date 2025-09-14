import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import optuna
from optuna.study import Study
from time import time

import pickle


output_path = '../outputs/random_forest/'
db_path = output_path + 'db/'
model_path = output_path+'model'

ganancia_acierto = 780000
costo_estimulo = 20000

def _imputacion(X: pd.DataFrame|np.ndarray , strategy="median") -> pd.DataFrame:

    imputer=SimpleImputer(missing_values=np.nan , strategy=strategy)
    X_imp = imputer.fit_transform(X)

    return pd.DataFrame(X_imp , columns=X.columns , index=X.index)



def _ganancia_prob(y_hat:pd.Series|np.ndarray , y:pd.Series|np.ndarray ,prop=1,class_index:int =1,threshold:int=0.025)->float:
    @np.vectorize
    def _ganancia_row(predicted , actual , threshold=0.025):
        return (predicted>=threshold) * (ganancia_acierto if actual=="BAJA+2" else -costo_estimulo)
    return _ganancia_row(y_hat[:,class_index] ,y).sum() /prop


def optim_hiperp(X:pd.DataFrame|np.ndarray ,y:pd.Series|np.ndarray , n_trials:int)-> Study:
    Xi=_imputacion(X)

    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 2000)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 200)
        max_features = trial.suggest_float('max_features', 0.05, 0.7)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_samples=0.7,
            random_state=42,
            n_jobs=12,
            oob_score=True
        )

        model.fit(Xi, y)

        return _ganancia_prob(model.oob_decision_function_, y)

    storage_name = "sqlite:///" + db_path + "optimization_tree.db"
    study_name = "exp_206_random-forest-opt"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=25)
    return study

def entrenamiento_rf(X:pd.DataFrame|np.ndarray ,y:pd.Series|np.ndarray , best_parameters = dict[str, object])->RandomForestClassifier:
    Xi=_imputacion(X)
    model_rf = RandomForestClassifier(
        n_estimators=100,
        #**study.best_params,
        **best_parameters,
        max_samples=0.7,
        random_state=42,
        n_jobs=12,
        oob_score=True )
    model_rf.fit(Xi,y)
    filename=model_path+'exp_206_random_forest_model_100.sav'
    pickle.dump(model_rf, open(filename, 'wb'))
    return model_rf
    
