import pandas as pd
import numpy as np
import os
import datetime 
import logging
import json

from src.config import *
from src.loader import cargar_datos
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_delta, feature_engineering_lag , feature_engineering_ratio,feature_engineering_linreg,feature_engineering_max_min
from src.preprocesamiento import split_train_binario, submuestreo, imputacion
from src.optimizacion import optim_hiperp_binaria 
from src.random_forests import  entrenamiento_rf
print("ya cargo todo")
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## PATH
path_input_data = PATH_INPUT_DATA
path_output_data=PATH_OUTPUT_DATA

path_output_optim = PATH_OUTPUT_OPTIMIZACION
db_path = path_output_optim + 'db/'
bestparms_path = path_output_optim+'best_params/'


## Carga de variables
n_subsample=N_SUBSAMPLE
mes_train = MES_TRAIN_SEGM
n_trials=N_TRIALS

## config basic logging
os.makedirs("logss",exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
nombre_log = f"log_{fecha}.log"

logging.basicConfig(
    level=logging.DEBUG, #Puede ser INFO o ERROR
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logss/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def main():
    logger.info("Inicio de ejecucion.")

    ## 0. load datos
    df=cargar_datos(path_input_data)
    print(df.head())

    ## 1. Contruccion de las columnas
    columnas=contruccion_cols(df)
    cat_cols=columnas[0]
    num_cols=columnas[1]
    cols_lag_delta=columnas[2]
    lista_regl_max_min = columnas[3]
    cols_ratios=columnas[4]


    ## 2. Feature Engineering
    df=feature_engineering_lag(df,cols_lag_delta,2)
    df=feature_engineering_delta(df,cols_lag_delta,2)
    df=feature_engineering_max_min(df,lista_regl_max_min)
    df=feature_engineering_ratio(df,cols_ratios)
    df=feature_engineering_linreg(df,lista_regl_max_min)

            # Guardo df
    # try:
    #     df.to_csv(path_output_data + "df_after_feat_eng.csv") 
    #     logger.info(f"df shape {df.shape} guardado en csv")
    # except Exception as e:
    #     logger.error(f"Error al guardar el df : {e}")
    #     raise


    ## 2. Preprocesamiento para entrenamiento

    # split X_train, y_train
    X_train,y_train = split_train_binario(df,mes_train)
    # imputacion X_train
    X_train_imp = imputacion(X_train)
    # submuestreo
    X_train_sample_imp ,y_train_sample = submuestreo(X_train_imp,y_train, n_subsample)
    

    ## 3. Optimizacion Hiperparametros
    # a- Modelo sampleado
    study_rf_sample = optim_hiperp_binaria(X_train_sample_imp , y_train_sample ,n_trials)
    

    # b- Modelo completo
    study_rf_completo = optim_hiperp_binaria(X_train_imp , y_train , n_trials) 

    ## 3. Random forest 





    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")


if __name__ =="__main__":
    main()