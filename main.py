import pandas as pd
import numpy as np
import os
import datetime 
import logging

from src.config import *
from src.loader import cargar_datos
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_delta, feature_engineering_lag , feature_engineering_ratio,feature_engineering_linreg,feature_engineering_max_min
from src.random_forests import optim_hiperp_binaria, entrenamiento_rf

## Carga de variables
n_subsample = 



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

def main():
    logger.info("Inicio de ejecucion.")

    ## 0. Cargar datos
    path="data/competencia_01.csv"
    df=cargar_datos(path)
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



    ## 2. Seleccion de submuestra
    n_sample_continua=10000
    mes =202104
    f = df["foto_mes"] == mes
    df = df.loc[f]


    path="data/"
    file="df_transf.csv"
    try:
        df.to_csv(path+file)
        logger.info(f"df shape {df.shape} guardado en csv")
    except Exception as e:
        logger.error(f"Error al guardar el df : {e}")
        raise

    X_train=df.drop(columns="clase_ternaria")
    y_train_ternaria = df["clase_ternaria"].copy()
    y_train=y_train_ternaria.map(lambda x : 0 if x =="Continua" else 1)

    np.random.seed(17)
    continua_sample = y_train[y_train ==0].sample(n_sample_continua).index
    bajas_1_2 = y_train[y_train ==1].index
    rf_index = continua_sample.union(bajas_1_2)
    X_train_rf = X_train.loc[rf_index]
    y_train_rf = y_train.loc[rf_index]

    ## 3. Random forest 

    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")


if __name__ =="__main__":
    main()