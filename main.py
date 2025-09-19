import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime 
import logging
import json


from src.config import *
from src.loader import cargar_datos
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_delta, feature_engineering_lag , feature_engineering_ratio,feature_engineering_linreg,feature_engineering_max_min
from src.preprocesamiento import split_train_binario, submuestreo, imputacion
from src.optimizacion_rf import optim_hiperp_binaria 
from src.random_forests import  entrenamiento_rf,distanceMatrix
from src.embedding import embedding_umap
from src.cluster import clustering_kmeans
print("ya cargo todo")
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## PATH
path_input_data = PATH_INPUT_DATA
path_output_data=PATH_OUTPUT_DATA
path_output_optim = PATH_OUTPUT_OPTIMIZACION
db_path = path_output_optim + 'db/'
bestparms_path = path_output_optim+'best_params/'
path_output_umap=PATH_OUTPUT_UMAP


## Carga de variables
n_subsample=N_SUBSAMPLE
mes_train = MES_TRAIN_SEGM
n_trials=N_TRIALS

## config basic logging
os.makedirs("logss",exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{fecha}.log"

logging.basicConfig(
    level=logging.INFO, #Puede ser INFO o ERROR
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
    # df=feature_engineering_lag(df,cols_lag_delta,2)
    # df=feature_engineering_delta(df,cols_lag_delta,2)
    # df=feature_engineering_max_min(df,lista_regl_max_min)
    # df=feature_engineering_ratio(df,cols_ratios)
    # df=feature_engineering_linreg(df,lista_regl_max_min)

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
    

    ## 3. Optimizacion Hiperparametros y entrenamiento rf - Guardo las cosas en sus funciones respectivas, pero creo que tendria que hacerlo aca
    # a- Modelo sampleado
    name_rf_sample=f"_sampleado_{fecha}"
    study_rf_sample = optim_hiperp_binaria(X_train_sample_imp , y_train_sample ,n_trials , name=name_rf_sample)
    best_params_sample=study_rf_sample.best_params
    model_rf_sample=entrenamiento_rf(X_train_sample_imp , y_train_sample ,best_params_sample,name=name_rf_sample)
    class_index = np.where(model_rf_sample.classes_ == 1)[0]
    proba_baja_sample=model_rf_sample.predict_proba(X_train_sample_imp)[:,class_index]
    distancia_sample = distanceMatrix(model_rf_sample,X_train_sample_imp)
    
    # b- Modelo completo
    # name_rf_completo=f"_completo_{fecha}"
    # study_rf_completo = optim_hiperp_binaria(X_train_imp , y_train , n_trials , name=name_rf_completo) 
    # best_params_completo=study_rf_completo.best_params
    # model_rf_completo=entrenamiento_rf(X_train_imp , y_train ,best_params_completo,name=name_rf_sample)
    # class_index=np.where(model_rf_completo.classes_==1)[0]
    # proba_baja_completo=model_rf_sample.predict_proba(X_train_sample_imp)[:,class_index] #Predigo solo el subsampleo que es el que voy a graficar
    # distancia_con_completo_sampleado=distanceMatrix(model_rf_completo,X_train_sample_imp) # Calculo la dist solo con el subsampleo

    # 4. Embedding - UMAP
    embedding_sample=embedding_umap(distancia_sample)
    # embedding_comple = embedding_umap(distancia_con_completo_sampleado)

    # 5. Grafico del embedding coloreado por los predicts
    plt.scatter(embedding_sample[:,0], embedding_sample[:,1], c=proba_baja_sample)
    plt.colorbar()
    file_image=f"embedding_umap{name_rf_sample}.png"
    plt.savefig(path_output_umap+file_image, dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Gráfico de embeddings con las probabilidades de baja
    # plt.scatter(embedding_comple[:,0], embedding_comple[:,1], c=proba_baja_completo)
    # plt.colorbar()
    # file_image=f"embedding_umap{name_rf_completo}.png"
    # plt.savefig(path_output_umap+file_image, dpi=300, bbox_inches="tight")
    # plt.close()

    # 7. Clusters
    clusters=[4,5,6,7]
    # embeddings = [embedding_sample,embedding_comple]
    # names = [name_rf_sample , name_rf_completo]
    # for emb,name in zip(embeddings,names):
    #     for k in clusters:
    #         clustering_kmeans(k,emb,name)

    for k in clusters:
        clustering_kmeans(k,embedding_sample,name_rf_sample)
    











    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")


if __name__ =="__main__":
    main()