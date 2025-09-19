#constr_lista_cols.py
import pandas as pd
import numpy as np
import logging

logger=logging.getLogger(__name__)

def contruccion_cols(df:pd.DataFrame|np.ndarray)->list[list]:
    logger.info("Comienzo de la extraccion de la seleccion de las columnas")

    # Columnas categoricas y numericas
    cat_cols =[]
    num_cols=[]
    for c in df.columns:
        if (df[c].nunique() <= 5):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    lista_t=[c for c in list(map(lambda x : x if x[0]=='t' and x not in cat_cols else np.nan ,df.columns )) if pd.notna(c)]
    lista_c=[c for c in list(map(lambda x : x if x[0]=='c' and x not in cat_cols else np.nan ,df.columns )) if pd.notna(c)]
    lista_m=[c for c in list(map(lambda x : x if x[0]=='m' and x not in cat_cols else np.nan ,df.columns )) if pd.notna(c)]
    lista_r=[c for c in df.columns if c not in (lista_t + lista_c + lista_m + cat_cols)]
 

    # Columnas lags y delta
    col_drops=["cliente_edad","numero_de_cliente","cliente_antiguedad"]
    cols_transf_previo=lista_m + lista_c+ lista_r
    cols_lag_delta = [c for c in cols_transf_previo if c not in col_drops]


    # Columnas para regresion lineal y max-min
    cols_drops=["active_quarter","clase_ternaria"]
    lista_regl_max_min = [c for c in cat_cols if c not in  cols_drops] + cols_lag_delta

    # Columnas para los ratios
    lista_cant=[c for c in list(map(lambda x : x if x[0]=='c' else np.nan ,df.columns )) if pd.notna(c)]
    lista_monto = [c for c in list(map(lambda x : x if x[0]=='m' else np.nan ,df.columns )) if pd.notna(c)]
    cols_ratios=[]
    for c in lista_cant:
        i=0
        while i < len(lista_monto) and c[1:] != lista_monto[i][1:]:
            i+=1
        if i < len(lista_monto):
            cols_ratios.append([lista_monto[i],c ])
    logger.info("Finalizacion de la construccion de las columnas")

    return [cat_cols ,num_cols ,cols_lag_delta ,lista_regl_max_min,cols_ratios ]


