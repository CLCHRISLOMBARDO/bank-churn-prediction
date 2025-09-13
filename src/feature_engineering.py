import pandas as pd
import numpy
import duckdb

def feature_engineering_lag(df:pd.DataFrame , columnas:list[str],cant_lag:int=1) -> pd.DataFrame:


    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """
    # Armado de la consulta SQL
    sql="SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(1,cant_lag+1):
                sql+= f",lag({attr},{i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            print(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df"

    # Ejecucion de la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df=con.execute(sql).df()
    con.close()
    

