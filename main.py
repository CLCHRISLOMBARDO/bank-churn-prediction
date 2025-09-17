import pandas as pd
import os
import datetime 
import logging 

from src.loader import cargar_datos
## config basic logging
os.makedirs("logs",exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
nombre_log = f"log_{fecha}.log"

logging.basicConfig(
    level=logging.DEBUG, #Puede ser INFO o ERROR
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Inicio de ejecucion.")

    ## Cargar datos
    path="data/competencia_01.csv"
    df=cargar_datos(path)

    print(df.head())

    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")


if __name__ =="__main__":
    main()