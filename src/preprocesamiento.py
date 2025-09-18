import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

import logging

logger = logging.getLogger(__name__)



def imputacion(X: pd.DataFrame|np.ndarray , strategy="median") -> pd.DataFrame:
    logger.info("Comienzo del preprocesamiento de los datos de train")

    imputer=SimpleImputer(missing_values=np.nan , strategy=strategy)
    X_imp = imputer.fit_transform(X)

    return pd.DataFrame(X_imp , columns=X.columns , index=X.index)

logger.info("Finalizacion del preprocesamiento de los datos de train")