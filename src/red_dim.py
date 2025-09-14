import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from umap import UMAP


def distanceMatrix(model:RandomForestClassifier, X:pd.DataFrame|np.ndarray)->np.ndarray:

    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:,0]
    proxMat = 1*np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:,i]
        proxMat += 1*np.equal.outer(a, a)

    proxMat = proxMat / nTrees

    return proxMat.max() - proxMat


def red_umap(md:np.ndarray, n_componentes:int=2,n_neighbors:int=20,min_dist:float=0.77 ,
             learning_rate:float=0.05 , metric:str="precomputed") ->np.ndarray:
    embedding_rf = UMAP(
    n_components=n_componentes,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    learning_rate=learning_rate,
    metric=metric,
    random_state=42,
    ).fit_transform(md)
    return embedding_rf
