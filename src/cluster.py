#cluster.py
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.config import PATH_OUTPUT_CLUSTER, SEMILLA

logger=logging.getLogger(__name__)
path_output_cluster=PATH_OUTPUT_CLUSTER

def clustering_kmeans(n_clusters:int , embedding:np.ndarray ,name:str):
    name=f"clusters_{n_clusters}"+name
    logger.info(f"Inicio del clustering {name}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEMILLA, n_init=10)
    clusters = kmeans.fit_predict(embedding)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(label='Cluster')

    for cluster_label in sorted(np.unique(clusters)):
        cluster_points = embedding[clusters == cluster_label]
        centroid = cluster_points.mean(axis=0)
        plt.text(centroid[0], centroid[1], str(cluster_label), fontsize=12, ha='center', va='center', color='black',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    try:
        file_image=f"{name}.png"
        plt.savefig(path_output_cluster+file_image, dpi=300, bbox_inches="tight")
        logger.info(f"Guardado del cluster {name}")
        logger.info(f"Finalizacion del clustering {name}")
    except Exception as e:
        logger.error(f"Error al guardar el archivo {file_image}")
    plt.close()
    