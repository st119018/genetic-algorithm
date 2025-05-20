from genetic_algorithm import GAClustering
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Optional, Tuple


def score_function(X: NDArray, labels: NDArray) -> float:
    """
    Compute score function on the input.

    Parameters
    ----------
    X : NDArray
        Input data
    labels: NDArray
        Labels of clusters

    Returns
    -------
    sf : float
        Measure of cluster quality
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    centroids = np.array(
        [X[labels == label].mean(axis=0) for label in unique_labels]
    )
    global_centroid = X.mean(axis=0)

    wcd = np.mean(
        [
            np.linalg.norm(X[labels == label] - centroids[i], axis=1).mean()
            for i, label in enumerate(unique_labels)
        ]
    )

    bcd = np.mean(np.linalg.norm(centroids - global_centroid, axis=1))

    diff = bcd - wcd
    sf = float(1.0 / (1.0 + np.exp(-diff)))
    return sf


def experiment(
    clusters: int, iterations: int, X: NDArray, y: Optional[NDArray] = None
) -> Tuple[pd.DataFrame, NDArray]:
    """
    Compare GA and KMeans clustering using adjusted
    Rand index (ARI) if y is given or score function (SF) if y is None.

    Parameters
    ----------
    clusters : int
        Number of clusters in the input data
    iterations : int
        Number of iterations
    X : NDArray
        Input data
    y : Optional[NDArray],default=None
        Labels
    Returns
    -------
    results : Tuple[pd.DataFrame, NDArray]
        Scores for both methods in each iteration and best found labels
    """
    results = []
    best_labels: Optional[NDArray] = None
    best_score: Optional[float] = None

    for i in range(iterations):
        km = KMeans(n_clusters=clusters, random_state=i, n_init=1)
        labels_km = km.fit_predict(X)

        if y is not None:
            ga = GAClustering(
                clusters, random_state=i, max_gen=100, pop_size=100
            )
            ga.fit(X)
            labels_ga = ga.predict(X)

            ari_score_ga = adjusted_rand_score(y, labels_ga)
            ari_score_km = adjusted_rand_score(y, labels_km)

            if best_score is None:
                best_score = ari_score_ga
            if ari_score_ga >= best_score:
                best_labels = labels_ga

            results.append(
                {"ARI (GA)": ari_score_ga, "ARI (KMeans)": ari_score_km}
            )
        else:
            ga = GAClustering(
                clusters,
                random_state=i,
                max_gen=150,
                pop_size=100,
                cross_rate=0.9,
                mut_rate=0.005,
            )
            ga.fit(X)
            labels_ga = ga.predict(X)

            sf_score_ga = score_function(X, labels_ga)
            sf_score_km = score_function(X, labels_km)

            if best_score is None:
                best_score = sf_score_ga
            if sf_score_ga >= best_score:
                best_labels = labels_ga

            results.append(
                {"SF (GA)": sf_score_ga, "SF (KMeans)": sf_score_km}
            )
    if best_labels is None:
        raise Exception("No valid solution was found.")

    return pd.DataFrame(results), best_labels
