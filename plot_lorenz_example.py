"""
Comparison of unsupervised clustering metrics with lorenz attractor data
========================================================================

This example shows the effect of different metrics on the lorenz attractor dataset when using different types of label arrays. For the different unsupervised clustering labels we use the AgglomerativeClustering algorithm by varying the connectivity and the linkage as well as the number of clusters (along the lines of the scikit-learn example: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py)

"""

# Authors: Jonas Köhne
# License: BSD 3 clause

import time
import string
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from mt3scm import mt3scm_score


def generate_lorenz_attractor_data(dt: float = 0.005, num_steps: int = 3000):
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot

    def disfun(x, y, z):
        x0 = 2
        y0 = 10
        z0 = 23
        return (1 - (x / x0) + (y / y0)) * z0 - z

    xs = np.empty(num_steps)
    ys = np.empty(num_steps)
    zs = np.empty(num_steps)

    # Set initial values
    xs[0], ys[0], zs[0] = (0.0, 1.0, 1.05)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps - 1):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    data = np.array([xs, ys, zs]).T
    feature_names = ["xs", "ys", "zs"]
    x_label, y_label, z_label = feature_names
    time_index = np.arange(start=0, stop=num_steps * dt, step=dt)
    idx = pd.TimedeltaIndex(time_index, unit="S", name="time")
    df = pd.DataFrame(data, columns=feature_names, index=idx)
    # xx, yy = np.meshgrid(range(-20, 20), range(-20, 20))
    # z = (1 - (xx/x0) + (yy/y0)) * z0
    label_array = np.zeros(df.shape[0])
    res = disfun(df[x_label], df[y_label], df[z_label])
    label_array = np.where(res < 1, 0, 1)
    df["label"] = label_array
    df = df.set_index("label", append=True)
    return df


def plot_lorenz_example(marker_size: float = 0.8) -> None:
    # Set style with seaborn
    sns.set_style("whitegrid")
    plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Computer Modern Serif"]})
    # Get lorenz attractor data as dataframe
    df = generate_lorenz_attractor_data(dt=0.005, num_steps=3001)

    X = df.values

    knn_graph = kneighbors_graph(X, 30, include_self=False)
    # Set linkage list
    linkage_list = ["average", "complete", "ward", "single"]
    # Set number of clusters list and connectivity
    n_clusters = [2, 3, 5, 10, 100, 1000]
    connect = [None, knn_graph]
    n_subfigs = len(n_clusters) * len(connect)

    fig = plt.figure(constrained_layout=True, figsize=(4 * len(linkage_list), 4 * n_subfigs))
    # Setting global title
    fig.suptitle(r"\textbf{'Multivariate Time Series Sub-Sequence CLustering Metric' (MT3SCM) Evaluation}")

    # Create subfigures for connectivity and number of clusters
    subfigs = fig.subfigures(n_subfigs, 1)
    subfig_index = 0
    # Subplots caption alphabet list
    alphabet_list = list(string.ascii_lowercase)
    subplot_labels = ["".join(x) for x in itertools.product(alphabet_list, repeat=2)]
    scores = {
        "mt3scm_score": mt3scm_score,
        "silhouette_score": silhouette_score,
        "calinski_harabasz_score": calinski_harabasz_score,
        "davies_bouldin_score": davies_bouldin_score
    }
    result_index_names = ["connectivity", "n_clusters", "linkage"]
    df_results = pd.DataFrame()
    # Counter
    idx = 0
    # Iterate over connectivity and number of clusters
    for connectivity in connect:
        for n_c in n_clusters:
            # Create subplots for all linkage variations
            axs = subfigs[subfig_index].subplots(1, 4, subplot_kw=dict(projection="3d"))
            subfigs[subfig_index].suptitle(f"connectivity = {connectivity is not None}, n_clusters = {n_c}")
            for subplots_index, linkage in enumerate(linkage_list):
                model = AgglomerativeClustering(linkage=linkage, connectivity=connectivity, n_clusters=n_c)
                t0 = time.time()
                model.fit(X)
                elapsed_time = time.time() - t0
                axs[subplots_index].scatter(X[:, 0], X[:, 1], X[:, 2], c=model.labels_, cmap=plt.cm.nipy_spectral, s=marker_size)
                results = {}
                for name, score_fn in scores.items():
                    results[name] = [score_fn(X, model.labels_)]
                res_idx = [[connectivity is not None], [n_c], [linkage]]
                index = pd.MultiIndex.from_arrays(res_idx, names=result_index_names)
                res_df = pd.DataFrame(np.array(list(results.values())).T, index=index, columns=results.keys())
                df_results = pd.concat([res_df, df_results], axis=0, verify_integrity=True, names=result_index_names)
                score = mt3scm_score(X, model.labels_)
                sil_score = silhouette_score(X, model.labels_)
                ch_score = calinski_harabasz_score(X, model.labels_)
                db_score = davies_bouldin_score(X, model.labels_)
                axs[subplots_index].set_title(
                    f"\\textbf{{Subfig. {subplot_labels[idx]}:}} {linkage=}\nmt3scm={score:.3f}, silhouette={sil_score:.3f},\ncalinski_harabasz={ch_score:.1f}, davies_bouldin={db_score:.3f}"
                )
                idx += 1
            subfig_index += 1
            plt.savefig(f"ClusterComparison.png", dpi=300)
    plt.close()
    df_results.to_csv("ClusterMetricComparisonResults.csv")

if __name__ == "__main__":
    plot_lorenz_example()
