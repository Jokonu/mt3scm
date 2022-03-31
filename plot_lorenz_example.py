"""
Comparison of unsupervised clustering metrics with lorenz attractor data
========================================================================

This example shows the effect of different metrics on the lorenz attractor dataset when using different types of label arrays. For the different unsupervised clustering labels we use the AgglomerativeClustering algorithm by varying the connectivity and the linkage as well as the number of clusters (along the lines of the scikit-learn example: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py)

"""

# Authors: Jonas Köhne
# License: BSD 3 clause

import string
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import numpy.typing as npt
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from mt3scm import MT3SCM


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
    label_array = np.zeros(df.shape[0])
    res = disfun(df[x_label], df[y_label], df[z_label])
    label_array = np.where(res < 1, 0, 1)
    df["label"] = label_array
    df = df.set_index("label", append=True)
    return df


def ax_scatter_3d(X, Y, Z, ax, labels, subplot_title: str = "Subplot Title", marker_size: float = 0.8, marker_size_array=None, marker="o", alpha=0.3, remove_ticks: bool = True):
    if marker_size_array is not None:
        marker_size = marker_size_array
    n_unique_labels = len(np.unique(labels))
    cmap = cm.get_cmap("viridis", len(np.unique(labels)))
    norm = Normalize(vmin=0, vmax=n_unique_labels, clip=False)
    scat = ax.scatter(X, Y, Z, c=labels, cmap=cmap, s=marker_size, marker=marker, norm=norm)
    ax.set_title(subplot_title)
    fig = plt.gcf()
    fig.colorbar(scat, ax=ax, shrink=0.5, pad=0.1)


def calc_unsupervised_metrics(X, label_array):
    # Metric comutations
    mt3 = MT3SCM(eps=5e-9)
    mt3scm_metric = mt3.mt3scm_score(X, label_array, standardize_subs_curve=True)
    metrics_dict = {}
    metrics_dict["mt3scm"] = mt3scm_metric
    metrics_dict["cc"] = mt3.wcc
    metrics_dict["masc-pos"] = mt3.masc_pos
    metrics_dict["masc-kt"] = mt3.masc_kt
    metrics_dict["silhouette"] = silhouette_score(X, label_array)
    metrics_dict["calinski"] = calinski_harabasz_score(X, label_array)
    metrics_dict["davies"] = davies_bouldin_score(X, label_array)
    return metrics_dict, mt3.kappa_X, mt3.tau_X


def generate_random_sequences(length: int = 1000, min_seq_length: int = 10, max_seq_length: int = 200, number_of_sequences: int = 10):
    data: np.ndarray = np.array([])
    while data.size < length:
        seq_len = np.random.randint(min_seq_length, max_seq_length)
        seq_id = np.random.choice(np.arange(number_of_sequences))
        seq_data = np.full(shape=(seq_len), fill_value=seq_id)
        data = np.append(data, seq_data) if data.size else seq_data
    data = data[:length].copy()
    _, label_array = np.unique(data, return_inverse=True)
    return label_array


def plot_random_examples(X: np.ndarray):
    n_sequences = [2, 3, 5, 10, 50, 200, 5000]
    min_max_seq_len = [(1, 2), (1, 100), (10, 20), (100, 500)]
    n_x_subfigs = 1
    n_x_subplots = len(min_max_seq_len)
    n_y_subfigs = len(n_sequences)
    n_y_subplots = 1
    alphabet_list = list(string.ascii_lowercase)
    subplot_labels = ["".join(x) for x in itertools.product(alphabet_list, repeat=2)]
    fig = plt.figure(constrained_layout=True, figsize=(4 * n_x_subplots, 4 * n_y_subfigs))
    # Setting global title
    fig.suptitle(r"\textbf{'Multivariate Time Series Sub-Sequence Clustering Metric' (MT3SCM) Evaluation}")
    # Create subfigures for connectivity and number of clusters
    subfigs = fig.subfigures(n_y_subfigs, n_x_subfigs, squeeze=True)
    idx = 0
    for subfig_index, number_of_sequences in enumerate(n_sequences):
        # Create subplots for all linkage variations
        axs = subfigs[subfig_index].subplots(n_y_subplots, n_x_subplots, subplot_kw=dict(projection="3d"), squeeze=True)
        subfigs[subfig_index].suptitle(f"Number of clusters: {number_of_sequences}")
        for subplots_index, (min_seq_len, max_seq_len) in enumerate(min_max_seq_len):
            # Generate random label sequences
            label_array = generate_random_sequences(length=X.shape[0], min_seq_length=min_seq_len, max_seq_length=max_seq_len, number_of_sequences=number_of_sequences)
            metrics, kappa_X, tau_X = calc_unsupervised_metrics(X, label_array)
            n_clusters = len(np.unique(label_array))
            subtitle = f"\\textbf{{Subfig. {subplot_labels[idx]}:}}, n_clusters={n_clusters}, min={min_seq_len}, max={max_seq_len}, \n\\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, masc-pos={metrics['masc-pos']:.3f}, masc-kt={metrics['masc-kt']:.3f},\nsil={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            # Scatter plot
            marker_sizes = np.log(kappa_X * tau_X * 100 + 1) * 5
            ax_scatter_3d(X[:, 0], X[:, 1], X[:, 2], axs[subplots_index], labels=label_array, subplot_title=subtitle, marker_size_array=marker_sizes)
            idx += 1
    plt.savefig(f"Random_example.png", dpi=300)
    plt.close()


def plot_agglomerative_clustering_example(X: np.ndarray) -> None:
    knn_graph = kneighbors_graph(X, 30, include_self=False)
    # Set linkage list
    linkage_list = ["average", "complete", "ward", "single"]
    # Set number of clusters list and connectivity
    n_clusters = [2, 3, 5, 10, 100, 1000]
    connect = [None, knn_graph]
    n_subfigs = len(n_clusters) * len(connect)
    fig = plt.figure(constrained_layout=True, figsize=(4 * len(linkage_list), 4 * n_subfigs))
    # Setting global title
    fig.suptitle(r"\textbf{'Multivariate Time Series Sub-Sequence Clustering Metric' (MT3SCM) Evaluation}")
    # Create subfigures for connectivity and number of clusters
    subfigs = fig.subfigures(n_subfigs, 1)
    subfig_index = 0
    # Subplots caption alphabet list
    alphabet_list = list(string.ascii_lowercase)
    subplot_labels = ["".join(x) for x in itertools.product(alphabet_list, repeat=2)]
    result_index_names = ["connectivity", "n_clusters", "linkage"]
    df_metrics = pd.DataFrame()
    # Counter
    idx = 0
    # Iterate over connectivity and number of clusters
    for connectivity in connect:
        for n_c in n_clusters:
            # Create subplots for all linkage variations
            axs = subfigs[subfig_index].subplots(1, len(linkage_list), subplot_kw=dict(projection="3d"))
            subfigs[subfig_index].suptitle(f"connectivity = {connectivity is not None}, n_clusters = {n_c}")
            for subplots_index, linkage in enumerate(linkage_list):
                print(f"Metrics for connectivity={connectivity is not None}, {linkage=}, {n_c=}")
                model = AgglomerativeClustering(linkage=linkage, connectivity=connectivity, n_clusters=n_c)
                model.fit(X)
                metrics, kappa_X, tau_X = calc_unsupervised_metrics(X, model.labels_)
                res_idx = [[connectivity is not None], [n_c], [linkage]]
                index = pd.MultiIndex.from_arrays(res_idx, names=result_index_names)
                data = np.fromiter(metrics.values(), dtype=float)
                data = np.expand_dims(data, 0)
                cols = list(metrics.keys())
                res_df = pd.DataFrame(data, index=index, columns=cols)
                df_metrics = pd.concat([res_df, df_metrics], axis=0, verify_integrity=True, names=result_index_names)

                subtitle = f"\\textbf{{Subfig. {subplot_labels[idx]}:}} {linkage=},\n\\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, masc-pos={metrics['masc-pos']:.3f}, masc-kt={metrics['masc-kt']:.3f},\nsil={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
                # Scatter plot
                marker_sizes = np.log(kappa_X * tau_X * 100 + 1) * 5
                ax_scatter_3d(X[:, 0], X[:, 1], X[:, 2], axs[subplots_index], labels=model.labels_, subplot_title=subtitle, marker_size_array=marker_sizes)
                idx += 1
            subfig_index += 1
    plt.savefig(f"ClusterComparison.png", dpi=300)
    plt.close()
    df_metrics.to_csv("ClusterMetricComparisonResults.csv")


def plot_examples():
    # Set style with seaborn
    sns.set_style("whitegrid")
    plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Computer Modern Serif"], "axes.grid": False})
    # Get lorenz attractor data as dataframe
    df = generate_lorenz_attractor_data(dt=0.005, num_steps=3001)
    X = df.values
    plot_agglomerative_clustering_example(X)
    plot_random_examples(X)


if __name__ == "__main__":
    plot_examples()
