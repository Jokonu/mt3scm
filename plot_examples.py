"""
Comparison of unsupervised clustering metrics with lorenz attractor data
========================================================================

This example shows the effect of different metrics on the lorenz attractor dataset when using different types of label arrays. For the different unsupervised clustering labels we use the AgglomerativeClustering algorithm by varying the connectivity and the linkage as well as the number of clusters (along the lines of the scikit-learn example: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py)

"""

# Authors: Jonas Köhne
# License: BSD 3 clause

# Standard Libraries Import
import itertools
import pdb
import string
from pathlib import Path

# Third Party Libraries Import
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans

# Own Libraries Import
from mt3scm import MT3SCM
RESOLUTION_DPI = 300
TRANSPARENT = False
GRAPHICS_FORMAT = "png"  # or png, pdf, svg

def generate_thomas_attractor_data(dt: float = 1, num_steps: int = 2000, b:float = 0.1615):
    def thomas(x, y, z, b=0.1998):
        x_dot = np.sin(y) - (b * x)
        y_dot = np.sin(z) - (b * y)
        z_dot = np.sin(x) - (b * z)
        return x_dot, y_dot, z_dot
    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    # Set initial values
    xs[0], ys[0], zs[0] = (2, 1, 1)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = thomas(xs[i], ys[i], zs[i], b)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    data = np.array([xs, ys, zs*10]).T
    return pd.DataFrame(data, columns=["xs", "ys", "zs"])

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
    data = np.array([xs, ys, zs*10]).T
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

def gen_synth_data():
    # generate curve 1
    start = (0, 0, 1)
    end = (3*np.pi, 0, -1)
    n_points = 50
    t = np.linspace(start=start[0], stop=end[0], num=n_points)
    xt = t
    yt = np.sin(t)
    zt = np.cos(t)
    X1 = np.stack((xt, yt, zt)).T
    labels1 = 1 + np.zeros(n_points)

    # generate passing from curve 1 to 2
    start = (3*np.pi, -0.6, -1)
    end = (3*np.pi, -1.85, -1)
    n_points = 15
    t = np.linspace(start=-start[1], stop=-end[1], num=n_points)
    t = t**2 / 2
    # t = np.geomspace(0.674, 100.0, num=1000)
    xt = start[0] + np.zeros(n_points)
    yt = -t
    zt = start[2] + np.zeros(n_points)
    X2 = np.stack((xt, yt, zt)).T
    labels2 = 2 + np.zeros(n_points)

    # generate curve2 from passing 1
    start = (3.5*np.pi, -2, -1)
    end = (0.5*np.pi, 2, 1)
    n_points = 80
    t = np.linspace(start=start[0], stop=end[0], num=n_points)
    # t2 = np.linspace(start=3*np.pi, stop=0, num=n_points)
    xt = np.linspace(start=3*np.pi, stop=0, num=n_points)
    # xt = t - xt[0]
    yt = np.cos(t) - 2
    zt = np.sin(t)
    X3 = np.stack((xt, yt, zt)).T
    endpoint = (X3[-1, 0], X3[-1, 1], X3[-1, 2])
    print(f"{endpoint=}")
    labels3 = 3 + np.zeros(n_points)

    # generate passing from curve 2 to 1
    start = (0, -2, 1)
    end = (0, 0.01, -1)
    n_points = 15
    t = np.linspace(start=0.01, stop=1.9, num=n_points)
    t = t**2 / 2
    # t = np.geomspace(0.674, 100.0, num=1000)
    xt = start[0] + np.zeros(n_points)
    yt = -t
    zt = start[2] + np.zeros(n_points)
    X4 = np.stack((xt, yt, zt)).T
    labels4 = 4 + np.zeros(n_points)

    X = np.concatenate([X1, X2, X3, X4])
    X = np.concatenate([X, X+0.00001, X-0.00001])
    labels = np.concatenate([labels1, labels2, labels3, labels4])
    labels = np.concatenate([labels, labels, labels])

    # Repeat the data to have more subsequences per cluster
    n_repeats = 10
    X = np.repeat(X, n_repeats, axis=0)
    labels = np.repeat(labels, n_repeats, axis=0)
    # Add some randomness to the data.
    X = np.random.rand(X.shape[0], X.shape[1]) * 0.00001 + X
    return X, labels


def ax_scatter_3d(X, Y, Z, ax, labels, subplot_title: str = "Subplot Title", marker_size: float = 0.8, marker_size_array=None, marker="o", plot_changepoints: bool = False, alpha=0.3, remove_ticks: bool = True):
    if marker_size_array is not None:
        marker_size = marker_size_array
    if plot_changepoints is True:
        labs = np.where(labels[:-1] != labels[1:], 1, 0)
        labs = np.concatenate([[0], labs])
        scat2 = ax.scatter(X[labs == 1], Y[labs == 1], Z[labs == 1], s=20, c="black", marker=".", alpha=alpha, label='Changepoint', zorder=10)
    n_unique_labels = len(np.unique(labels))
    cmap = cm.get_cmap("viridis", n_unique_labels)
    norm = Normalize(vmin=0, vmax=n_unique_labels, clip=False)
    scat = ax.scatter(X, Y, Z, c=labels, cmap=cmap, s=marker_size, marker=marker, norm=norm)
    fig = plt.gcf()
    clb = fig.colorbar(scat, ax=ax, shrink=0.5, pad=0.2)
    clb.ax.set_title("Cluster ID", fontsize=10)
    if n_unique_labels < 11:
        tick_locs = (np.arange(n_unique_labels) + 0.5)
        clb.set_ticks(tick_locs)
        clb.set_ticklabels(np.arange(n_unique_labels))
    ax.set_title(subplot_title, fontsize=10)
    # ax.legend(fontsize=8)


def calc_unsupervised_metrics(X, label_array):
    # Metric comutations
    # mt3 = MT3SCM(eps=5e-9, include_speed_acceleration=True, distance_fn="manhatten")
    # mt3 = MT3SCM(eps=5e-9, include_acceleration=True, include_speed_acceleration=False, distance_fn="euclidean")
    mt3 = MT3SCM()
    # mt3 = MT3SCM(eps=5e-9, include_acceleration=True, include_speed_acceleration=False, distance_fn="manhatten")
    # mt3scm_metric = mt3.mt3scm_score(X, label_array, standardize_subs_curve=True)
    mt3scm_metric = mt3.mt3scm_score(X, label_array, edge_offset=0)
    metrics_dict = {}
    metrics_dict["mt3scm"] = mt3scm_metric
    metrics_dict["wcc"] = mt3.wcc
    metrics_dict["cc"] = mt3.cc
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
    n_unique_labels = len(np.unique(label_array))
    if n_unique_labels == 1:
        label_array[-1] = 1
    return label_array


def plot_random_examples(X: np.ndarray, dataset_name: str = "", n_clusters:list[int] = [2, 4, 10, 50, 200]):
    min_max_seq_len = [(1, 2), (1, 100), (10, 20), (100, 500)]
    n_x_subfigs = 1
    n_x_subplots = len(min_max_seq_len)
    n_y_subfigs = len(n_clusters)
    n_y_subplots = 1
    alphabet_list = list(string.ascii_lowercase)
    subplot_labels = ["".join(x) for x in itertools.product(alphabet_list, repeat=2)]
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.sans-serif": ["Computer Modern Roman"], "axes.grid": False})
    fig = plt.figure(1, constrained_layout=True, figsize=(4 * n_x_subplots, 4 * n_y_subfigs))
    # Setting global title
    fig.suptitle(r"\textbf{'Multivariate Time Series Sub-Sequence Clustering Metric' (MT3SCM) Evaluation}")
    # Create subfigures for connectivity and number of clusters
    subfigs = fig.subfigures(n_y_subfigs, n_x_subfigs, squeeze=True)
    idx = 0
    result_index_names = ["max_n_sequences", "n_clusters", "min_seq_len", "max_seq_len"]
    df_metrics = pd.DataFrame()
    for subfig_index, n_c in enumerate(n_clusters):
        # Create subplots for all linkage variations
        axs = subfigs[subfig_index].subplots(n_y_subplots, n_x_subplots, subplot_kw=dict(projection="3d"), squeeze=True)
        subfigs[subfig_index].suptitle(f"Number of clusters: {n_c}")
        for subplots_index, (min_seq_len, max_seq_len) in enumerate(min_max_seq_len):
            # Generate random label sequences
            label_array = generate_random_sequences(length=X.shape[0], min_seq_length=min_seq_len, max_seq_length=max_seq_len, number_of_sequences=n_c)
            metrics, kappa_X, tau_X = calc_unsupervised_metrics(X, label_array)
            # create and collect the results in a dataframe for further analysis
            res_df = create_results_dataframe_from_dict(metrics, index=[[n_c], [n_c],  [min_seq_len], [max_seq_len]], index_names=result_index_names)
            df_metrics = pd.concat([res_df, df_metrics], axis=0, verify_integrity=True, names=result_index_names)
            subtitle = f"\\textbf{{Subfig. {subplot_labels[idx]}:}}, n_clusters={n_c}, min={min_seq_len}, max={max_seq_len}, \n\\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['masc-pos']:.3f}, sp={metrics['masc-kt']:.3f},\nsilhouette={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            # Scatter plot
            marker_sizes = np.log((np.abs(kappa_X * tau_X * 100) + 1) ** 2)
            print(f"Drawing random example {n_c=} {min_seq_len=} {max_seq_len=}")
            ax_scatter_3d(X[:, 0], X[:, 1], X[:, 2], axs[subplots_index], labels=label_array, subplot_title=subtitle, marker_size_array=marker_sizes)
            # single_fig = plt.figure(2, constrained_layout=False, figsize=(4, 4))
            # ax = single_fig.add_subplot(projection="3d", computed_zorder=False)
            subtitle = f"n_clusters={n_c}, min={min_seq_len}, max={max_seq_len}, \\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['masc-pos']:.3f}, sp={metrics['masc-kt']:.3f},\nsilhouette={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            # ax_scatter_3d(X[:, 0], X[:, 1], X[:, 2], ax, labels=label_array, subplot_title=subtitle, marker_size_array=marker_sizes)
            # plt.figure(2)
            plot_name = f"ClusterMetricComparisonRandom-{dataset_name}-{subplot_labels[idx]}.{GRAPHICS_FORMAT}"
            # subplot_path = Path().cwd() / "plots"
            # Path(subplot_path).mkdir(parents=True, exist_ok=True)
            # plt.savefig(subplot_path / plot_name, pad_inches=0, bbox_inches="tight", transparent=TRANSPARENT, dpi=RESOLUTION_DPI, format=GRAPHICS_FORMAT)
            # plt.close(2)
            plot_single_figure(X, labels=label_array, subtitle=subtitle, plot_name=plot_name, marker_sizes=marker_sizes)
            idx += 1
    plot_name = f"ClusterMetricComparisonRandom-{dataset_name}.png"
    print(f"Saving plot with name: {plot_name}")
    plt.figure(1)
    plt.savefig(plot_name, dpi=300)
    plt.close(1)
    # Save resulting dataframe
    df_metrics.to_pickle(f"ClusterMetricComparisonResultsRandom-{dataset_name}.pkl")
    # df_metrics.to_csv(f"ClusterMetricComparisonResultsRandom-{dataset_name}.csv")

def plot_single_figure(X:np.ndarray, labels:np.ndarray, subtitle:str, plot_name:str, marker_sizes:np.ndarray):
    single_fig = plt.figure(2, constrained_layout=False, figsize=(4, 4))
    ax = single_fig.add_subplot(projection="3d", computed_zorder=False)
    ax_scatter_3d(X[:, 0], X[:, 1], X[:, 2], ax, labels=labels, subplot_title=subtitle, marker_size_array=marker_sizes)
    plt.figure(2)
    subplot_path = Path().cwd() / "plots"
    Path(subplot_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(subplot_path / plot_name, pad_inches=0, bbox_inches="tight", transparent=TRANSPARENT, dpi=RESOLUTION_DPI, format=GRAPHICS_FORMAT)
    plt.close(2)

def plot_agglomerative_clustering_example(X: np.ndarray, dataset_name: str = "", n_clusters:list[int] = [2, 3, 5, 10, 100, 1000], connect:list = [False], linkage_list:list[str] = ["average", "complete", "ward", "single"]) -> None:
    if connect[0] is False:
        connect = [None, kneighbors_graph(X, 30, include_self=False, n_jobs=20)]
    n_subfigs = len(n_clusters) * len(connect)
    fig = plt.figure(constrained_layout=True, figsize=(4 * len(linkage_list), 4 * n_subfigs))
    # Setting global title
    fig.suptitle(r"\textbf{'Multivariate Time Series Sub-Sequence Clustering Metric' (MT3SCM) Evaluation}")
    # Create subfigures for connectivity and number of clusters
    subfigs = fig.subfigures(n_subfigs, 1, squeeze=False)
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
            axs = subfigs[subfig_index][0].subplots(1, len(linkage_list), subplot_kw=dict(projection="3d"), squeeze=False)
            subfigs[subfig_index][0].suptitle(f"connectivity = {connectivity is not None}, n_clusters = {n_c}")
            for subplots_index, linkage in enumerate(linkage_list):
                model = AgglomerativeClustering(linkage=linkage, connectivity=connectivity, n_clusters=n_c)
                X_scaled = StandardScaler().fit_transform(X)
                model.fit(X_scaled)
                metrics, kappa_X, tau_X = calc_unsupervised_metrics(X, model.labels_)
                # create and collect the results in a dataframe for further analysis
                res_df = create_results_dataframe_from_dict(metrics, index=[[connectivity is not None], [n_c], [linkage]], index_names=result_index_names)
                df_metrics = pd.concat([res_df, df_metrics], axis=0, verify_integrity=True, names=result_index_names)
                subtitle = f"\\textbf{{Subfig. {subplot_labels[idx]}:}} {linkage=},\n\\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, masc-pos={metrics['masc-pos']:.3f}, masc-kt={metrics['masc-kt']:.3f},\nsil={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
                # Scatter plot
                marker_sizes = np.log((np.abs(kappa_X * tau_X * 100) + 1) ** 2)
                print(f"Drawing agglomerative example connectivity={connectivity is not None} {n_c=} {linkage=}")
                ax_scatter_3d(X[:, 0], X[:, 1], X[:, 2], axs[0][subplots_index], labels=model.labels_, subplot_title=subtitle, marker_size_array=marker_sizes)
                # single_fig = plt.figure(2, constrained_layout=False, figsize=(4, 4))
                # ax = single_fig.add_subplot(projection="3d", computed_zorder=False)
                subtitle = f"n_clusters={n_c}, {linkage=}, connectivity={connectivity is not None}, \\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['masc-pos']:.3f}, sp={metrics['masc-kt']:.3f},\nsilhouette={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
                # ax_scatter_3d(X[:, 0], X[:, 1], X[:, 2], ax, labels=model.labels_, subplot_title=subtitle, marker_size_array=marker_sizes)
                # plt.figure(2)
                plot_name = f"ClusterMetricComparisonAgglomerative-{dataset_name}-{subplot_labels[idx]}.{GRAPHICS_FORMAT}"
                # subplot_path = Path().cwd() / "plots"
                # Path(subplot_path).mkdir(parents=True, exist_ok=True)
                # plt.savefig(subplot_path / plot_name, pad_inches=0, bbox_inches="tight", transparent=TRANSPARENT, dpi=RESOLUTION_DPI, format=GRAPHICS_FORMAT)
                # plt.close(2)
                plot_single_figure(X, labels=model.labels_, subtitle=subtitle, plot_name=plot_name, marker_sizes=marker_sizes)
                idx += 1
            subfig_index += 1
    plot_name = f"ClusterMetricComparisonAgglomerative-{dataset_name}.png"
    print(f"Saving plot with name: {plot_name}")
    plt.figure(1)
    plt.savefig(plot_name, dpi=300)
    plt.close(1)
    # Save resulting dataframe
    df_metrics.to_pickle(f"ClusterMetricComparisonResultsAgglomerative-{dataset_name}.pkl")
    df_metrics.to_csv(f"ClusterMetricComparisonResultsAgglomerative-{dataset_name}.csv")


def plot_kmeans_example(X: np.ndarray, dataset_name: str = "", n_clusters:list[int] = [2, 3, 5, 10, 100], methods:list[str] = ["euclidean", "dtw", "softdtw"]) -> None:
    n_subfigs = len(n_clusters)
    n_subplots = len(methods)
    fig = plt.figure(constrained_layout=True, figsize=(4 * n_subplots, 4 * n_subfigs))
    # Setting global title
    fig.suptitle(r"\textbf{'Multivariate Time Series Sub-Sequence Clustering Metric' (MT3SCM) Evaluation}")
    # Create subfigures for connectivity and number of clusters
    subfigs = fig.subfigures(n_subfigs, 1, squeeze=False)
    # Subplots caption alphabet list
    alphabet_list = list(string.ascii_lowercase)
    subplot_labels = ["".join(x) for x in itertools.product(alphabet_list, repeat=2)]
    result_index_names = ["method", "n_clusters"]
    df_metrics = pd.DataFrame()
    # Counter
    idx = 0
    # Iterate over connectivity and number of clusters
    for subfig_index, n_c in enumerate(n_clusters):
        # Create subplots for all linkage variations
        axs = subfigs[subfig_index][0].subplots(1, n_subplots, subplot_kw=dict(projection="3d"), squeeze=False)
        subfigs[subfig_index][0].suptitle(f"n_clusters = {n_c}")
        for subplots_index, method in enumerate(methods):
            X_scaled = StandardScaler().fit_transform(X)
            model = TimeSeriesKMeans(n_clusters=n_c, metric=method, max_iter=5, random_state=42, n_jobs=-1).fit(X_scaled)
            metrics, kappa_X, tau_X = calc_unsupervised_metrics(X, model.labels_)
            # create and collect the results in a dataframe for further analysis
            res_df = create_results_dataframe_from_dict(metrics, index=[[method], [n_c]], index_names=result_index_names)
            df_metrics = pd.concat([res_df, df_metrics], axis=0, verify_integrity=True, names=result_index_names)
            subtitle = f"\\textbf{{Subfig. {subplot_labels[idx]}:}} {method=},\n\\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, masc-pos={metrics['masc-pos']:.3f}, masc-kt={metrics['masc-kt']:.3f},\nsil={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            # Scatter plot
            marker_sizes = np.log((np.abs(kappa_X * tau_X * 100) + 1) ** 2)
            print(f"Drawing TimeSeriesKMeans example connectivity={method=} {n_c=}")
            ax_scatter_3d(X[:, 0], X[:, 1], X[:, 2], axs[0][subplots_index], labels=model.labels_, subplot_title=subtitle, marker_size_array=marker_sizes)
            subtitle = f"{method=}, \\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, masc-pos={metrics['masc-pos']:.3f}, masc-kt={metrics['masc-kt']:.3f},\nsil={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            plot_name = f"ClusterMetricComparisonTimeSeriesKMeans-{dataset_name}-{subplot_labels[idx]}.{GRAPHICS_FORMAT}"
            plot_single_figure(X, labels=model.labels_, subtitle=subtitle, plot_name=plot_name, marker_sizes=marker_sizes)
            idx += 1
    plot_name = f"ClusterMetricComparisonTimeSeriesKMeans-{dataset_name}.png"
    print(f"Saving plot with name: {plot_name}")
    plt.savefig(plot_name, dpi=300)
    plt.close()
    # Save resulting dataframe
    df_metrics.to_pickle(f"ClusterMetricComparisonResultsTimeSeriesKMeans-{dataset_name}.pkl")
    df_metrics.to_csv(f"ClusterMetricComparisonResultsTimeSeriesKMeans-{dataset_name}.csv")

def create_results_dataframe_from_dict(metrics: dict[str, float], index:list, index_names:list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(index, names=index_names)
    data = np.fromiter(metrics.values(), dtype=float)
    data = np.expand_dims(data, 0)
    cols = list(metrics.keys())
    df = pd.DataFrame(data, index=index, columns=cols)
    return df

def plot_examples():
    # Set style with seaborn
    sns.set_style("whitegrid")
    plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Computer Modern Serif"], "axes.grid": False})
    # Get lorenz attractor data as dataframe
    df_lorenz = generate_lorenz_attractor_data(dt=0.005, num_steps=3001)
    # Get thomas attractor data as dataframe
    df_thomas = generate_thomas_attractor_data(dt=0.05, num_steps=10000, b = 0.1)
    # Get Synthetic dataset
    X_synth, labels_synth = gen_synth_data()
    # datasets = {"lorenz": df_lorenz.values, "thomas": df_thomas.values, "synthetic": X_synth}
    datasets = {"synthetic": X_synth}
    n_clusters = [2, 4, 10, 50, 200]
    for name, data in datasets.items():
        plot_agglomerative_clustering_example(data, dataset_name=name, n_clusters=n_clusters)
        plot_random_examples(data, dataset_name=name, n_clusters=n_clusters)
        plot_kmeans_example(data, dataset_name=name, n_clusters=n_clusters)


def plot_one_example():
    # Set style with seaborn
    sns.set_style("whitegrid")
    plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Computer Modern Serif"], "axes.grid": False})
    # Get lorenz attractor data as dataframe
    df_lorenz = generate_lorenz_attractor_data(dt=0.005, num_steps=3001)
    X_lorenz = StandardScaler().fit_transform(df_lorenz.values)
    # Get thomas attractor data as dataframe
    df_thomas = generate_thomas_attractor_data(dt=0.05, num_steps=10000, b = 0.1)
    X_thomas = StandardScaler().fit_transform(df_thomas.values)
    # datasets = {"thomas_single_example": X_thomas, "lorenz_single_example": X_lorenz}
    datasets = {"lorenz_single_example": X_lorenz}
    for name, data in datasets.items():
        plot_agglomerative_clustering_example(data, dataset_name=name, n_clusters=[3], connect=[kneighbors_graph(data, 30, include_self=False, n_jobs=-1)], linkage_list=["average"])

def plot_curvature_torsion_example():
    plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Computer Modern Serif"], "axes.grid": False})
    # Get thomas attractor data as dataframe
    df_thomas = generate_thomas_attractor_data(dt=0.05, num_steps=500, b = 0.1)
    X_thomas = StandardScaler().fit_transform(df_thomas.values)
    mt3 = MT3SCM()
    kappa, tau, speed, acceleration = mt3.compute_curvature(X_thomas)
    curv_data = [kappa, tau]
    curv_data = {"Curvature": kappa, "Torsion": tau, "Speed": speed, "Acceleration": acceleration}
    # n_subfigs = len(speed_data.keys())
    n_subfigs = 1
    n_subplots = len(curv_data.keys())
    fig = plt.figure(constrained_layout=True, figsize=(4 * n_subplots, 4 * n_subfigs))
    # Setting global title
    # fig.suptitle(r"\textbf{'Multivariate Time Series Sub-Sequence Clustering Metric' (MT3SCM) Components}")
    # Create subfigures for connectivity and number of clusters
    subfigs = fig.subfigures(n_subfigs, 1, squeeze=False)
    subfig_index = 0
    subplots_index = 0
    axs = subfigs[subfig_index][0].subplots(1, n_subplots, subplot_kw=dict(projection="3d"), squeeze=False)
    subfigs[subfig_index][0].suptitle(f"")
    for subplots_index, (name, data) in enumerate(curv_data.items()):
        subtitle = f"{name}"
        cmap = cm.get_cmap("viridis")
        marker="."
        ax = axs[0][subplots_index]
        data = (data - data.min()) / (np.std(data))
        color = np.log((np.abs(data* 10) + 1) ** 2)*100
        scat = ax.scatter(X_thomas[:, 0], X_thomas[:, 1], X_thomas[:, 2], c=color, cmap=cmap, s=color, marker=marker)
        ax.set_title(subtitle)
        fig = plt.gcf()
        clb = fig.colorbar(scat, ax=ax, shrink=0.5, pad=0.1)
        clb.set_ticks([color.min(),color.max() / 2, color.max()])
        clb.set_ticklabels(['Low', 'Medium', 'High'])
        # clb.ax.set_title(name, fontsize=10)
    plot_name = f"ClusterMetricCurvatureTorsionExample.png"
    print(f"Saving plot with name: {plot_name}")
    plt.savefig(plot_name, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Standard Libraries Import
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--one-only", dest="one", help="plot only a single example", action="store_true")
    parser.add_argument("-c", "--curve", dest="curve", help="plot  curvature torsion example", action="store_true")
    parser.add_argument("-k", "--kmeans", dest="kmeans", help="plot kmeans example", action="store_true")
    args = parser.parse_args()
    if args.one is True:
        print(f"Plotting a single example only..")
        plot_one_example()
    elif args.curve is True:
        print(f"Plotting curvature torsion example..")
        plot_curvature_torsion_example()
    elif args.kmeans is True:
        print(f"Plotting curvature torsion example..")
        plot_kmeans_example()
    else:
        print(f"Plotting all examples..")
        plot_examples()
    print(f"Done")


