"""
Comparison of unsupervised clustering metrics with lorenz attractor data
========================================================================

This example shows the effect of different metrics on the lorenz attractor dataset when using different types of label arrays. For the different unsupervised clustering labels we use the AgglomerativeClustering algorithm by varying the connectivity and the linkage as well as the number of clusters (along the lines of the scikit-learn example: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py)

"""

# Authors: Jonas Köhne
# License: BSD 3 clause

# Standard Libraries Import
import itertools
import string
from pathlib import Path

# Third Party Libraries Import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans

# Own Libraries Import
import helpers
from mt3scm import MT3SCM

RESOLUTION_DPI = 300
TRANSPARENT = False
GRAPHICS_FORMAT = "png"  # or png, pdf, svg


def publication_plot_random_examples(plot_name: str = "metric_random", graphics_format: str = GRAPHICS_FORMAT, resolution_dpi: int = RESOLUTION_DPI):
    # Get lorenz attractor data as dataframe
    df_lorenz = helpers.generate_lorenz_attractor_data(dt=0.005, num_steps=3001)
    # Get thomas attractor data as dataframe
    df_thomas = helpers.generate_thomas_attractor_data(dt=0.05, num_steps=10000, b=0.1)
    n_y_subplots = 2
    n_x_subplots = 2
    fig_titles = string.ascii_lowercase[:(n_y_subplots * n_x_subplots)]
    helpers.set_plot_params()
    # textwidth of two column paper: 17.75cm
    cm = 1/2.54
    fig, axs = plt.subplots(n_y_subplots, n_x_subplots, subplot_kw=dict(projection="3d"), constrained_layout=True, figsize=(17.75*cm*n_x_subplots/2, 17.75*cm*n_y_subplots/2), squeeze=False)
    # plt.subplots_adjust(wspace=0.2,hspace=0.2)
    min_max_seq_len = [(1, 2), (100, 500), (1, 2), (100, 500)]
    feat_names = ["x", "y", "z"]
    data = [df_lorenz.values, df_lorenz.values, df_thomas.values, df_thomas.values]
    df_mets = pd.DataFrame()
    for s_i, ax in enumerate(axs.flat):
        X = data[s_i]
        min_seq_len = min_max_seq_len[s_i][0]
        max_seq_len = min_max_seq_len[s_i][1]
        n_c = 2
        y = helpers.generate_random_sequences(length=X.shape[0], min_seq_length=min_seq_len, max_seq_length=max_seq_len, number_of_sequences=n_c)
        metrics, kappa_X, tau_X = helpers.calc_unsupervised_metrics(X, y, edge_offset=5, n_min_subs=1)
        print(f"Plotted plot {fig_titles[s_i]} with metrics davies-bouldin: {metrics['davies']:.2n}, calinski-harabasz: {metrics['calinski']:.2n}, silhouette: {metrics['silhouette']:.2n}, mt3scm: {metrics['mt3scm']:.2n}, cc={metrics['cc']:.2n}, wcc={metrics['wcc']:.2n},sl= {metrics['sl']:.2n}, sp={metrics['sp']:.2n}")
        marker_sizes = np.log((np.abs(kappa_X * tau_X * 100) + 1) ** 2)
        print(f"Drawing random example {n_c=} {min_seq_len=} {max_seq_len=} mt3scm={metrics['mt3scm']:.3f}")
        helpers.ax_scatter_3d( X[:, 0], X[:, 1], X[:, 2], ax, labels=y, subplot_title=f"({fig_titles[s_i]})", marker_size_array=marker_sizes, fontsize=8, set_ticks_and_labels=True)
        df_mets = pd.concat([df_mets, pd.DataFrame(metrics, index=[fig_titles[s_i]])])
    cols = ["davies", "calinski", "silhouette", "mt3scm", "cc", "wcc", "sl", "sp"]
    df_mets = df_mets[cols]
    la_style = df_mets.style
    la_style = la_style.format('{:.2G}')
    la_style.to_latex(f"{plot_name}.tex")
    print(f"mets: \n{df_mets}")
    print(f"Saving plot with name: {plot_name}")
    subplot_path = Path().cwd() / "plots"
    Path(subplot_path).mkdir(parents=True, exist_ok=True)
    plt.figure(1)
    plt.savefig(
        subplot_path / str(plot_name + "." + graphics_format),
        pad_inches=0,
        bbox_inches="tight",
        transparent=TRANSPARENT,
        dpi=resolution_dpi,
        format=graphics_format
    )
    plt.close(1)

def publication_plot_agglomerative_examples(plot_name: str = "metric_agglomerative", graphics_format: str = GRAPHICS_FORMAT, resolution_dpi: int = RESOLUTION_DPI):
    # Get lorenz attractor data as dataframe
    df_lorenz = helpers.generate_lorenz_attractor_data(dt=0.005, num_steps=3001, scale_zs=1)
    n_y_subplots = 2
    n_x_subplots = 2
    fig_titles = string.ascii_lowercase[:(n_y_subplots * n_x_subplots)]
    helpers.set_plot_params()
    # textwidth of two column paper: 17.75cm
    cm = 1/2.54
    fig, axs = plt.subplots(n_y_subplots, n_x_subplots, subplot_kw=dict(projection="3d"), constrained_layout=True, figsize=(17.75*cm*n_x_subplots/2, 17.75*cm*n_y_subplots/2), squeeze=False)
    linkages = ["average", "ward", "average", "ward"]
    n_cs = [4, 4, 10, 50]
    X = df_lorenz.values
    df_mets = pd.DataFrame()
    for s_i, ax in enumerate(axs.flat):
        linkage = linkages[s_i]
        n_c = n_cs[s_i]
        agc_model = AgglomerativeClustering(linkage=linkage, connectivity=None, n_clusters=n_c)
        X_scaled = StandardScaler().fit_transform(X)
        agc_model.fit(X_scaled)
        y = agc_model.labels_
        metrics, kappa_X, tau_X = helpers.calc_unsupervised_metrics(X, y, edge_offset=5, n_min_subs=1, include_std_num_points=True)
        print(f"Plotted plot {fig_titles[s_i]} with metrics davies-bouldin: {metrics['davies']:.2n}, calinski-harabasz: {metrics['calinski']:.2n}, silhouette: {metrics['silhouette']:.2n}, mt3scm: {metrics['mt3scm']:.2n}, cc={metrics['cc']:.2n}, wcc={metrics['wcc']:.2n},sl= {metrics['sl']:.2n}, sp={metrics['sp']:.2n}")
        marker_sizes = np.log((np.abs(kappa_X * tau_X * 100) + 1) ** 2)
        print(f"Drawing random example {n_c=} {linkage=} mt3scm={metrics['mt3scm']:.3f}")
        helpers.ax_scatter_3d( X[:, 0], X[:, 1], X[:, 2], ax, labels=y, subplot_title=f"({fig_titles[s_i]})", marker_size_array=marker_sizes, fontsize=8, set_ticks_and_labels=True)
        df_mets = pd.concat([df_mets, pd.DataFrame(metrics, index=[fig_titles[s_i]])])
    cols = ["davies", "calinski", "silhouette", "mt3scm", "cc", "wcc", "sl", "sp"]
    df_mets = df_mets[cols]
    la_style = df_mets.style
    la_style = la_style.format('{:.2G}')
    la_style.to_latex(f"{plot_name}.tex")
    print(f"Saving plot with name: {plot_name}")
    subplot_path = Path().cwd() / "plots"
    Path(subplot_path).mkdir(parents=True, exist_ok=True)
    plt.figure(1)
    plt.savefig(
        subplot_path / str(plot_name + "." + graphics_format),
        pad_inches=0,
        bbox_inches="tight",
        transparent=TRANSPARENT,
        dpi=resolution_dpi,
        format=graphics_format
    )
    plt.close(1)

def plot_random_examples(
    X: np.ndarray, dataset_name: str = "", n_clusters: list[int] = [2, 4, 10, 50, 200]
):
    min_max_seq_len = [(1, 2), (1, 100), (10, 20), (100, 500)]
    n_x_subfigs = 1
    n_x_subplots = len(min_max_seq_len)
    n_y_subfigs = len(n_clusters)
    n_y_subplots = 1
    alphabet_list = list(string.ascii_lowercase)
    subplot_labels = ["".join(x) for x in itertools.product(alphabet_list, repeat=2)]
    helpers.set_plot_params()
    fig = plt.figure(
        1, constrained_layout=True, figsize=(4 * n_x_subplots, 4 * n_y_subfigs)
    )
    # Setting global title
    fig.suptitle(
        r"\textbf{'Multivariate Time Series Sub-Sequence Clustering Metric' (MT3SCM) Evaluation}"
    )
    # Create subfigures for connectivity and number of clusters
    subfigs = fig.subfigures(n_y_subfigs, n_x_subfigs, squeeze=True)
    idx = 0
    result_index_names = ["max_n_sequences", "n_clusters", "min_seq_len", "max_seq_len"]
    df_metrics = pd.DataFrame()
    for subfig_index, n_c in enumerate(n_clusters):
        # Create subplots for all linkage variations
        axs = subfigs[subfig_index].subplots(
            n_y_subplots, n_x_subplots, subplot_kw=dict(projection="3d"), squeeze=True
        )
        subfigs[subfig_index].suptitle(f"Number of clusters: {n_c}")
        for subplots_index, (min_seq_len, max_seq_len) in enumerate(min_max_seq_len):
            # Generate random label sequences
            label_array = helpers.generate_random_sequences(
                length=X.shape[0],
                min_seq_length=min_seq_len,
                max_seq_length=max_seq_len,
                number_of_sequences=n_c,
            )
            metrics, kappa_X, tau_X = helpers.calc_unsupervised_metrics(X, label_array)
            n_unique_labels = len(np.unique(label_array))
            # create and collect the results in a dataframe for further analysis
            res_df = create_results_dataframe_from_dict(
                metrics,
                index=[[n_c], [n_unique_labels], [min_seq_len], [max_seq_len]],
                index_names=result_index_names,
            )
            df_metrics = pd.concat(
                [res_df, df_metrics],
                axis=0,
                verify_integrity=True,
                names=result_index_names,
            )
            subtitle = f"\\textbf{{Subfig. {subplot_labels[idx]}:}}, n_clusters={n_unique_labels}, min={min_seq_len}, max={max_seq_len}, \n\\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['sl']:.3f}, sp={metrics['sp']:.3f},\nsilhouette={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            # Scatter plot
            marker_sizes = np.log((np.abs(kappa_X * tau_X * 100) + 1) ** 2)
            print(f"Drawing random example {n_c=} {min_seq_len=} {max_seq_len=} mt3scm={metrics['mt3scm']:.3f}")
            helpers.ax_scatter_3d(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                axs[subplots_index],
                labels=label_array,
                subplot_title=subtitle,
                marker_size_array=marker_sizes,
            )
            # single_fig = plt.figure(2, constrained_layout=False, figsize=(4, 4))
            # ax = single_fig.add_subplot(projection="3d", computed_zorder=False)
            subtitle = f"n_clusters={n_unique_labels}, min={min_seq_len}, max={max_seq_len}, \\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['sl']:.3f}, sp={metrics['sp']:.3f},\nsilhouette={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            plot_name = f"ClusterMetricComparisonRandom-{dataset_name}-{subplot_labels[idx]}.{GRAPHICS_FORMAT}"
            plot_single_figure(
                X,
                labels=label_array,
                subtitle=subtitle,
                plot_name=plot_name,
                marker_sizes=marker_sizes,
            )
            idx += 1
    plot_name = f"ClusterMetricComparisonRandom-{dataset_name}.png"
    print(f"Saving plot with name: {plot_name}")
    plt.figure(1)
    plt.savefig(plot_name, dpi=300)
    plt.close(1)
    # Save resulting dataframe
    df_metrics.to_pickle(f"ClusterMetricComparisonResultsRandom-{dataset_name}.pkl")
    # df_metrics.to_csv(f"ClusterMetricComparisonResultsRandom-{dataset_name}.csv")


def plot_one_short_examples(
    X: np.ndarray, dataset_name: str = "", n_clusters: list[int] = [1, 2, 3, 4]
):
    min_max_seq_len = [(1, 2), (1, 10), (5, 10), (10, 50)]
    n_x_subfigs = 1
    n_x_subplots = len(min_max_seq_len)
    n_y_subfigs = len(n_clusters)
    n_y_subplots = 1
    alphabet_list = list(string.ascii_lowercase)
    subplot_labels = ["".join(x) for x in itertools.product(alphabet_list, repeat=2)]
    helpers.set_plot_params()
    fig = plt.figure(
        1, constrained_layout=True, figsize=(4 * n_x_subplots, 4 * n_y_subfigs)
    )
    # Setting global title
    fig.suptitle(
        r"\textbf{'Multivariate Time Series Sub-Sequence Clustering Metric' (MT3SCM) Evaluation}"
    )
    # Create subfigures for connectivity and number of clusters
    subfigs = fig.subfigures(n_y_subfigs, n_x_subfigs, squeeze=True)
    idx = 0
    result_index_names = ["max_n_sequences", "n_clusters", "min_seq_len", "max_seq_len"]
    df_metrics = pd.DataFrame()
    for subfig_index, n_c in enumerate(n_clusters):
        # Create subplots for all linkage variations
        axs = subfigs[subfig_index].subplots(
            n_y_subplots, n_x_subplots, subplot_kw=dict(projection="3d"), squeeze=True
        )
        subfigs[subfig_index].suptitle(f"Number of clusters: {n_c}")
        for subplots_index, (min_seq_len, max_seq_len) in enumerate(min_max_seq_len):
            # Generate random label sequences
            label_array = helpers.generate_one_random_sequence(
                length=X.shape[0],
                min_seq_length=min_seq_len,
                max_seq_length=max_seq_len,
                number_of_short_sequences=n_c,
            )
            metrics, kappa_X, tau_X = helpers.calc_unsupervised_metrics(X, label_array)
            n_unique_labels = len(np.unique(label_array))
            # create and collect the results in a dataframe for further analysis
            res_df = create_results_dataframe_from_dict(
                metrics,
                index=[[n_c], [n_unique_labels], [min_seq_len], [max_seq_len]],
                index_names=result_index_names,
            )
            df_metrics = pd.concat(
                [res_df, df_metrics],
                axis=0,
                verify_integrity=True,
                names=result_index_names,
            )
            subtitle = f"\\textbf{{Subfig. {subplot_labels[idx]}:}}, n_clusters={n_unique_labels}, min={min_seq_len}, max={max_seq_len}, \n\\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['sl']:.3f}, sp={metrics['sp']:.3f},\nsilhouette={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            # Scatter plot
            marker_sizes = np.log((np.abs(kappa_X * tau_X * 100) + 1) ** 2)
            print(f"Drawing random example {n_c=} {min_seq_len=} {max_seq_len=}")
            helpers.ax_scatter_3d(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                axs[subplots_index],
                labels=label_array,
                subplot_title=subtitle,
                marker_size_array=marker_sizes,
            )
            # single_fig = plt.figure(2, constrained_layout=False, figsize=(4, 4))
            # ax = single_fig.add_subplot(projection="3d", computed_zorder=False)
            subtitle = f"n_clusters={n_unique_labels}, min={min_seq_len}, max={max_seq_len}, \\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['sl']:.3f}, sp={metrics['sp']:.3f},\nsilhouette={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            plot_name = f"ClusterMetricComparisonShortRandom-{dataset_name}-{subplot_labels[idx]}.{GRAPHICS_FORMAT}"
            plot_single_figure(
                X,
                labels=label_array,
                subtitle=subtitle,
                plot_name=plot_name,
                marker_sizes=marker_sizes,
            )
            idx += 1
    plot_name = f"ClusterMetricComparisonShortRandom-{dataset_name}.png"
    print(f"Saving plot with name: {plot_name}")
    plt.figure(1)
    plt.savefig(plot_name, dpi=300)
    plt.close(1)
    # Save resulting dataframe
    df_metrics.to_pickle(f"ClusterMetricComparisonResultsShortRandom-{dataset_name}.pkl")
    # df_metrics.to_csv(f"ClusterMetricComparisonResultsRandom-{dataset_name}.csv")


def plot_single_figure(
    X: np.ndarray,
    labels: np.ndarray,
    subtitle: str,
    plot_name: str,
    marker_sizes: np.ndarray,
):
    single_fig = plt.figure(2, constrained_layout=False, figsize=(4, 4))
    ax = single_fig.add_subplot(projection="3d", computed_zorder=False)
    helpers.ax_scatter_3d(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        ax,
        labels=labels,
        subplot_title=subtitle,
        marker_size_array=marker_sizes,
    )
    plt.figure(2)
    subplot_path = Path().cwd() / "plots"
    Path(subplot_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        subplot_path / plot_name,
        pad_inches=0,
        bbox_inches="tight",
        transparent=TRANSPARENT,
        dpi=RESOLUTION_DPI,
        format=GRAPHICS_FORMAT,
    )
    plt.close(2)


def plot_agglomerative_clustering_example(
    X: np.ndarray,
    dataset_name: str = "",
    n_clusters: list[int] = [2, 3, 5, 10, 100, 1000],
    connect: list = [False],
    linkage_list: list[str] = ["average", "complete", "ward", "single"],
) -> None:
    if connect[0] is False:
        connect = [None, kneighbors_graph(X, 30, include_self=False, n_jobs=20)]
    n_subfigs = len(n_clusters) * len(connect)
    helpers.set_plot_params()
    fig = plt.figure(
        constrained_layout=True, figsize=(4 * len(linkage_list), 4 * n_subfigs)
    )
    # Setting global title
    fig.suptitle(
        r"\textbf{'Multivariate Time Series Sub-Sequence Clustering Metric' (MT3SCM) Evaluation}"
    )
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
            axs = subfigs[subfig_index][0].subplots(
                1, len(linkage_list), subplot_kw=dict(projection="3d"), squeeze=False
            )
            subfigs[subfig_index][0].suptitle(
                f"connectivity = {connectivity is not None}, n_clusters = {n_c}"
            )
            for subplots_index, linkage in enumerate(linkage_list):
                model = AgglomerativeClustering(
                    linkage=linkage, connectivity=connectivity, n_clusters=n_c
                )
                X_scaled = StandardScaler().fit_transform(X)
                model.fit(X_scaled)
                metrics, kappa_X, tau_X = helpers.calc_unsupervised_metrics(
                    X, model.labels_
                )
                # create and collect the results in a dataframe for further analysis
                res_df = create_results_dataframe_from_dict(
                    metrics,
                    index=[[connectivity is not None], [n_c], [linkage]],
                    index_names=result_index_names,
                )
                df_metrics = pd.concat(
                    [res_df, df_metrics],
                    axis=0,
                    verify_integrity=True,
                    names=result_index_names,
                )
                subtitle = f"\\textbf{{Subfig. {subplot_labels[idx]}:}} {linkage=},\n\\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['sl']:.3f}, sp={metrics['sp']:.3f},\nsil={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
                # Scatter plot
                marker_sizes = np.log((np.abs(kappa_X * tau_X * 100) + 1) ** 2)
                print(
                    f"Drawing agglomerative example connectivity={connectivity is not None} {n_c=} {linkage=}"
                )
                helpers.ax_scatter_3d(
                    X[:, 0],
                    X[:, 1],
                    X[:, 2],
                    axs[0][subplots_index],
                    labels=model.labels_,
                    subplot_title=subtitle,
                    marker_size_array=marker_sizes,
                )
                subtitle = f"n_clusters={n_c}, {linkage=}, connectivity={connectivity is not None}, \n\\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['sl']:.3f}, sp={metrics['sp']:.3f},\nsilhouette={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
                plot_name = f"ClusterMetricComparisonAgglomerative-{dataset_name}-{subplot_labels[idx]}.{GRAPHICS_FORMAT}"
                plot_single_figure(
                    X,
                    labels=model.labels_,
                    subtitle=subtitle,
                    plot_name=plot_name,
                    marker_sizes=marker_sizes,
                )
                idx += 1
            subfig_index += 1
    plot_name = f"ClusterMetricComparisonAgglomerative-{dataset_name}.png"
    print(f"Saving plot with name: {plot_name}")
    plt.figure(1)
    plt.savefig(plot_name, dpi=300)
    plt.close(1)
    # Save resulting dataframe
    df_metrics.to_pickle(
        f"ClusterMetricComparisonResultsAgglomerative-{dataset_name}.pkl"
    )
    df_metrics.to_csv(f"ClusterMetricComparisonResultsAgglomerative-{dataset_name}.csv")


def plot_kmeans_example(
    X: np.ndarray,
    dataset_name: str = "",
    n_clusters: list[int] = [2, 3, 5, 10, 100],
    methods: list[str] = ["euclidean", "dtw", "softdtw"],
) -> None:
    n_subfigs = len(n_clusters)
    n_subplots = len(methods)
    helpers.set_plot_params()
    fig = plt.figure(constrained_layout=True, figsize=(4 * n_subplots, 4 * n_subfigs))
    # Setting global title
    fig.suptitle(
        r"\textbf{'Multivariate Time Series Sub-Sequence Clustering Metric' (MT3SCM) Evaluation}"
    )
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
        axs = subfigs[subfig_index][0].subplots(
            1, n_subplots, subplot_kw=dict(projection="3d"), squeeze=False
        )
        subfigs[subfig_index][0].suptitle(f"n_clusters = {n_c}")
        for subplots_index, method in enumerate(methods):
            X_scaled = StandardScaler().fit_transform(X)
            model = TimeSeriesKMeans(
                n_clusters=n_c, metric=method, max_iter=5, random_state=42, n_jobs=-1
            ).fit(X_scaled)
            metrics, kappa_X, tau_X = helpers.calc_unsupervised_metrics(
                X, model.labels_
            )
            # create and collect the results in a dataframe for further analysis
            res_df = create_results_dataframe_from_dict(
                metrics, index=[[method], [n_c]], index_names=result_index_names
            )
            df_metrics = pd.concat(
                [res_df, df_metrics],
                axis=0,
                verify_integrity=True,
                names=result_index_names,
            )
            subtitle = f"\\textbf{{Subfig. {subplot_labels[idx]}:}} {method=},\n\\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['sl']:.3f}, sp={metrics['sp']:.3f},\nsil={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            # Scatter plot
            marker_sizes = np.log((np.abs(kappa_X * tau_X * 100) + 1) ** 2)
            print(f"Drawing TimeSeriesKMeans example connectivity={method=} {n_c=}")
            helpers.ax_scatter_3d(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                axs[0][subplots_index],
                labels=model.labels_,
                subplot_title=subtitle,
                marker_size_array=marker_sizes,
            )
            subtitle = f"{method=}, \\textbf{{mt3scm={metrics['mt3scm']:.3f}}},\ncc={metrics['cc']:.3f}, wcc={metrics['wcc']:.3f}, sl={metrics['sl']:.3f}, sp={metrics['sp']:.3f},\nsil={metrics['silhouette']:.3f}, calinski={metrics['calinski']:.1f}, davies={metrics['davies']:.3f}"
            plot_name = f"ClusterMetricComparisonTimeSeriesKMeans-{dataset_name}-{subplot_labels[idx]}.{GRAPHICS_FORMAT}"
            plot_single_figure(
                X,
                labels=model.labels_,
                subtitle=subtitle,
                plot_name=plot_name,
                marker_sizes=marker_sizes,
            )
            idx += 1
    plot_name = f"ClusterMetricComparisonTimeSeriesKMeans-{dataset_name}.png"
    print(f"Saving plot with name: {plot_name}")
    plt.figure(1)
    plt.savefig(plot_name, dpi=300)
    plt.close(1)
    # Save resulting dataframe
    df_metrics.to_pickle(
        f"ClusterMetricComparisonResultsTimeSeriesKMeans-{dataset_name}.pkl"
    )
    df_metrics.to_csv(
        f"ClusterMetricComparisonResultsTimeSeriesKMeans-{dataset_name}.csv"
    )


def create_results_dataframe_from_dict(
    metrics: dict[str, float], index: list, index_names: list[str]
) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(index, names=index_names)
    data = np.fromiter(metrics.values(), dtype=float)
    data = np.expand_dims(data, 0)
    cols = list(metrics.keys())
    df = pd.DataFrame(data, index=index, columns=cols)
    return df


def plot_examples(algorithms: list[str] = ["agglomerative", "random", "kmeans", "shortrandom"]):
    # Get lorenz attractor data as dataframe
    df_lorenz = helpers.generate_lorenz_attractor_data(dt=0.005, num_steps=3001)
    # Get thomas attractor data as dataframe
    df_thomas = helpers.generate_thomas_attractor_data(dt=0.05, num_steps=10000, b=0.1)
    # Get Synthetic dataset
    X_synth, labels_synth = helpers.gen_synth_data()
    datasets = {"lorenz": df_lorenz.values, "thomas": df_thomas.values, "synthetic": X_synth}
    n_clusters = [2, 4, 10, 50, 200]
    for name, data in datasets.items():
        if "agglomerative" in algorithms:
            plot_agglomerative_clustering_example(
                data, dataset_name=name, n_clusters=n_clusters
            )
        if "random" in algorithms:
            plot_random_examples(data, dataset_name=name, n_clusters=n_clusters)
        if "shortrandom" in algorithms:
            plot_one_short_examples(data, dataset_name=name)
        if "kmeans" in algorithms:
            plot_kmeans_example(data, dataset_name=name, n_clusters=n_clusters)


def plot_one_example():
    # Get lorenz attractor data as dataframe
    df_lorenz = helpers.generate_lorenz_attractor_data(dt=0.005, num_steps=3001)
    X_lorenz = StandardScaler().fit_transform(df_lorenz.values)
    # Get thomas attractor data as dataframe
    df_thomas = helpers.generate_thomas_attractor_data(dt=0.05, num_steps=10000, b=0.1)
    X_thomas = StandardScaler().fit_transform(df_thomas.values)
    # datasets = {"thomas_single_example": X_thomas, "lorenz_single_example": X_lorenz}
    datasets = {"lorenz_single_example": X_lorenz}
    for name, data in datasets.items():
        plot_agglomerative_clustering_example(
            data,
            dataset_name=name,
            n_clusters=[3],
            connect=[kneighbors_graph(data, 30, include_self=False, n_jobs=-1)],
            linkage_list=["average"],
        )


def create_3d_figure(df: pd.DataFrame, xyz_labels: list[str] = ["x", "y", "z"], predicted_class_array: np.ndarray = None, plot_changepoints: bool = True):
    fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    X = df.values
    mt3 = MT3SCM()
    if predicted_class_array is None:
        predicted_class_array = np.ones(X.shape[0])
        predicted_class_array[0] = 0
    _ = mt3.mt3scm_score(X, predicted_class_array, standardize_subs_curve=True)
    kappa = np.log((np.abs(mt3.kappa_X * 1) + 1) ** 1)
    helpers.ax_scatter_3d_original(X[:, 0], X[:, 1], X[:, 2], ax, kappa=kappa, xyz_labels=xyz_labels, subplot_title=None)
    return fig


def plot_lorenz_attractor_3d(plot_name: str = "lorenz-attractor-3d", graphics_format: str = GRAPHICS_FORMAT, resolution_dpi: int = RESOLUTION_DPI):
    df = helpers.generate_lorenz_attractor_data()
    helpers.set_plot_params()
    fig = create_3d_figure(df)
    subplot_path = Path().cwd() / "plots"
    Path(subplot_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        subplot_path / str(plot_name + "." + graphics_format),
        pad_inches=0,
        bbox_inches="tight",
        transparent=TRANSPARENT,
        dpi=resolution_dpi,
        format=graphics_format
    )
    # fig.savefig(subplot_path / plot_name, pad_inches=0, bbox_inches="tight", transparent=TRANSPARENT, dpi=RESOLUTION_DPI, format=GRAPHICS_FORMAT)
    plt.close(fig)
    print(f"Saving plot with name: {plot_name}")


def plot_curvature_torsion_example(plot_name: str = "qualitative_curve_parameters", graphics_format: str = GRAPHICS_FORMAT, resolution_dpi: int = RESOLUTION_DPI):
    helpers.set_plot_params()
    df_thomas = helpers.generate_thomas_attractor_data(dt=0.05, num_steps=500, b=0.1)
    X = StandardScaler().fit_transform(df_thomas.values)
    mt3 = MT3SCM()
    kappa, tau, speed, acceleration = mt3.compute_curvature(X)
    curv_data = {
        "Curvature": kappa,
        "Torsion": tau,
        "Speed": speed,
        "Acceleration": acceleration,
    }
    n_y_subplots = 2
    n_x_subplots = 2
    fig_titles = string.ascii_lowercase[:(n_y_subplots * n_x_subplots)]
    helpers.set_plot_params()
    # idx = 0
    # result_index_names = ["max_n_sequences", "n_clusters", "min_seq_len", "max_seq_len"]
    cm = 1/2.54
    fig, axs = plt.subplots(n_y_subplots, n_x_subplots, subplot_kw=dict(projection="3d"), constrained_layout=True, figsize=(17.75*cm*n_x_subplots/2, 17.75*cm*n_y_subplots/2), squeeze=False)
    for s_i, (ax, (name, data)) in enumerate(zip(axs.flat, curv_data.items())):
        helpers.ax_scatter_3d_original(X[:, 0], X[:, 1], X[:, 2], ax, kappa=data, line_scaling_factor=100, set_ticks_and_labels=False, pad=0.0, subplot_title=f"({fig_titles[s_i]})")
    print(f"Saving plot with name: {plot_name}")
    subplot_path = Path().cwd() / "plots"
    Path(subplot_path).mkdir(parents=True, exist_ok=True)
    plt.figure(1)
    plt.savefig(
        subplot_path / str(plot_name + "." + graphics_format),
        bbox_inches="tight",
        transparent=TRANSPARENT,
        dpi=resolution_dpi,
        format=graphics_format
    )
    plt.close(1)
    # Plot single figures
    # for subplots_index, (name, data) in enumerate(curv_data.items()):
    #     fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    #     ax = fig.add_subplot(projection="3d", computed_zorder=False)
    #     helpers.ax_scatter_3d_original(X[:, 0], X[:, 1], X[:, 2], ax, kappa=data, line_scaling_factor=100, set_ticks_and_labels=False, pad=0.0)
    #     plot_name = f"{name}.{GRAPHICS_FORMAT}"
    #     subplot_path = Path().cwd() / "plots"
    #     Path(subplot_path).mkdir(parents=True, exist_ok=True)
    #     fig.savefig(subplot_path / plot_name, pad_inches=0.01, bbox_inches="tight", transparent=TRANSPARENT, dpi=RESOLUTION_DPI, format=GRAPHICS_FORMAT)
    #     plt.close(fig)

if __name__ == "__main__":
    # Standard Libraries Import
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--one-only",
        dest="one",
        help="plot only a single example",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--curve",
        dest="curve",
        help="plot  curvature torsion example",
        action="store_true",
    )
    parser.add_argument(
        "-k", "--kmeans", dest="kmeans", help="plot kmeans example", action="store_true"
    )
    parser.add_argument(
        "-s", "--short", dest="short", help="plot short example", action="store_true"
    )
    parser.add_argument(
        "-a", "--agglomerative", dest="agglomerative", help="plot agglomerative example", action="store_true"
    )
    parser.add_argument(
        "-r", "--random", dest="random", help="plot random example", action="store_true"
    )
    args = parser.parse_args()
    if args.one is True:
        print(f"Plotting a single example only..")
        # plot_one_example()
        publication_plot_agglomerative_examples()
        # publication_plot_random_examples()
    if args.curve is True:
        print(f"Plotting curvature torsion example..")
        plot_curvature_torsion_example()
        plot_lorenz_attractor_3d()
    if args.kmeans is True:
        print(f"Plotting examples for kmeans..")
        plot_examples(algorithms=["kmeans"])
    if args.agglomerative is True:
        print(f"Plotting examples for agglomerative..")
        plot_examples(algorithms=["agglomerative"])
    if args.random is True:
        print(f"Plotting examples for random..")
        plot_examples(algorithms=["random"])
    if args.short is True:
        print(f"Plotting one short example..")
        plot_examples(algorithms=["shortrandom"])
    # publication_plot_agglomerative_examples()
    print(f"Done")
