"""Create synthetic example plots"""

# Author: Jonas Köhne <jokohonas@gmail.com>
# License: BSD 3 clause

# Standard Libraries Import
from pathlib import Path

# Third Party Libraries Import
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Own Libraries Import
from helpers import gen_synth_data, set_plot_params
from mt3scm import MT3SCM
import helpers
import string
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering


RESOLUTION_DPI = 300
TRANSPARENT = False
GRAPHICS_FORMAT = "pdf"  # or png, pdf, svg

def scatter_plot(X, ax, x_label, y_label, z_label, labels: np.ndarray, autorotate_labels: bool = True, subplot_title: str = None, loc:str="best", marker_size:float=10.0, legend_title:str="Cluster", framealpha:float = 0.6):
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, s=marker_size)
    ax.set_title(subplot_title)
    legend1 = ax.legend(*scatter.legend_elements(), loc=loc, title=legend_title, framealpha=framealpha)
    ax.add_artist(legend1)
    ax.zaxis.set_rotate_label(autorotate_labels)
    ax.yaxis.set_rotate_label(autorotate_labels)
    ax.xaxis.set_rotate_label(autorotate_labels)
    ax.set_zlabel(z_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax

def publication_plot_metric_perfect():
    X, labels = gen_synth_data()
    mt3 = MT3SCM()
    kappa, tau, speed, acceleration = mt3.compute_curvature(X)
    X_all_curve_params = np.array([kappa, tau, acceleration]).T
    # Score calculation and plotting
    sil = silhouette_score(X, labels)
    print(f"X {sil=}")
    mt3scm_metric = mt3.mt3scm_score(X, labels, edge_offset=5)
    labels_centers = mt3.df_centers.index.get_level_values("c_id")
    labels_curve = mt3.df_curve.index.get_level_values("c_id")
    print(f"Plotting koehn7..")
    medians = mt3.df_centers.values
    curves = mt3.df_curve.values
    data = [X, medians, curves]
    y = [labels, labels_centers, labels_curve]
    feat_names = [
        ["x", "y", "z"],
        ["x", "y", "z"],
        [r"$\overline{\kappa}$", r"$\overline{\tau}$", r"$\overline{a}$"]
    ]
    markers_size = [10, 100, 100]
    n_x_subfigs = 3
    n_y_subfigs = 1
    n_y_subplots = 1
    n_x_subplots = 1
    fig_titles = string.ascii_lowercase[:n_x_subfigs]
    helpers.set_plot_params()
    # textwidth of two column paper: 17.75cm
    cm = 1/2.54
    fig = plt.figure(1, constrained_layout=False, figsize=(17.75*cm * 2, 17.75*cm / 1.5))
    # Create subfigures for connectivity and number of clusters
    subfigs = fig.subfigures(n_y_subfigs, n_x_subfigs, squeeze=True)
    for subfig_index in range(n_x_subfigs * n_y_subfigs):
        # Create subplots for all linkage variations
        axs = subfigs[subfig_index].subplots(n_y_subplots, n_x_subplots, subplot_kw=dict(projection="3d"), squeeze=True)
        subfigs[subfig_index].suptitle(f"({fig_titles[subfig_index]})", fontsize=8)
        ax = scatter_plot(data[subfig_index], axs, feat_names[subfig_index][0], feat_names[subfig_index][1], feat_names[subfig_index][2], y[subfig_index], autorotate_labels=False, marker_size=markers_size[subfig_index])
    plot_name = f"koehn7.pdf"
    print(f"Saving plot with name: {plot_name}")
    plt.figure(1)
    plt.savefig(plot_name, dpi=300)
    plt.close(1)

def publication_plot_metric_agglom_example():
    df_lorenz = helpers.generate_lorenz_attractor_data(dt=0.005, num_steps=3001, scale_zs=10.0)
    X = df_lorenz.values
    mt3 = MT3SCM(include_std_num_points=False)
    kappa, tau, _, acceleration = mt3.compute_curvature(X)
    X_all_curve_params = np.array([acceleration, kappa, tau]).T
    clustering = AgglomerativeClustering(n_clusters=10).fit(X_all_curve_params)
    labels = clustering.labels_
    _ = mt3.mt3scm_score(X, labels, edge_offset=0, n_min_subs=1)
    labels_centers = mt3.df_centers.index.get_level_values("c_id")
    labels_curve = mt3.df_curve.index.get_level_values("c_id")
    print(f"Plotting koehn10..")
    medians = mt3.df_centers.values
    curves = mt3.df_curve.values
    data = [X_all_curve_params, X, curves, medians]
    y = [labels, labels, labels_curve, labels_centers]
    feat_names = [
        [r"$\kappa$", r"$\tau$", r"$a$"],
        ["x", "y", "z"],
        [r"$\overline{\kappa}$", r"$\overline{\tau}$", r"$\overline{a}$"],
        ["x", "y", "z"],
    ]
    markers_size = [100, 10, 100, 100]
    n_y_subplots = 2
    n_x_subplots = 2
    fig_titles = string.ascii_lowercase[:(n_y_subplots * n_x_subplots)]
    helpers.set_plot_params()
    # textwidth of two column paper: 17.75cm
    cm = 1/2.54
    fig, axs = plt.subplots(n_y_subplots, n_x_subplots, subplot_kw=dict(projection="3d"), constrained_layout=True, figsize=(17.75*cm, 17.75*cm), squeeze=False)
    plt.subplots_adjust(wspace=0.2,hspace=0.2)
    for s_i, ax in enumerate(axs.flat):
        metrics, _, _ = helpers.calc_unsupervised_metrics(data[s_i], y[s_i], edge_offset=5, n_min_subs=1)
        print(f"Plotted plot {fig_titles[s_i]} with metrics davies-bouldin: {metrics['davies']:.2n}, calinski-harabasz: {metrics['calinski']:.2n}, silhouette: {metrics['silhouette']:.2n}, mt3scm: {metrics['mt3scm']:.2n}, cc={metrics['cc']:.2n}, wcc={metrics['wcc']:.2n},sl= {metrics['sl']:.2n}, sp={metrics['sp']:.2n}")
        ax = scatter_plot(data[s_i], ax, feat_names[s_i][0], feat_names[s_i][1], feat_names[s_i][2], y[s_i], autorotate_labels=False, marker_size=markers_size[s_i])
        ax.set_title(f"({fig_titles[s_i]})", fontsize=8)
        ax.tick_params(labelsize=8)
    plot_name = f"koehn10.pdf"
    print(f"Saving plot with name: {plot_name}")
    plt.figure(1)
    # plt.tight_layout(h_pad=8)
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close(1)

def plot_testing_results(X: np.ndarray, score: float, labels: np.ndarray, test_name: str, marker_size:float=10.0, fig_suptitle:str=None, subplot_title:str=None, loc:str="best", legend_title:str="Cluster", feature_names:list[str]=["x", "y", "z"], autorotate_labels: bool =True):
    x_label, y_label, z_label = feature_names
    set_plot_params()
    fig = plt.figure(
        1, constrained_layout=True, figsize=(4, 4)
    )
    if fig_suptitle is not None:
        fig.suptitle(fig_suptitle)
    davies = davies_bouldin_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    silhouette = silhouette_score(X, labels)
    if subplot_title is None:
        subplot_title = f"{davies=:.2n}, {calinski=:.2n}, {silhouette=:.2n}"
    ax = fig.add_subplot(projection="3d")
    ax = scatter_plot(X, ax, x_label, y_label, z_label, labels)
    test_plots_path: Path = Path("test_plots")
    Path(test_plots_path).mkdir(parents=True, exist_ok=True)
    full_plot_name: Path = test_plots_path / str(test_name + "." + GRAPHICS_FORMAT)
    # plt.tight_layout()
    plt.savefig(
        full_plot_name,
        pad_inches=0,
        bbox_inches="tight",
        transparent=TRANSPARENT,
        dpi=RESOLUTION_DPI,
        format=GRAPHICS_FORMAT,
    )
    print(f"Plotted and saved: {test_name} with metrics: {davies=:.2n}, {calinski=:.2n}, {silhouette=:.2n}")
    plt.close()

def plot_agglomerative_lorenz_on_new_feature_space():
    df_lorenz = helpers.generate_lorenz_attractor_data(dt=0.005, num_steps=3001)
    X = df_lorenz.values
    mt3 = MT3SCM()
    kappa, tau, speed, acceleration = mt3.compute_curvature(X)
    X_all_curve_params = np.array([acceleration, kappa, tau]).T
    clustering = AgglomerativeClustering(n_clusters=10).fit(X_all_curve_params)
    labels = clustering.labels_
    mt3scm_metric = mt3.mt3scm_score(X, labels, edge_offset=0, n_min_subs=1)

    davies = davies_bouldin_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    silhouette = silhouette_score(X, labels)
    subplot_title = f"mt3scm={mt3scm_metric:.2n}\ncc={mt3.cc:.2n}, wcc={mt3.wcc:.2n},sl= {mt3.masc_pos:.2n}, sp={mt3.masc_kt:.2n}\n{davies=:.2n}, {calinski=:.2n}, {silhouette=:.2n}"
    plot_testing_results(X, mt3scm_metric, labels, "agglomerative_lorenz-feature-space-example")
    labels_centers = mt3.df_centers.index.get_level_values("c_id")
    feature_names = mt3.df_centers.columns.to_list()[2:]
    # feature_names = mt3.df_centers.columns.to_list()
    plot_testing_results(mt3.df_centers.values, mt3scm_metric, labels_centers, "agglomerative_lorenz-feature-space-centers", marker_size=100, feature_names=feature_names)
    labels_curve = mt3.df_curve.index.get_level_values("c_id")
    feature_names = mt3.df_curve.columns.to_list()[2:]
    # feature_names = mt3.df_curve.columns.to_list()
    plot_testing_results(mt3.df_curve.values, mt3scm_metric, labels_curve, "agglomerative_lorenz-feature-space-curve-parameteres", marker_size=100, loc="best", feature_names=feature_names)
    subplot_title = f"mt3scm={mt3scm_metric:.2n}\ncc={mt3.cc:.2n}, wcc={mt3.wcc:.2n},sl= {mt3.masc_pos:.2n}, sp={mt3.masc_kt:.2n}"
    plot_testing_results(X_all_curve_params, mt3scm_metric, labels, "agglomerative_lorenz-feature-space-allcurve-params", marker_size=100, feature_names=["acceleration", "curvature", "torsion"], loc="best", subplot_title=subplot_title)
    print(subplot_title)

def plot_agglomerative_own_synth_on_new_feature_space():
    # Third Party Libraries Import
    from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering

    # Own Libraries Import
    import helpers
    df = helpers.gen_curve_synth_data()

    X = df.values
    true_labels = df.index.get_level_values("label")
    mt3 = MT3SCM()
    kappa, tau, speed, acceleration = mt3.compute_curvature(X)
    X_all_curve_params = np.array([acceleration, kappa, tau]).T
    # clustering = AgglomerativeClustering(n_clusters=4).fit(X_all_curve_params)
    # labels = clustering.labels_
    labels = true_labels
    mt3scm_metric = mt3.mt3scm_score(X, labels, edge_offset=0, n_min_subs=1)

    davies = davies_bouldin_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    silhouette = silhouette_score(X, labels)
    subplot_title = f"mt3scm={mt3scm_metric:.2n}\ncc={mt3.cc:.2n}, wcc={mt3.wcc:.2n},sl= {mt3.masc_pos:.2n}, sp={mt3.masc_kt:.2n}\n{davies=:.2n}, {calinski=:.2n}, {silhouette=:.2n}"
    plot_testing_results(X, mt3scm_metric, labels, "agglomerative_ownsynth-feature-space-example")
    labels_centers = mt3.df_centers.index.get_level_values("c_id")
    plot_testing_results(mt3.df_centers.values, mt3scm_metric, labels_centers, "agglomerative_ownsynth-feature-space-centers", marker_size=100, feature_names=mt3.df_centers.columns.to_list()[2:])
    labels_curve = mt3.df_curve.index.get_level_values("c_id")
    plot_testing_results(mt3.df_curve.values, mt3scm_metric, labels_curve, "agglomerative_ownsynth-feature-space-curve-parameteres", marker_size=100, loc="best", feature_names=mt3.df_curve.columns.to_list()[2:])
    subplot_title = f"mt3scm={mt3scm_metric:.2n}\ncc={mt3.cc:.2n}, wcc={mt3.wcc:.2n},sl= {mt3.masc_pos:.2n}, sp={mt3.masc_kt:.2n}"
    plot_testing_results(X_all_curve_params, mt3scm_metric, labels, "agglomerative_ownsynth-feature-space-allcurve-params", marker_size=100, feature_names=["acceleration", "curvature", "torsion"], loc="best", subplot_title=subplot_title)
    print(subplot_title)

def generate_graphics_for_publication():
    X, labels = gen_synth_data()
    mt3 = MT3SCM()
    kappa, tau, speed, acceleration = mt3.compute_curvature(X)
    X_all_curve_params = np.array([kappa, tau, acceleration]).T
    # Score calculation and plotting
    sil = silhouette_score(X, labels)
    print(f"X {sil=}")
    mt3scm_metric = mt3.mt3scm_score(X, labels, edge_offset=5)
    labels_centers = mt3.df_centers.index.get_level_values("c_id")
    labels_curve = mt3.df_curve.index.get_level_values("c_id")
    print(f"Plotting koehn7..")
    plot_testing_results(X, mt3scm_metric, labels, "constant-curvature-data-example", subplot_title="", legend_title="Cluster", autorotate_labels=False)
    print(f"Plotting koehn8..")
    plot_testing_results(mt3.df_centers.values, mt3scm_metric, labels_centers, "subsequence-centers", marker_size=100, subplot_title="", legend_title="Cluster", autorotate_labels=False)
    print(f"Plotting koehn9..")
    feat_names = [r"$\overline{\kappa}$", r"$\overline{\tau}$", r"$\overline{a}$"]
    plot_testing_results(mt3.df_curve.values, mt3scm_metric, labels_curve, "subsequence-curve-parameteres", marker_size=100, feature_names=feat_names, subplot_title="", legend_title="Cluster", autorotate_labels=False)

def main():
    X, labels = gen_synth_data()
    mt3 = MT3SCM()
    kappa, tau, speed, acceleration = mt3.compute_curvature(X)
    X_all_curve_params = np.array([kappa, tau, acceleration]).T
    # Score calculation and plotting
    sil = silhouette_score(X, labels)
    print(f"X {sil=}")
    mt3scm_metric = mt3.mt3scm_score(X, labels, edge_offset=5)

    plot_curvature(mt3.kappa_X, file_name="kappa")
    plot_curvature(mt3.tau_X, file_name="tau")
    plot_curvature(mt3.acceleration_X, file_name="acceleration")
    plot_testing_results(X, mt3scm_metric, labels, "constant-curvature-data-example")
    print(f"{mt3scm_metric=:.2n}\n{mt3.cc=:.2n}, {mt3.wcc=:.2n}, {mt3.masc_pos=:.2n}, {mt3.masc_kt=:.2n}")
    print(f"{mt3.df_centers=}")
    print(f"{mt3.df_curve=}")
    print(f"{mt3.cccs=}")
    labels_centers = mt3.df_centers.index.get_level_values("c_id")
    # plot_testing_results(mt3.df_centers.values, mt3scm_metric, labels_centers, "subsequence-centers", marker_size=100, feature_names=mt3.df_centers.columns.to_list())
    plot_testing_results(mt3.df_centers.values, mt3scm_metric, labels_centers, "subsequence-centers", marker_size=100, feature_names=mt3.df_centers.columns.to_list()[2:])
    sil = silhouette_score(mt3.df_centers.values, labels_centers)
    print(f"df_centers {sil=:.2n}")
    labels_curve = mt3.df_curve.index.get_level_values("c_id")
    # plot_testing_results(mt3.df_curve.values, mt3scm_metric, labels_curve, "subsequence-curve-parameteres", marker_size=100, feature_names=mt3.df_curve.columns.to_list())
    plot_testing_results(mt3.df_curve.values, mt3scm_metric, labels_curve, "subsequence-curve-parameteres", marker_size=100, feature_names=mt3.df_curve.columns.to_list()[2:])
    subplot_title = f"mt3scm={mt3scm_metric:.2n}\ncc={mt3.cc:.2n}, wcc={mt3.wcc:.2n},sl= {mt3.masc_pos:.2n}, sp={mt3.masc_kt:.2n}"
    plot_testing_results(X_all_curve_params, mt3scm_metric, labels, "subsequence-allcurve-params", marker_size=100, feature_names=["curvature", "torsion", "acceleration"], loc="best", subplot_title=subplot_title)
    sil = silhouette_score(mt3.df_curve.values, labels_curve)
    print(f"df_curve {sil=:.2n}")
    print(f"{mt3scm_metric=:.2n}\n{mt3.cc=:.2n}, {mt3.wcc=:.2n}, {mt3.masc_pos=:.2n}, {mt3.masc_kt=:.2n}")


def plot_curvature(X, file_name: str = "curvature"):
    feature_names = ["x", "y", "z"]
    x_label, y_label, z_label = feature_names
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Computer Modern Serif"],
            "axes.grid": False,
        }
    )
    fig = plt.figure()
    fig.suptitle(f"\\textbf{{MT3SCM: {file_name}}}")
    ax = fig.add_subplot()
    scatter = ax.plot(X)
    test_plots_path: Path = Path("test_plots")
    Path(test_plots_path).mkdir(parents=True, exist_ok=True)
    full_plot_name: Path = test_plots_path / str(file_name + ".png")
    plt.savefig(full_plot_name)
    plt.close()


def curvature(X: np.ndarray):
    """Curvature calculation for testing purposes with slightly different formula. Computes the same result as the MT3SCM class. Only Curvature calculation. No torsion.

    Args:
        X (np.ndarray): numpy array of shape (n_samples, n_features)
    """
    g1 = np.gradient(X, axis=0)
    g2 = np.gradient(g1, axis=0)
    g3 = np.gradient(g2, axis=0)
    # This is the formula for curvature $\kappa$ in n-dimensional space and without looping over time
    first = np.linalg.norm(g1, axis=1) ** 2
    second = np.linalg.norm(g2, axis=1) ** 2
    third = np.einsum("ij,ij->i", g1, g2) ** 2
    top = np.sqrt((first * second) - (third))
    bottom = np.linalg.norm(g1, axis=1) ** 3
    kappa = np.divide(top, bottom)
    plt.plot(np.array(kappa))
    plt.savefig("curvature-test.png")
    plt.close()


if __name__ == "__main__":
    publication_plot_metric_perfect()
    publication_plot_metric_agglom_example()
    # generate_graphics_for_publication()
    # main()
    # plot_agglomerative_lorenz_on_new_feature_space()
    # plot_agglomerative_own_synth_on_new_feature_space()