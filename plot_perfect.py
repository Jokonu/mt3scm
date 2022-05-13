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

RESOLUTION_DPI = 300
TRANSPARENT = False
GRAPHICS_FORMAT = "png"  # or png, pdf, svg

def plot_testing_results(X: np.ndarray, score: float, labels: np.ndarray, test_name: str, marker_size:float=10.0, fig_suptitle:str=None, subplot_title:str=None, loc:str="upper left", legend_title:str="Clusters", feature_names:list[str]=["x", "y", "z"]):
    x_label, y_label, z_label = feature_names
    set_plot_params()
    fig = plt.figure(
        1, constrained_layout=True, figsize=(4, 4)
    )
    if fig_suptitle is not None:
        fig.suptitle(fig_suptitle)
    if subplot_title is None:
        davies = davies_bouldin_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        silhouette = silhouette_score(X, labels)
        subplot_title = f"{davies=:.2n}, {calinski=:.2n}, {silhouette=:.2n}"
    if X.shape[1] >= 3:
        ax = fig.add_subplot(projection="3d")
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, s=marker_size)
        ax.set_zlabel(z_label)
    elif X.shape[1] == 2:
        ax = fig.add_subplot()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels)
    ax.set_title(subplot_title)
    legend1 = ax.legend(*scatter.legend_elements(), loc=loc, title=legend_title)
    ax.add_artist(legend1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    test_plots_path: Path = Path("test_plots")
    Path(test_plots_path).mkdir(parents=True, exist_ok=True)
    full_plot_name: Path = test_plots_path / str(test_name + ".png")
    # plt.tight_layout()
    plt.savefig(
        full_plot_name,
        pad_inches=0,
        bbox_inches="tight",
        transparent=TRANSPARENT,
        dpi=RESOLUTION_DPI,
        format=GRAPHICS_FORMAT,
    )
    plt.close()

def plot_agglomerative_lorenz_on_new_feature_space():
    import helpers
    from sklearn.cluster import OPTICS, DBSCAN, AgglomerativeClustering
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
    plot_testing_results(mt3.df_centers.values, mt3scm_metric, labels_centers, "agglomerative_lorenz-feature-space-centers", marker_size=100, feature_names=mt3.df_centers.columns.to_list())
    labels_curve = mt3.df_curve.index.get_level_values("c_id")
    plot_testing_results(mt3.df_curve.values, mt3scm_metric, labels_curve, "agglomerative_lorenz-feature-space-curve-parameteres", marker_size=100, loc="best", feature_names=mt3.df_curve.columns.to_list())
    subplot_title = f"mt3scm={mt3scm_metric:.2n}\ncc={mt3.cc:.2n}, wcc={mt3.wcc:.2n},sl= {mt3.masc_pos:.2n}, sp={mt3.masc_kt:.2n}"
    plot_testing_results(X_all_curve_params, mt3scm_metric, labels, "agglomerative_lorenz-feature-space-allcurve-params", marker_size=100, feature_names=["acceleration", "curvature", "torsion"], loc="best", subplot_title=subplot_title)
    print(subplot_title)


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
    plot_testing_results(mt3.df_centers.values, mt3scm_metric, labels_centers, "subsequence-centers", marker_size=100, feature_names=mt3.df_centers.columns.to_list())
    sil = silhouette_score(mt3.df_centers.values, labels_centers)
    print(f"df_centers {sil=:.2n}")
    labels_curve = mt3.df_curve.index.get_level_values("c_id")
    plot_testing_results(mt3.df_curve.values, mt3scm_metric, labels_curve, "subsequence-curve-parameteres", marker_size=100, feature_names=mt3.df_curve.columns.to_list())
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
    main()
    plot_agglomerative_lorenz_on_new_feature_space()
