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
from helpers import gen_synth_data
from mt3scm import MT3SCM

RESOLUTION_DPI = 300
TRANSPARENT = False
GRAPHICS_FORMAT = "png"  # or png, pdf, svg

def plot_testing_results(X: np.ndarray, score: float, labels: np.ndarray, test_name: str):
    # Third Party Libraries Import
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
    # fig.suptitle(f"\\textbf{{MT3SCM: {test_name}}}")
    davies = davies_bouldin_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    silhouette = silhouette_score(X, labels)
    # subplot_title = f"mt3scm={score:.2n}, {davies=:.2n}, {calinski=:.2n}, {silhouette=:.2n}"
    subplot_title = f"{davies=:.2n}, {calinski=:.2n}, {silhouette=:.2n}"
    if X.shape[1] >= 3:
        ax = fig.add_subplot(projection="3d")
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
        ax.set_zlabel(z_label)
    elif X.shape[1] == 2:
        ax = fig.add_subplot()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels)
    ax.set_title(subplot_title)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Clusters")
    ax.add_artist(legend1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    test_plots_path: Path = Path("test_plots")
    Path(test_plots_path).mkdir(parents=True, exist_ok=True)
    full_plot_name: Path = test_plots_path / str(test_name + ".png")
    plt.tight_layout()
    plt.savefig(
        full_plot_name,
        pad_inches=0,
        bbox_inches="tight",
        transparent=TRANSPARENT,
        dpi=RESOLUTION_DPI,
        format=GRAPHICS_FORMAT,
    )
    plt.close()


def main():
    X, labels = gen_synth_data()
    # Score calculation and plotting
    sil = silhouette_score(X, labels)
    print(f"X {sil=}")
    mt3 = MT3SCM()
    mt3scm_metric = mt3.mt3scm_score(X, labels, edge_offset=5)
    plot_curvature(mt3.kappa_X, file_name="kappa")
    plot_curvature(mt3.tau_X, file_name="tau")
    plot_testing_results(X, mt3scm_metric, labels, "constant-curvature-data-example.png")
    print(f"{mt3scm_metric=:.2n}\n{mt3.cc=:.2n}, {mt3.wcc=:.2n}, {mt3.masc_pos=:.2n}, {mt3.masc_kt=:.2n}")
    print(f"{mt3.df_centers=}")
    print(f"{mt3.df_curve=}")
    print(f"{mt3.cccs=}")
    labels = mt3.df_centers.index.get_level_values("c_id")
    plot_testing_results(mt3.df_centers.values, mt3scm_metric, labels, "subsequence-centers.png")
    sil = silhouette_score(mt3.df_centers.values, labels)
    print(f"df_centers {sil=:.2n}")
    labels = mt3.df_curve.index.get_level_values("c_id")
    plot_testing_results(mt3.df_curve.values, mt3scm_metric, labels, "subsequence-curve-parameteres.png")
    sil = silhouette_score(mt3.df_curve.values, labels)
    print(f"df_curve {sil=:.2n}")


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
