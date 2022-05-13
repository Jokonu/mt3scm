"""
Comparison of unsupervised clustering metrics
=============================================

This python script analysis the results from the examples

"""

# Authors: Jonas Köhne
# License: BSD 3 clause

from pathlib import Path

# Third Party Libraries Import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm as cm
import seaborn as sns

# Own Libraries Import
import helpers


RESOLUTION_DPI = 300
TRANSPARENT = False
GRAPHICS_FORMAT = "png"  # or png, pdf, svg


def set_plot_style():
    plt.style.use("plot_style.txt")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": ["Computer Modern Roman"],
            "axes.grid": False,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
        }
    )

def load_results_into_dataframe(
    dataset_names: list[str] = ["lorenz", "thomas", "synthetic"],
    algo_type_names: list[str] = ["Random", "Agglomerative", "TimeSeriesKMeans"],
):
    dfs = []
    full_index_set = set()
    for dataset_name in dataset_names:
        for algo_type_name in algo_type_names:
            df = pd.read_pickle(f"ClusterMetricComparisonResults{algo_type_name}-{dataset_name}.pkl")
            full_index_set.update(df.index.names)
            df = df[["mt3scm", "cc", "wcc", "sl", "sp", "silhouette", "davies", "calinski"]]
            df["dataset"] = dataset_name
            df["algorithm"] = algo_type_name
            dfs.append(df.reset_index())
    df_all = pd.concat(dfs, axis=0)
    full_index_set.update(["dataset", "algorithm"])
    df_all = df_all.set_index(list(full_index_set)).sort_index()
    df_all = df_all[["mt3scm", "cc", "wcc", "sl", "sp", "silhouette", "davies", "calinski"]]
    return df_all


def analyse_metrics():
    set_plot_style()
    df = load_results_into_dataframe()
    # df["wccsp"] = df["wcc"] + df["sp"]
    df.groupby(by=["algorithm", "dataset"]).max()
    df = df[["mt3scm", "silhouette", "davies", "calinski"]]
    df = df.xs(["Agglomerative", "lorenz"], level=["algorithm", "dataset"])
    df=(df-df.min())/(df.max()-df.min())
    df_mets = df.melt(value_vars=df.columns, var_name="metric", value_name="value")
    ax = sns.violinplot(x="metric", y="value", data=df_mets)
    subplot_path = Path().cwd() / "plots"
    plot_name = f"metrics_violin.{GRAPHICS_FORMAT}"
    plt.savefig(
        subplot_path / plot_name,
        pad_inches=0,
        bbox_inches="tight",
        transparent=TRANSPARENT,
        dpi=RESOLUTION_DPI,
        format=GRAPHICS_FORMAT,
    )
    plt.close()


def analyse_correlation():
    set_plot_style()
    df = load_results_into_dataframe()
    df_corr = df.corr()
    A = df_corr.values
    mask = np.tri(A.shape[0], k=0)
    A = np.ma.array(A, mask=np.logical_not(mask))  # mask out the lower triangle
    cmap = cm.get_cmap("YlGn").copy()  # jet doesn't have white color
    cmap.set_bad("w")  # default value is 'k'
    fig, ax = plt.subplots()
    im, cbar = helpers.heatmap(
        A,
        df_corr.columns.to_list(),
        df_corr.columns.to_list(),
        ax=ax,
        cmap=cmap,
        cbarlabel="pearson correlation",
    )
    texts = helpers.annotate_heatmap(im, valfmt="{x:.1f}")
    plt.tight_layout()
    plt.savefig(f"ClusterMetricsCorrelation.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Standard Libraries Import
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--analyse", dest="analyse", help="analyse", action="store_true")
    parser.add_argument("-c", "--correlation", dest="correlation", help="analyse correlation of the metrics", action="store_true")
    args = parser.parse_args()
    if args.correlation is True:
        print(f"Analysing the results..")
        analyse_correlation()
    else:
        analyse_metrics()
        print(f"Nothing.")
        pass
    print(f"Done")
