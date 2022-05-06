"""
Comparison of unsupervised clustering metrics
=============================================

This python script analysis the results from the examples

"""

# Authors: Jonas Köhne
# License: BSD 3 clause

# Third Party Libraries Import
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm as cm


def analyse(dataset_names:list[str] = ["lorenz", "thomas"], algo_type_names:list[str] = ["Random", "Agglomerative", "TimeSeriesKMeans"]):
    plt.style.use('plot_style.txt')
    dfs = []
    full_index_set = set()
    for dataset_name in dataset_names:
        for algo_type_name in algo_type_names:
            df= pd.read_pickle(f"ClusterMetricComparisonResults{algo_type_name}-{dataset_name}.pkl")
            full_index_set.update(df.index.names)
            fig, ax = plt.subplots()
            df_corr = df.corr()
            A = df_corr.values
            mask =  np.tri(A.shape[0], k=0)
            A = np.ma.array(A, mask=np.logical_not(mask)) # mask out the lower triangle
            cmap = cm.get_cmap("YlGn").copy() # jet doesn't have white color
            cmap.set_bad('w') # default value is 'k'
            im, cbar = heatmap(A, df_corr.columns.to_list(), df_corr.columns.to_list(), ax=ax, cmap=cmap, cbarlabel="pearson correlation")
            texts = annotate_heatmap(im, valfmt="{x:.1f}")
            # fig.tight_layout()
            plt.savefig(f"ClusterMetricsCorrelation-{algo_type_name}-{dataset_name}.png",bbox_inches='tight')
            plt.close()
            dfs.append(df.reset_index())
    df_all = pd.concat(dfs, axis=0)
    df_all = df_all.set_index(list(full_index_set)).sort_index()
    df_all = df_all.rename(columns={"masc-pos":"sl", "masc-kt":"sp"})
    df_corr = df_all.corr()
    A = df_corr.values
    mask =  np.tri(A.shape[0], k=0)
    A = np.ma.array(A, mask=np.logical_not(mask)) # mask out the lower triangle
    cmap = cm.get_cmap("YlGn").copy() # jet doesn't have white color
    cmap.set_bad('w') # default value is 'k'
    fig, ax = plt.subplots()
    im, cbar = heatmap(A, df_corr.columns.to_list(), df_corr.columns.to_list(), ax=ax, cmap=cmap, cbarlabel="pearson correlation")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")
    plt.tight_layout()
    plt.savefig(f"ClusterMetricsCorrelation.png",bbox_inches='tight')
    plt.close()


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not isinstance(data[i, j], np.ma.core.MaskedConstant):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

    return texts


if __name__ == "__main__":
    # Standard Libraries Import
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--analyse", dest="analyse", help="analyse", action="store_true")
    args = parser.parse_args()
    if args.analyse is True:
        print(f"Analysing the results..")
        analyse()
    else:
        print(f"Nothing.")
        pass
    print(f"Done")
