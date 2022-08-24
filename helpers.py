"""Helper functions for example plots"""

# Author: Jonas Köhne <jokohonas@gmail.com>
# License: BSD 3 clause

# Standard Libraries Import
import logging
from pathlib import Path

# Third Party Libraries Import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Own Libraries Import
from mt3scm import MT3SCM


def setupLogger(logging_file_path: Path = None, name: str = "jizzle", loglevel="INFO", log_to_file: bool = False):
    if loglevel is None:
        loglevel = "INFO"
    if logging_file_path is None:
        logging_file_path = Path.cwd() / "logs"
        logging_file_name = logging_file_path / "logger.log"
        Path(logging_file_path).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    # create formatter and add it to the handlers
    fmt = "%(asctime)s - %(name)s %(levelname)s %(funcName)s: %(message)s"
    if log_to_file is True:
        # create file handler which logs warning messages
        fh = logging.FileHandler(logging_file_name)
        fh.setLevel(loglevel)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    # coloredlogs.install(level=loglevel, logger=logger, fmt=fmt)
    return logger

def set_plot_params():
    # sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 8,
            "font.sans-serif": ["Computer Modern Roman"],
            "axes.grid": False,
            "image.cmap": cm.get_cmap("viridis")
        }
    )

def ax_scatter_3d_original(X, Y, Z, ax, kappa: np.ndarray, xyz_labels: list[str] = ["x", "y", "z"], subplot_title: str = None, marker_size: float = 0.8, marker_size_array=None, marker="o", plot_changepoints: bool = True, alpha=0.3, line_scaling_factor: float = 1.0, set_ticks_and_labels: bool = True, pad: float=0.2):
    cmap = cm.get_cmap("viridis")
    marker="."
    data = kappa
    data = (data - data.min()) / (np.std(data))
    color = np.log((np.abs(data* 10) + 1) ** 2) * line_scaling_factor
    scat = ax.scatter(X, Y, Z, c=color, cmap=cmap, s=color, marker=marker)
    ax.set_title(subplot_title)
    fig = plt.gcf()
    clb = fig.colorbar(scat, ax=ax, shrink=0.5, pad=pad)
    clb.set_ticks([color.min(),color.max() / 2, color.max()])
    if set_ticks_and_labels is True:
        ax.set_xlabel(f"{xyz_labels[0]} [-]")
        ax.set_ylabel(f"{xyz_labels[1]} [-]")
        ax.set_zlabel(f"{xyz_labels[2]} [-]")
        ax.tick_params(labelsize=8)
        clb.set_ticklabels(['Low', 'Medium', 'High'], rotation = 45)
        clb.ax.tick_params(labelsize=8)
    else:
        ax.tick_params(labelbottom = False, labelleft=False)
        # clb.set_ticklabels(['L', 'M', 'H'], rotation = 0)
        clb.set_ticklabels(['', '', ''], rotation = 0)
        clb.ax.tick_params(labelsize=8)
        # clb.ax.tick_params(labelright=False)

def ax_scatter_3d(
    X,
    Y,
    Z,
    ax,
    labels,
    subplot_title: str = "Subplot Title",
    marker_size: float = 0.8,
    marker_size_array=None,
    marker="o",
    plot_changepoints: bool = False,
    alpha=0.3,
    remove_ticks: bool = True,
):
    if marker_size_array is not None:
        marker_size = marker_size_array
    if plot_changepoints is True:
        labs = np.where(labels[:-1] != labels[1:], 1, 0)
        labs = np.concatenate([[0], labs])
        scat2 = ax.scatter(
            X[labs == 1],
            Y[labs == 1],
            Z[labs == 1],
            s=20,
            c="black",
            marker=".",
            alpha=alpha,
            label="Changepoint",
            zorder=10,
        )
    n_unique_labels = len(np.unique(labels))
    cmap = cm.get_cmap("viridis", n_unique_labels)
    norm = Normalize(vmin=0, vmax=n_unique_labels, clip=False)
    scat = ax.scatter(X, Y, Z, c=labels, cmap=cmap, s=marker_size, marker=marker, norm=norm)
    fig = plt.gcf()
    clb = fig.colorbar(scat, ax=ax, shrink=0.5, pad=0.2)
    clb.ax.set_title("Cluster ID", fontsize=10)
    if n_unique_labels < 11:
        tick_locs = np.arange(n_unique_labels) + 0.5
        clb.set_ticks(tick_locs)
        clb.set_ticklabels(np.arange(n_unique_labels))
    ax.set_title(subplot_title, fontsize=10)
    # ax.legend(fontsize=8)


def calc_unsupervised_metrics(X, label_array, edge_offset: int = 3, n_min_subs: int = 2):
    mt3 = MT3SCM()
    mt3scm_metric = mt3.mt3scm_score(X, label_array, edge_offset=edge_offset, n_min_subs=n_min_subs)
    metrics_dict = {}
    metrics_dict["mt3scm"] = mt3scm_metric
    metrics_dict["wcc"] = mt3.wcc
    metrics_dict["cc"] = mt3.cc
    metrics_dict["masc_pos"] = mt3.masc_pos
    metrics_dict["sl"] = mt3.masc_pos
    metrics_dict["masc-kt"] = mt3.masc_kt
    metrics_dict["sp"] = mt3.masc_kt
    metrics_dict["silhouette"] = silhouette_score(X, label_array)
    metrics_dict["calinski"] = calinski_harabasz_score(X, label_array)
    metrics_dict["davies"] = davies_bouldin_score(X, label_array)
    return metrics_dict, mt3.kappa_X, mt3.tau_X


def gen_synth_data():
    # generate curve 1
    start = (0, 0, 1)
    end = (3 * np.pi, 0, -1)
    n_points = 50
    t = np.linspace(start=start[0], stop=end[0], num=n_points)
    xt = t
    yt = np.sin(t)
    zt = np.cos(t)
    X1 = np.stack((xt, yt, zt)).T
    labels1 = 1 + np.zeros(n_points)

    # generate passing from curve 1 to 2
    start = (3 * np.pi, -0.5, -1)
    end = (3 * np.pi, -1.88, -1)
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
    start = (3.5 * np.pi, -2, -1)
    end = (0.5 * np.pi, 2, 1)
    n_points = 300
    t = np.linspace(start=start[0], stop=end[0], num=n_points)
    # t2 = np.linspace(start=3*np.pi, stop=0, num=n_points)
    xt = np.linspace(start=3 * np.pi, stop=0, num=n_points)
    # xt = t - xt[0]
    yt = np.cos(t) - 2
    zt = np.sin(t)
    X3 = np.stack((xt, yt, zt)).T
    endpoint = (X3[-1, 0], X3[-1, 1], X3[-1, 2])
    labels3 = 3 + np.zeros(n_points)

    # generate passing from curve 2 to 1
    start = (0, -2, 1)
    end = (0, 0.01, -1)
    n_points = 80
    t = np.linspace(start=0.35, stop=1.9, num=n_points)
    t = t**2 / 2
    # t = np.geomspace(0.674, 100.0, num=1000)
    xt = start[0] + np.zeros(n_points)
    yt = -t
    yt = np.flip(yt)
    zt = start[2] + np.zeros(n_points)
    X4 = np.stack((xt, yt, zt)).T
    labels4 = 4 + np.zeros(n_points)
    # import pdb;pdb.set_trace()

    X = np.concatenate([X1, X2, X3, X4])
    X = np.concatenate([X, X + 0.00001, X - 0.00001])
    # X = np.concatenate([X, X, X])
    labels = np.concatenate([labels1, labels2, labels3, labels4])
    labels = np.concatenate([labels, labels, labels])

    # Repeat the data to have more subsequences per cluster
    n_repeats = 2
    X = np.tile(X, (n_repeats, 1))
    labels = np.tile(labels, n_repeats)
    # Add some randomness to the data.
    # X = np.random.rand(X.shape[0], X.shape[1]) * 0.01 + X
    X = np.random.rand(X.shape[0], X.shape[1]) * 0.00001 + X

    # Create DataFrame
    df = pd.DataFrame(data=X, columns=["x", "y", "z"])
    # Add index with labels
    df["time"] = np.arange(0, X.shape[0])
    df["label"] = labels
    df = df.set_index(["time", "label"])
    # Save to pickle
    parent_path = Path(__file__).parent.resolve()
    df.plot()
    plt.savefig(parent_path / "own_synth.png")
    plt.close()
    # mets, _, _ = calc_unsupervised_metrics(X, labels)
    # print(mets)
    return X, labels


def generate_thomas_attractor_data(dt: float = 1, num_steps: int = 2000, b: float = 0.1615):
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
    data = np.array([xs, ys, zs * 10]).T
    return pd.DataFrame(data, columns=["xs", "ys", "zs"])


def generate_lorenz_attractor_data(dt: float = 0.005, num_steps: int = 3000, scale_zs: float = 1.0):
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
    data = np.array([xs, ys, zs*scale_zs]).T
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


def generate_random_sequences(
    length: int = 1000,
    min_seq_length: int = 10,
    max_seq_length: int = 200,
    number_of_sequences: int = 10,
):
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

def generate_one_random_sequence(
    length: int = 1000,
    min_seq_length: int = 5,
    max_seq_length: int = 10,
    number_of_short_sequences: int = 1,
):
    data: np.ndarray = np.array([])
    label_array: np.ndarray = np.zeros(length)
    for seq_id in range(1, number_of_short_sequences + 1):
    # while data.size < length:
        seq_len = np.random.randint(min_seq_length, max_seq_length)
        seq_pos_start = np.random.randint(1, length -1)
        label_array[seq_pos_start:seq_pos_start + seq_len] = seq_id
    # _, label_array = np.unique(data, return_inverse=True)
    # n_unique_labels = len(np.unique(label_array))
    # if n_unique_labels == 1:
        # label_array[-1] = 1
    return label_array


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


def gen_curve_synth_data():
    # generate curve 1
    start = (0, 0, 1)
    end = (3*np.pi, 0, -1)
    n_points = 150
    t = np.linspace(start=start[0], stop=end[0], num=n_points)
    xt = t
    yt = np.sin(t)
    zt = np.cos(t)
    X1 = np.stack((xt, yt, zt)).T
    labels1 = 1 + np.zeros(n_points)

    # generate passing from curve 1 to 2
    start = (3*np.pi, -0.5, -1)
    end = (3*np.pi, -1.88, -1)
    n_points = 45
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
    n_points = 240
    t = np.linspace(start=start[0], stop=end[0], num=n_points)
    # t2 = np.linspace(start=3*np.pi, stop=0, num=n_points)
    xt = np.linspace(start=3*np.pi, stop=0, num=n_points)
    # xt = t - xt[0]
    yt = np.cos(t) - 2
    zt = np.sin(t)
    X3 = np.stack((xt, yt, zt)).T
    endpoint = (X3[-1, 0], X3[-1, 1], X3[-1, 2])
    labels3 = 3 + np.zeros(n_points)

    # generate passing from curve 2 to 1
    start = (0, -2, 1)
    end = (0, 0.01, -1)
    n_points = 50
    t = np.linspace(start=0.35, stop=1.9, num=n_points)
    t = t**2 / 2
    # t = np.geomspace(0.674, 100.0, num=1000)
    xt = start[0] + np.zeros(n_points)
    yt = -t
    yt = np.flip(yt)
    zt = start[2] + np.zeros(n_points)
    X4 = np.stack((xt, yt, zt)).T
    labels4 = 4 + np.zeros(n_points)
    # import pdb;pdb.set_trace()
    X = np.concatenate([X1, X2, X3, X4])
    X = np.concatenate([X, X+0.00001, X-0.00001])
    # Concatenate the subsequence labels
    labels = np.concatenate([labels1, labels2, labels3, labels4])
    instance = np.zeros_like(labels)
    # Concatenate the instances
    instances = np.concatenate([instance, instance + 1, instance + 1])
    # Concatenate the instance labels
    labels = np.concatenate([labels, labels, labels])
    # Repeat the data to have more subsequences per cluster
    n_repeats = 10
    X = np.tile(X, (n_repeats, 1))
    labels = np.tile(labels, n_repeats)
    instances = np.tile(instances, n_repeats)
    # Add some randomness to the data.
    # X = np.random.rand(X.shape[0], X.shape[1]) * 0.01 + X
    X = np.random.rand(X.shape[0], X.shape[1]) * 0.00001 + X
    # Create DataFrame
    df = pd.DataFrame(data=X, columns=["x", "y", "z"])
    # Add index with labels
    df["time"] = np.arange(0, X.shape[0])
    df["label"] = labels
    df["instance"] = instances
    df = df.set_index(["time", "label", "instance"])
    return df

def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
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
