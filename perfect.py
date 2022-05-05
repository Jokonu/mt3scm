import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
from mt3scm import mt3scm_score, MT3SCM
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

def plot_testing_results(X:np.ndarray, score:float, labels:np.ndarray, test_name: str):
    # Third Party Libraries Import
    feature_names = ["x", "y", "z"]
    x_label, y_label, z_label = feature_names
    plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Computer Modern Serif"], "axes.grid": False})
    fig = plt.figure()
    fig.suptitle(f"\\textbf{{MT3SCM: {test_name}}}")
    subplot_title = f"mt3scm score={score:.3f}"
    if X.shape[1] >= 3:
        ax = fig.add_subplot(projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
        ax.set_zlabel(z_label)
    elif X.shape[1] == 2:
        ax = fig.add_subplot()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels)
    ax.set_title(subplot_title)
    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Clusters")
    ax.add_artist(legend1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    test_plots_path: Path = Path("test_plots")
    Path(test_plots_path).mkdir(parents=True, exist_ok=True)
    full_plot_name: Path = test_plots_path / str(test_name +".png")
    plt.savefig(full_plot_name)
    plt.close()

def main():

    # labels[0] = 0
    # generate curve 1
    start = (0, 0, 1)
    end = (3*np.pi, 0, -1)
    n_points = 60
    t = np.linspace(start=start[0], stop=end[0], num=n_points)
    xt = t
    yt = np.sin(t)
    zt = np.cos(t)
    X1 = np.stack((xt, yt, zt)).T
    labels1 = 1 + np.zeros(n_points)

    # # generate curve 1
    # start = (0, 0, 1)
    # end = (3*np.pi, 0, -1)
    # n_points = 60
    # t = np.linspace(start=start[0], stop=end[0], num=n_points)
    # xt = t
    # yt = np.sin(t)
    # zt = np.cos(t)
    # X1 = np.stack((xt, yt, zt)).T
    # labels1 = 1 + np.zeros(n_points)
    # labels1[0:10] = 0
    # labels1[10:20] = 2
    # labels1[30:40] = 0

    # generate passing from curve 1 to 2
    start = (3*np.pi, 0.01, -1)
    end = (3*np.pi, -2, -1)
    n_points = 100
    # t = np.geomspace(start=start[1], stop=-end[1], num=n_points)
    t = np.linspace(start=0.1, stop=2, num=n_points)
    t = t**2 / 2
    # t = np.geomspace(0.674, 100.0, num=1000)
    xt = start[0] + np.zeros(n_points)
    yt = -t
    zt = start[2] + np.zeros(n_points)
    X2 = np.stack((xt, yt, zt)).T
    labels2 = 2 + np.zeros(n_points)
    # labels2[0] = -1
    # labels[0] = 0

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
    # labels3[0] = -1

    # generate passing from curve 2 to 1
    start = (0, -2, 1)
    end = (0, 0.01, -1)
    n_points = 30
    # t = np.geomspace(start=0.01, stop=2, num=n_points)
    t = np.linspace(start=0, stop=2, num=n_points)
    t = t**2 / 2
    # t = np.geomspace(0.674, 100.0, num=1000)
    xt = start[0] + np.zeros(n_points)
    # yt = np.flip(t)
    yt = -t
    zt = start[2] + np.zeros(n_points)
    X4 = np.stack((xt, yt, zt)).T
    labels4 = 4 + np.zeros(n_points)
    # labels4[0] = -1
    # labels[0] = 0


    # n = 1000
    # theta_max = 4 * np.pi
    # theta = np.linspace(0, theta_max, n)
    # x1 = theta
    # z1 =  np.sin(theta)
    # y1 =  np.cos(theta)
    # label = np.ones(n)

    # n = 1000
    # theta_max = 4 * np.pi
    # theta = np.linspace(0, theta_max, n)
    # x2 = theta + x1.max()
    # z2=  np.sin(-theta)
    # y2 =  np.cos(-theta)
    # label2 = np.ones(n) + 1

    # xs = np.concatenate([x1, x2])
    # ys = np.concatenate([y1, y2])
    # zs = np.concatenate([z1, z2])
    # labels = np.concatenate([label, label2])
    # X = np.stack((xs, ys, zs)).T
    X = np.concatenate([X1, X2, X3, X4])
    # X = np.concatenate([X1, ])
    # X = np.concatenate([X1, X2, X3])
    X = np.concatenate([X, X+0.00001, X-0.00001])
    # X = np.concatenate([X])
    labels = np.concatenate([labels1, labels2, labels3, labels4])
    # labels = np.concatenate([labels1, labels2, labels3])
    # labels = np.concatenate([labels1])
    labels = np.concatenate([labels, labels, labels])
    # labels = np.concatenate([labels])


    # FIltering

    # b, a = signal.butter(8, 0.125)
    # b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.
    # b, a = signal.butter(3, 0.2, btype='low', analog = True)
    # X = signal.filtfilt(b, a, X, method="gust")
    # labels[0:100] = 0
    # labels[-10:] = 5

    # Score calculation and plotting
    # score = mt3scm_score(X, labels)
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    curvature(X)
    sil = silhouette_score(X, labels)
    print(f"X {sil=}")
    mt3 = MT3SCM(eps=5e-10, include_acceleration=True, include_speed_acceleration=False, distance_fn="euclidean", scale_input_data=False)
    mt3scm_metric = mt3.mt3scm_score(X, labels, standardize_subs_curve=False, edge_offset=5, max_curve_parameter_value=1e0)

    plot_curvature(mt3.kappa_X, file_name="kappa")
    plot_curvature(mt3.tau_X, file_name="tau")
    plot_testing_results(X, mt3scm_metric, labels, "helix1.png")
    print(f"{mt3scm_metric=:.3}\n{mt3.cc=:.3}, {mt3.wcc=:.3}, {mt3.masc_pos=:.3}, {mt3.masc_kt=:.3}")
    print(f"{mt3.df_centers=}")
    print(f"{mt3.df_curve=}")
    print(f"{mt3.cccs=}")
    labels = mt3.df_centers.index.get_level_values("c_id")
    plot_testing_results(mt3.df_centers.values, mt3scm_metric, labels, "centers.png")
    sil = silhouette_score(mt3.df_centers.values, labels)
    print(f"df_centers {sil=}")
    labels = mt3.df_curve.index.get_level_values("c_id")
    plot_testing_results(mt3.df_curve.values, mt3scm_metric, labels, "curveparams.png")
    sil = silhouette_score(mt3.df_curve.values, labels)
    print(f"df_curve {sil=}")

def plot_curvature(X, file_name: str = "curvature"):
    # Third Party Libraries Import
    feature_names = ["x", "y", "z"]
    x_label, y_label, z_label = feature_names
    plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Computer Modern Serif"], "axes.grid": False})
    fig = plt.figure()
    fig.suptitle(f"\\textbf{{MT3SCM: {file_name}}}")
    ax = fig.add_subplot()
    scatter = ax.plot(X)
    # subplot_title = f"mt3scm score={score:.3f}"
    # ax.set_title(subplot_title)
    # legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Clusters")
    # ax.add_artist(legend1)
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)
    # ax.set_zlabel(z_label)
    test_plots_path: Path = Path("test_plots")
    Path(test_plots_path).mkdir(parents=True, exist_ok=True)
    full_plot_name: Path = test_plots_path / str(file_name +".png")
    plt.savefig(full_plot_name)
    plt.close()

def curvature(X):
    # X = np.array([X[:,2], X[:,1], X[:,0]]).T
    # X = np.array([X[:,1], X[:,0], X[:,2]]).T
    # X = np.array([X[:,0], X[:,1], X[:,2]]).T
    g1 = np.gradient(X, axis=0)
    g2 = np.gradient(g1, axis=0)
    g3 = np.gradient(g2, axis=0)
    # t = 1
    # kappa=[]
    # for t in range(X.shape[0]):
    #     x1 = np.cross(g1[t], g2[t])
    #     top = np.linalg.norm(x1, axis=0)
    #     bottom = np.linalg.norm(g1[t], axis=0) ** 3
    #     k = np.divide(top, bottom)
    #     kappa.append(k)
    # plt.plot(np.array(kappa))
    # plt.savefig("debug")
    # plt.close()


    # THis is the correct one for genearl and without looping over time
    first = np.linalg.norm(g1, axis=1)**2
    second = np.linalg.norm(g2, axis=1)**2
    third = np.einsum("ij,ij->i", g1, g2)**2
    top = np.sqrt((first * second) - (third))
    bottom = np.linalg.norm(g1, axis=1) ** 3
    kappa3 = np.divide(top, bottom)
    plt.plot(np.array(kappa3))
    plt.savefig("debug3")
    plt.close()

    # t = 1
    # kappa2=[]
    # for t in range(X.shape[0]):
    #     top = np.sqrt((np.linalg.norm(g1[t], axis=0) * np.linalg.norm(g2[t], axis=0)) - (np.matmul(g1[t], g2[t])**2))
    #     bottom = np.linalg.norm(g1[t], axis=0) ** 3
    #     k = np.divide(top, bottom)
    #     kappa2.append(k)
    # kappa2 = np.array(kappa2)
    # plt.plot(np.array(kappa2))
    # plt.savefig("debug2")
    # plt.close()
    # top = np.einsum("ij,ij->i", g1, g2)
    # top = np.sqrt((np.linalg.norm(g1[t], axis=0) * np.linalg.norm(g2[t], axis=0)) - (np.matmul(g1[t], g2[t])**2))

if __name__ == "__main__":
    main()