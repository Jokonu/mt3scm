"""Tests for 'Multivariate Time Series Sub-Sequence Clustering Metric'"""

# Author: Jonas Köhne <jokohonas@gmail.com>
# License: BSD 3 clause

# Standard Libraries Import
import inspect
from pathlib import Path

# Third Party Libraries Import
import numpy as np
import pytest
import toml

# Own Libraries Import
from mt3scm import __version__, mt3scm_score


def test_version():
    assert __version__ == '0.2.0'


def test_versions_are_in_sync():
    """Checks if the pyproject.toml and package.__init__.py __version__ are in sync."""

    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = toml.loads(open(str(path)).read())
    pyproject_version = pyproject["tool"]["poetry"]["version"]

    package_init_version = __version__
    assert package_init_version == pyproject_version

# Test for a straight line

# Test for a constant helix for each cluster. since only one subsequence in cluster the adapted silhouette coefficient (asc) for the centers is zero. The asc for kappa and torsion is also zero.
def test_helix():
    n = 1000
    theta_max = 4 * np.pi
    theta = np.linspace(0, theta_max, n)
    x1 = theta
    z1 =  np.sin(theta)
    y1 =  np.cos(theta)
    label = np.ones(n)

    n = 1000
    theta_max = 4 * np.pi
    theta = np.linspace(0, theta_max, n)
    x2 = theta + x1.max()
    z2=  np.sin(-theta)
    y2 =  np.cos(-theta)
    label2 = np.ones(n) + 1

    xs = np.concatenate([x1, x2])
    ys = np.concatenate([y1, y2])
    zs = np.concatenate([z1, z2])
    labels = np.concatenate([label, label2])
    X = np.stack((xs, ys, zs)).T
    score = mt3scm_score(X, labels)
    score_expected: float = -0.33
    np.testing.assert_allclose(score, score_expected, atol=1e-2)

# def test_helix2():
#     n = 1000
#     nl = 1000
#     theta_max = 4 * np.pi
#     theta = np.linspace(0, theta_max, n)
#     x1 = theta
#     z1 =  np.sin(theta)
#     y1 =  np.cos(theta)
#     label = np.ones(nl)

#     n = 1000
#     theta_max = 4 * np.pi
#     theta = np.linspace(0, theta_max, n)
#     x2 = theta + x1.max()
#     z2=  np.sin(theta)
#     y2 =  np.cos(theta)
#     label2 = np.ones(nl) + 1

#     xs = np.concatenate([x1, x2, x1 +20 , x2 +20])
#     ys = np.concatenate([y1, y2, y1, y2])
#     zs = np.concatenate([z1, z2, z1, z2])
#     # labels = np.concatenate([label, label2])
#     labels = np.concatenate([label, label2, label, label2])
#     # labels = np.concatenate([label, label, label2, label2])
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(xs, ys, zs, c=labels)
#     plt.savefig("curve.png")
#     plt.close()
#     X = np.stack((xs, ys, zs)).T
#     score = mt3scm_score(X, labels)
#     assert score == pytest.approx(0.5, 0.05)

# def plot_testing_results(X:np.ndarray, score:float, labels:np.ndarray, test_name: str):
#     # Third Party Libraries Import
#     import matplotlib.pyplot as plt
#     plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Computer Modern Serif"], "axes.grid": False})
#     fig = plt.figure()
#     fig.suptitle(f"\\textbf{{MT3SCM: {test_name}}}")
#     ax = fig.add_subplot(projection='3d')
#     scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
#     subplot_title = f"mt3scm score={score:.3f}"
#     ax.set_title(subplot_title)
#     legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Clusters")
#     ax.add_artist(legend1)
#     test_plots_path: Path = Path("test_plots")
#     Path(test_plots_path).mkdir(parents=True, exist_ok=True)
#     full_plot_name: Path = test_plots_path / str(test_name +".png")
#     plt.savefig(full_plot_name)
#     plt.close()

# # Create two constant curves for each cluster (only one subsequence per cluster). Metric should be approx 0.0 because only one subsequence per cluster means curvature consistency is zero, and curvatures, torsion, speed and acceleration are equal.
def test_curvature():
    seq_length = 100
    x = np.linspace(start=0, stop=0.9, num=seq_length)
    y = np.sqrt(1 - x**2)
    y2 = np.flip(y)*-1 + (+ 1 + y.min()) -  (1 - y.min())
    y_all = np.concatenate([y[:-1], y2])
    labels = np.concatenate([np.ones(seq_length)[:-1], np.ones(seq_length)*2])
    x_all = np.linspace(start=0, stop=2, num=seq_length*2 - 1)
    X = np.stack((x_all, y_all, x_all)).T
    score = mt3scm_score(X, labels)
    # plot_testing_results(X, score, labels, inspect.currentframe().f_code.co_name)
    score_expected: float = 0.0
    np.testing.assert_allclose(score, score_expected, atol=1e-2)

def test_normalize():
    X = np.random.random((10000, 3))
    y1 = np.ones((5000))
    y2 = np.ones((5000)) + 1
    labels = np.concatenate((y1, y2), axis=0)
    score = mt3scm_score(X, labels, standardize_subs_curve=False)
    # plot_testing_results(X, score, labels, inspect.currentframe().f_code.co_name)
    score_expected: float = 0.15
    np.testing.assert_allclose(score, score_expected, atol=5e-2)

def test_standardize():
    X = np.random.random((10000, 3))
    y1 = np.ones((5000))
    y2 = np.ones((5000)) + 1
    labels = np.concatenate((y1, y2), axis=0)
    score = mt3scm_score(X, labels, standardize_subs_curve=True)
    # plot_testing_results(X, score, labels, inspect.currentframe().f_code.co_name)
    score_expected: float = 0.0
    np.testing.assert_allclose(score, score_expected, atol=5e-3)

def test_constant_values():
    X = np.ones((1000, 3))
    labels = np.zeros((1000))
    with pytest.raises(ValueError):
        score = mt3scm_score(X, labels)

def test_one_cluster():
    X = np.random.randint((1000, 3))
    labels = np.zeros((1000))
    with pytest.raises(ValueError):
        score = mt3scm_score(X, labels)

def test_one_feature():
    X = np.ones((1000, 1))
    labels = np.zeros((1000))
    with pytest.raises(ValueError):
        score = mt3scm_score(X, labels)

