from mt3scm import __version__
from mt3scm import mt3scm_score
import pytest
import numpy as np

def test_version():
    assert __version__ == '0.1.0'

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
    assert score == pytest.approx(0.5, 0.05)

# # Create two constant curves for each cluster (only one subsequence per cluster). Metric should be approx 0.25 because torsion is always constant(zero) therefore the standard deviation is 1. The curvature is constant and has a
# def test_curvature():
#     seq_length = 100
#     x = np.linspace(start=0, stop=0.9, num=seq_length)
#     y = np.sqrt(1 - x**2)
#     y2 = np.flip(y)*-1 + (+ 1 + y.min()) -  (1 - y.min())
#     y_all = np.concatenate([y[:-1], y2])
#     labels = np.concatenate([np.ones(seq_length)[:-1], np.ones(seq_length)*2])
#     x_all = np.linspace(start=0, stop=2, num=seq_length*2 - 1)
#     X = np.stack((x_all, y_all, x_all)).T
#     score = mt3scm_score(X, labels)
#     assert score == pytest.approx(0.25, 0.05)
