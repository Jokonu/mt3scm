# Multivariate Time Series Sub-Sequence Clustering Metric

![](coverage.svg)

This repository provides a Python package for computing a multivariate time series subsequence clustering metric[^koehn].


## Motivation

To our knowledge no existing clustering metric exists, that takes the time space variations like curvature, acceleration or torsion in a multidimensional space into consideration. We believe using these curve parameters, is an intuitive method to measure similarities between mechatronic system state changes or subsequences in multivariate time-series data (MTSD) in general.

## Details

Our MT3SCM score consists of three main components

$$
mt3scm = (cc_w + s_L + s_P) / 3
$$

The weighted curvature consistency ( $cc_w$ ), the silhouette location based ( $s_L$ ) and the silhouette curve-parameter based ( $s_P$ ). When making the attempt of clustering TSD, it is subjective and domain specific. Nevertheless, we try to take the intuitive approach of treating MTSD as space curves and use the parameterization as a similarity measure. This is done in two different ways. First we create new features by computing the curve parameters sample by sample (e.g.: curvature, torsion, acceleration) and determine their standard deviation for each cluster. Our hypothesis is, that with a low standard deviation of the curve parameters inside a cluster, the actions of a mechatronic system in this cluster are similar. We call this the curvature consistency ( $cc$ )

$$
cc_w = \frac{ \sum\limits_{i=1}^n cc_i \times N_{i}}{\sum\limits_{i=1}^{n} N^i}
$$

The second procedure is to apply these newly computed features, which are computed to scalar values per subsequence, onto a well established internal clustering metric, the silhouette score[^rous1]

The computation of the $cc$ comprises the calculation of the curvature $\kappa$ and the torsion $\tau$ at every time step $t$ with $\boldsymbol{x}_{t}$ .

Afterwards the $cc$ is calculated per cluster $i \in \mathcal{I}$ , by taking the empirical standard deviation for each curve parameter (exemplarily for $\kappa$ in with the set of subsequence indexes $\mathcal{J}_i$ within our cluster $i$ ).
The arithmetic mean of the standard deviations for the curvature $\kappa$, torsion $\tau$ and the acceleration $a$ results in the final $cc$ per cluster.

The main idea of this approach is to combine three main parts inside one metric.
First incentive is to reward a **low standard deviation of the curve parameters** in between a cluster (accomplished by $cc$ ).
Second, to benchmark the clusters **spatial separation based on the new feature space** (curve parameters, accomplished by $s_P$ ).
And third, to benchmark the clusters **spatial separation based on the median of the subsequence in the original feature space** (accomplished by $s_L$ ).

# Usage

    $ python -m plot_examples -car
    $ python -m plot_perfect

## Creating plots

    $ python -m plot_examples -car
    $ python -m plot_perfect

## Comparison of unsupervised clustering metrics with lorenz attractor data


This example shows the effect of different metrics on the lorenz attractor dataset when using different types of label arrays. For the different unsupervised clustering labels we use the AgglomerativeClustering algorithm by varying the connectivity and the linkage as well as the number of clusters (along the lines of the scikit-learn example: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py)


![](ClusterMetricComparisonAgglomerative-lorenz.png)

## Development steps

### Formatting

Formatting is done with black and isort
.. code:: bash

    $ black mt3scm
    $ isort . --profile black

### Linting

For this project we use pylint and flake8

    $ pylint mt3scm -d "C0301,C0103,E0401" --fail-under=8
    $ flake8 mt3scm --max-line-length=120 --ignore=E501,W503

### Type checking

    $ mypy mt3scm


### Testing
Testing the package with pytest and generating code coverage information.

    $ pytest

    # Produce .coverage file to use with coverage-badge
    $ pytest --cov-report= --cov=mt3scm tests/

    # run for generating the badge
    $ coverage-badge -o coverage.svg -f

    # For adding verbosity:
    $ pytest -vvv

    # For adding testcoverage:
    $ pytest --cov=mt3scm tests/

    # exclude slow or long marked tests:
    $ pytest -m "not (slow or long)"

    # disable warnings to show:
    $ pytest -v --disable-pytest-warnings

    # Printing my own logs with level Debug
    $ pytest --log-level=DEBUG

    # Rerun only last failed test with --lf, --last-failed or --ff, --failed-first
    $ pytest --lf

    # Generating html report
    $ pytest --cov=mt3scm --cov-report=html


## References

[^koehn]: Köhne, J. et al. Autoencoder based iterative modeling and multivariate time-series subsequence clustering algorithm

[^rous1]: "Rousseeuw, P. J. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics 20. PII: 0377042787901257, 53–65. ISSN: 03770427 (1987)"
