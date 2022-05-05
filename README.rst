Multivariate Time Series Sub-Sequence Clustering Metric
=======================================================
.. image:: coverage.svg
    :alt: coverage badge

This repository provides a Python package for computing a multivariate time series subsequence clustering metric.

Status::

    Work in progress 🚧



Comparison of unsupervised clustering metrics with lorenz attractor data
------------------------------------------------------------------------

This example shows the effect of different metrics on the lorenz attractor dataset when using different types of label arrays. For the different unsupervised clustering labels we use the AgglomerativeClustering algorithm by varying the connectivity and the linkage as well as the number of clusters (along the lines of the scikit-learn example: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py)


.. image:: ClusterMetricComparisonAgglomerative-lorenz.png

Development steps
-----------------
Formatting
~~~~~~~~~~
Formatting is done with black and isort
.. code:: bash

    $ black mt3scm
    $ isort . --profile black

Linting
~~~~~~~
For this project we use pylint and flake8
.. code:: bash

    $ pylint mt3scm -d "C0301,C0103,E0401" --fail-under=8
    $ flake8 mt3scm --max-line-length=120 --ignore=E501,W503

Type checking
~~~~~~~~~~~~~
.. code:: bash

    $ mypy mt3scm


Testing
~~~~~~~

Testing the package with pytest and generating code coverage information.

.. code:: bash

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
