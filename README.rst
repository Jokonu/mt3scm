Multivariate Time Series Sub-Sequence CLustering Metric
=======================================================

This repository provides a Python package for computing a multivariate time series subsequence clustering metric.

Status::

    Work in progress 🚧


Testing
-------

Testing the package with pytest and generating code coverage information.

.. code:: bash

    $ pytest

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
    $ pytest --html=report.html --self-contained-html
