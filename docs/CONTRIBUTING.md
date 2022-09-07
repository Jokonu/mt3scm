# Contributing

Thank you for taking the time and interest to contribute to this project.
You are welcome to
- create a bug report
- submit a pull request
- suggest improvement or feature


## Development steps

If you plan on contribute code here are some commands to help keep the code clean.

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
