# ME700-Assignment-2 Matrix Structural Analysis code
* This is the code to run the matrix structural analysis for the given problem statement.
* You can find the tutorial notebook in `src/tutorials.ipynb`
# mastanJ-2025-MECHE-BU
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![codecov](https://codecov.io/gh/erfanhamdi/mastanJ/graph/badge.svg?token=ZOJJW4Z03P)](https://codecov.io/gh/erfanhamdi/mastanJ)
[![tests](https://github.com/erfanhamdi/mastanJ/actions/workflows/code-coverage.yml/badge.svg)](https://github.com/erfanhamdi/mastanJ/actions)

## Setup instructions
1. Create a conda environment and activate it
```
conda create --name mastanJ-env python=3.9.13
conda activate mastanJ-env
```
2. Clone/Download the code and change directory to it Install the base requirements
```
pip install --upgrade pip setuptools wheel
```
3. Install the requirements by running this command in the root directory.
```
pip install -e .
```
4. You can run the tests using the `pytest` module
```
pytest -v --cov=setupexample  --cov-report term-missing
```