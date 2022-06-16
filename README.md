# CircuitGraph LogicLocking

[![Python package](https://github.com/circuitgraph/logiclocking/actions/workflows/python-package.yml/badge.svg)](https://github.com/circuitgraph/logiclocking/actions/workflows/python-package.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


Implementations of various logic locks and attacks.

## Overview

This library provides both locks and attacks on logic locks. Interfacing with circuits is performed using [CircuitGraph](https://github.com/circuitgraph/circuitgraph).

Here's a simple example of locking a basic benchmark circuit. The circuit to lock must be input as a `circuitgraph.Circuit`. The locked circuits is returned as a `circuitgraph.Circuit` and the key is returned as a dictionary mapping key inputs to their correct logical values.

```python
import circuitgraph as cg
from logiclocking import locks, write_key

c = cg.from_lib("c880")
num_keys = 32
cl, k = locks.xor_lock(c, num_keys)

cg.to_file(cl, "c880_locked.v")
write_key(k, "c880_locked_key.txt")
```

The documentation can be found [here](https://circuitgraph.github.io/logiclocking/).

## Installing

Logiclocking is not yet available on PyPi, so you must install locally.

```shell
cd <install location>
git clone https://github.com/cmu-actl/logiclocking.git
cd logiclocking
pip install -e .
```

To run the miter attack or use `check_for_difference`, you must install python-sat

`pip install python-sat`

If you would like to use the Decision Tree Attack, you must also install sklearn.

`pip install scikit-learn`

## Contributing

Tests are run using the builtin unittest framework. Some basic linting is performed using flake8.
```shell
pip instsall flake8
make test
```

Code should be formatted using [black](https://black.readthedocs.io/en/stable/).
[Pre-commit](https://pre-commit.com) is used to automatically run black on commit.
```shell
pip install black pre-commit
pre-commit install
```
Pre-commit also runs a few other hooks, including a docstring formatter and linter. Docs follow the `numpy` documentation convention.

