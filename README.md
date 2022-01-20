# CircuitGraph Logic Locking

Locking circuits with various logic locking techniques.

[![Build Status](https://app.travis-ci.com/circuitgraph/logiclocking.svg?token=iNbNxbyCMbSysAQsDskF&branch=master)](https://app.travis-ci.com/github/circuitgraph/logiclocking)

## Overview

This library provides both locks and attacks on logic locks. Interfacing with circuits is performed using [CircuitGraph](https://github.com/circuitgraph/circuitgraph).

Here's a simple example of locking a basic benchmark circuit. The circuit to lock must be input as a `circuitgraph.Circuit`. The locked circuits is returned as a `circuitgraph.Circuit` and the key is returned as a dictionary mapping key inputs to their correct logical values.

```python
import circuitgraph as cg
from logiclocking import locks

c = cg.from_lib('c880')
num_keys = 32
cl, k = locks.xor_lock(c, num_keys)

cg.to_file(cl, 'c880_locked.v')
with open('c880_locked.key', 'w') as f:
    f.write(f'{k}\n')
```

## Installing

Logiclocking is not yet available on PyPi, so you must install locally.

```shell
cd <install location>
git clone https://github.com/circuitgraph/logiclocking.git
cd logiclocking
pip install -r requirements.txt
pip install -e .
```
