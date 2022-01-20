from ast import literal_eval

import circuitgraph as cg
from circuitgraph.sat import sat
from circuitgraph.transform import miter


def check_for_difference(oracle, locked_circuit, key):
    """
    Checks if there is a difference between an oracle and a locked circuit
    with a specific key applied.

    Parameters
    ---------
    oracle: circuitgraph.CircuitGraph
            the unlocked circuit to check against.
    locked_circuit: circuitgraph.CircuitGraph
            the locked circuit to apply the key to.
    key: dict of str:bool
            the key to check

    Returns
    -------
    False or dict of str:bool
            False if there is no difference, otherwise the assignment that
            produced a difference.
    """
    m = miter(oracle, locked_circuit)
    key = {f'c1_{k}': v for k, v in key.items()}

    live = sat(m, assumptions=key)
    if not live:
        return True

    return sat(m, assumptions={'sat': True, **key})


def unroll(locked_circuit,
           key,
           num_unroll,
           D="D",
           Q="Q",
           ignore_pins="CK",
           initial_values=None):
    prefix = "cg_unroll"
    locked_circuit_unrolled = cg.sequential_unroll(locked_circuit,
                                                   num_unroll,
                                                   D,
                                                   Q,
                                                   ignore_pins=ignore_pins,
                                                   initial_values=initial_values)
    for k in key:
        iter_keys = [f"{k}_{prefix}_{i}" for i in range(num_unroll + 1)]
        locked_circuit_unrolled.set_type(iter_keys, "buf")
        locked_circuit_unrolled.add(k, "input", fanout=iter_keys)

    oracle_unrolled = cg.copy(locked_circuit_unrolled)
    for k, v in key.items():
        oracle_unrolled.set_type(k, str(int(v)))

    return oracle_unrolled, locked_circuit_unrolled


def write_key(key, filename):
    """
    Write a key dictionary to a file
    """
    with open(filename, "w") as f:
        f.write(str(key) + "\n")


def read_key(filename):
    """
    Read a key dictionary from a file
    """
    with open(filename) as f:
        return literal_eval(f.read())
