"""Various utils for working with logic-locked circuits."""
from ast import literal_eval

import circuitgraph as cg


def check_for_difference(oracle, locked_circuit, key):
    """
    Check the correctness of a key.

    Returns any differences between the oracle and a locked circuit
    with a specific key applied.

    Parameters
    ----------
    oracle: circuitgraph.Circuit
            The unlocked circuit to check against.
    locked_circuit: circuitgraph.Circuit
            The locked circuit to apply the key to.
    key: dict of str:bool
            The key to check

    Returns
    -------
    False or dict of str:bool
            False if there is no difference, otherwise the assignment that
            produced a difference.

    """
    m = cg.tx.miter(oracle, locked_circuit)
    key = {f"c1_{k}": v for k, v in key.items()}

    live = cg.sat.solve(m, assumptions=key)
    if not live:
        return True

    return cg.sat.solve(m, assumptions={"sat": True, **key})


def locked_unroll(
    locked_circuit,
    key,
    num_copies,
    D="D",
    Q="Q",
    ignore_pins="CK",
    initial_values=None,
    remove_unloaded=True,
    prefix="cg_unroll",
):
    """
    Unrolls a sequential circuit to prepare for a combinational attack.

    This can be used for locks applied on sequential circuits that prevent
    scan-chain access.
    Note that this function uses the `prefix` variable to identify unrolled
    nodes, so choosing a prefix that is already used in nodes in the sequential
    circuit can cause undefined behavior.

    Parameters
    ----------
    locked_circuit: circuitgraph.Circuit
            The circuit to unroll
    key: list of str or dict of str:bool
            The key inputs. If a dictionary is passed in, this key is used
            to construct an unrolled oracle and this is returned in addition
            to the unrolled locked circuit
    num_copies: int
            The number of unrolled copies of the circuit to create
    D: str
            The name of the D pin of the sequential elements
    Q: str
            The name of the Q pin of the sequential elements
    ignore_pins: str or list of str
            The pins on the sequential elements to ignore
    initial_values: str or dict of str:str
            The initial values of the data ports for the first timestep.
            If None, the ports will be added as primary inputs.
            If a single value ('0', '1', or 'x'), every flop will get that value.
            Can also pass in dict mapping flop names to values.
    remove_unloaded: bool
            If True, unloaded inputs will be removed after unrolling. This can remove
            unused sequential signals such as the clock and reset.
    prefix: str
            The prefix to use for naming unrolled nodes.

    Returns
    -------
    circuitgraph.Circuit, circuitgraph.Circuit, dict of str:list of str
            The unrolled locked circuit, the unrolled oracle, and the io map.

    """
    locked_circuit_unrolled, io_map = cg.tx.sequential_unroll(
        locked_circuit,
        num_copies,
        D,
        Q,
        ignore_pins=ignore_pins,
        initial_values=initial_values,
        remove_unloaded=remove_unloaded,
        prefix=prefix,
    )

    for k in key:
        locked_circuit_unrolled.set_type(io_map[k], "buf")
        locked_circuit_unrolled.add(k, "input", fanout=io_map[k])
        del io_map[k]

    oracle_unrolled = locked_circuit_unrolled.copy()
    for k, v in key.items():
        oracle_unrolled.set_type(k, str(int(v)))

    return locked_circuit_unrolled, oracle_unrolled, io_map


def write_key(key, filename):
    """Write a key dictionary to a file."""
    with open(filename, "w") as f:
        f.write(str(key) + "\n")


def read_key(filename):
    """Read a key dictionary from a file."""
    with open(filename) as f:
        return literal_eval(f.read())
