from ast import literal_eval

import networkx as nx
from networkx.exception import NetworkXNoCycle
import circuitgraph as cg
from circuitgraph.sat import sat
from circuitgraph.transform import miter


def check_for_difference(oracle, locked_circuit, key):
    """
    Checks if there is a difference between an oracle and a locked circuit
    with a specific key applied.

    Parameters
    ---------
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
    m = miter(oracle, locked_circuit)
    key = {f"c1_{k}": v for k, v in key.items()}

    live = sat(m, assumptions=key)
    if not live:
        return True

    return sat(m, assumptions={"sat": True, **key})


def unroll(
    locked_circuit,
    key,
    num_unroll,
    D="D",
    Q="Q",
    ignore_pins="CK",
    initial_values=None,
    prefix="cg_unroll",
):
    """
    Unrolls a sequential circuit to prepare for a combinational attack. This can be
    used for locks applied on sequential circuits that prevent scan-chain access.

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
    num_unroll: int
            The number of times to unroll the circuit
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
    prefix: str
            The prefix to use for naming unrolled nodes.

    Returns
    -------
    circuitgraph.Circuit
            The unrolled locked circuit. If a dictionary was passed in, the oracle
            is also returned after the locked circuit
    """
    locked_circuit_unrolled = cg.sequential_unroll(
        locked_circuit,
        num_unroll,
        D,
        Q,
        ignore_pins=ignore_pins,
        initial_values=initial_values,
        prefix=prefix,
    )
    for k in key:
        iter_keys = [
            locked_circuit.uid(f"{k}_{prefix}_{i}") for i in range(num_unroll + 1)
        ]
        locked_circuit_unrolled.set_type(iter_keys, "buf")
        locked_circuit_unrolled.add(k, "input", fanout=iter_keys)

    oracle_unrolled = cg.copy(locked_circuit_unrolled)
    if isinstance(key, dict):
        for k, v in key.items():
            oracle_unrolled.set_type(k, str(int(v)))

        return locked_circuit_unrolled, oracle_unrolled
    else:
        return locked_circuit_unrolled


def acyclic_unroll(c):
    """
    Transform a cyclic circuit into an acyclic circuit
    """
    if c.blackboxes:
        raise ValueError("remove blackboxes")

    if not c.is_cyclic():
        raise ValueError("Circuit is not cyclic")

    # find feedback nodes
    feedback = set([e[0] for e in _approx_min_fas(c.graph)])

    # get startpoints
    sp = c.startpoints()

    # create acyclic circuit
    acyc = cg.Circuit(name=f"acyc_{c.name}")
    for n in sp:
        acyc.add(n, "input")

    # create copy with broken feedback
    c_cut = cg.copy(c)
    for f in feedback:
        fanout = c.fanout(f)
        c_cut.disconnect(f, fanout)
        c_cut.add(f"aux_in_{f}", "buf", fanout=fanout)
    c_cut.set_output(c.outputs(), False)

    # cut feedback
    for i in range(len(feedback) + 1):
        # instantiate copy
        acyc.add_subcircuit(c_cut, f"c{i}", {n: n for n in sp})

        if i > 0:
            # connect to last
            for f in feedback:
                acyc.connect(f"c{i-1}_{f}", f"c{i}_aux_in_{f}")
        else:
            # make feedback inputs
            for f in feedback:
                acyc.set_type(f"c{i}_aux_in_{f}", "input")

    # connect outputs
    for o in c.outputs():
        acyc.add(o, "buf", fanin=f"c{i}_{o}", output=True)

    cg.lint(acyc)
    if acyc.is_cyclic():
        raise ValueError("circuit still cyclic")
    return acyc


def _approx_min_fas(DG):
    DGC = DG.copy()
    s1, s2 = [], []
    while DGC.nodes:
        # find sinks
        sinks = [n for n in DGC.nodes if DGC.out_degree(n) == 0]
        while sinks:
            s2 += sinks
            DGC.remove_nodes_from(sinks)
            sinks = [n for n in DGC.nodes if DGC.out_degree(n) == 0]

        # find sources
        sources = [n for n in DGC.nodes if DGC.in_degree(n) == 0]
        while sources:
            s1 += sources
            DGC.remove_nodes_from(sources)
            sources = [n for n in DGC.nodes if DGC.in_degree(n) == 0]

        # choose max in/out degree difference
        if DGC.nodes:
            n = max(DGC.nodes, key=lambda x: DGC.out_degree(x) - DGC.in_degree(x))
            s1.append(n)
            DGC.remove_node(n)

    ordering = s1 + list(reversed(s2))
    feedback_edges = [
        e for e in DG.edges if ordering.index(e[0]) > ordering.index(e[1])
    ]
    feedback_edges = [(u, v) for u, v in feedback_edges if u in nx.descendants(DG, v)]

    DGC = DG.copy()
    DGC.remove_edges_from(feedback_edges)
    try:
        if nx.find_cycle(DGC):
            raise ValueError("Circuit still cyclic")
    except NetworkXNoCycle:
        pass

    return feedback_edges


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
