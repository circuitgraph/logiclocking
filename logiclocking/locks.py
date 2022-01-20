from itertools import product, zip_longest
import random
from random import randint, choice, choices, sample, shuffle

import circuitgraph as cg
from circuitgraph.sat import sat
from circuitgraph.transform import sensitivity_transform
from circuitgraph import logic
from pysat.solvers import Cadical
from pysat.formula import IDPool
from pysat.card import *


def trll(c, keylen, s1_s2_ratio=1, shuffle_key=True):
    """
    Locks a circuitgraph with Truly Random Logic Locking as outlined in
    N. Limaye, E. Kalligeros, N. Karousos, I. G. Karybali and O. Sinanoglu, 
    "Thwarting All Logic Locking Attacks: Dishonest Oracle With Truly Random 
    Logic Locking," in IEEE Transactions on Computer-Aided Design of 
    Integrated Circuits and Systems, vol. 40, no. 9, pp. 1740-1753,
    Sept. 2021.

    Parameters
    ----------
    c: circuitgraph.Circuit
            The circuit to lock.
    keylen: int
            The number of key bits to add.
    s1_s2_ratio: int or str
            The ratio between number of key gate locations locked where an
            inverter exists in the original design (s1) or where an inverter
            does not exist in the original design (s2). The paper leaves this
            value at 1 (meaning s1=s2=keylen/2), but they note that this
            could be adjusted based on the ratio of the locations where there
            is an inverter in the original netlist. Setting this parameter
            to the string "infer" will do this adjustment. I.e. 
            s1 = keylen*r, s2 = keylen*(1-r), where r is the number of 
            inverters in the circuit divided by the total number of gates.
    shuffle_key: bool
            By default, the key input labels are shuffled at the end of the
            algorithm so the labelling does not reveal which portion of the
            algorithm the key input was added during.

    Returns:
    circuitgraph.Circuit, dict of str:bool
            The locked circuit and the correct key value for each key input.
    """
    cl = cg.copy(c)

    if keylen % 2 != 0:
        raise NotImplementedError
    if s1_s2_ratio == "infer":
        raise NotImplementedError
    
    s1 = int((keylen // 2)*s1_s2_ratio)
    if s1 > keylen:
        raise ValueError(f"Unusable s1_s2_ratio: {s1_s2_ratio}")
    s2 = keylen - s1

    # NOTE: The algorithm in the paper selects s1a/s1b/s2a/s2b sizes using
    #       a "rand_dist" function on s1/s2. Alternatively, we could sample
    #       a new bernouli variable for each gate, which would result in
    #       much more evenly-sized distributions. Going with the paper
    #       implementation for now... They also don't specify a specific
    #       type of distribution, so defaulting to uniform
    s1a = randint(0, s1)
    s1b = s1 - s1a

    s2a = randint(0, s2)
    s2b = s2 - s2a

    inv_gates = list(c.filter_type("not"))
    random.shuffle(inv_gates)
    rem_gates = list(c.nodes() - c.io() - c.filter_type(
        ("not", "bb_input", "bb_output", "0", "1", "x")))
    random.shuffle(rem_gates)

    j = 0
    k = dict()
    # Replace existing inv_gates with XOR key-gates
    for _ in range(s1a):
        sel_gate = inv_gates.pop()
        ki = f"key_{j}"
        cl.add(ki, "input")
        k[ki] = True
        cl.set_type(sel_gate, "xor")
        cl.connect(ki, sel_gate)
        j += 1

    # Add XOR key-gates before existing inv_gates
    for _ in range(s1b):
        sel_gate = inv_gates.pop()
        ki = f"key_{j}"
        cl.add(ki, "input")
        k[ki] = False
        inv_fanin = cl.fanin(sel_gate)
        assert len(inv_fanin) == 1
        cl.disconnect(inv_fanin, sel_gate)
        cl.add(f"key_gate_{j}", "xor", fanin=inv_fanin | {ki}, fanout=sel_gate)
        j += 1
    
    # Add XOR key-gates and INV gates after existing rem_gates
    for _ in range(s2a):
        sel_gate = rem_gates.pop()
        ki = f"key_{j}"
        cl.add(ki, "input")
        k[ki] = True
        sel_fanout = cl.fanout(sel_gate)
        cl.disconnect(sel_gate, sel_fanout)
        cl.add(f"key_gate_{j}", "xor", fanin=(sel_gate, ki))
        cl.add(f"key_inv_{j}", "not", fanin=f"key_gate_{j}", fanout=sel_fanout)
        j += 1

    # Add XOR key-gates after existing rem_gates
    for _ in range(s2b):
        sel_gate = rem_gates.pop()
        ki = f"key_{j}"
        cl.add(ki, "input")
        k[ki] = False
        sel_fanout = cl.fanout(sel_gate)
        cl.disconnect(sel_gate, sel_fanout)
        cl.add(f"key_gate_{j}",
               "xor",
               fanin=(sel_gate, ki),
               fanout=sel_fanout)
        j += 1

    # Shuffle keys
    if shuffle_key:
        new_order = list(range(keylen))
        shuffle(new_order)
        shuffled_k = dict()
        intermediate_mapping = dict()
        final_mapping = dict()
        for old_idx, new_idx in enumerate(new_order):
            shuffled_k[f"key_{new_idx}"] = k[f"key_{old_idx}"]
            intermediate_mapping[f"key_{old_idx}"] = f"key_{old_idx}_temp"
            final_mapping[f"key_{old_idx}_temp"] = f"key_{new_idx}"
        cl.relabel(intermediate_mapping)
        cl.relabel(final_mapping)
        return cl, shuffled_k
    else:
        return cl, k


def xor_lock(c, keylen, key_prefix='key_', replacement=False):
    """
    Locks a circuitgraph with a random xor lock as outlined in
    J. A. Roy, F. Koushanfar and I. L. Markov, "Ending Piracy of Integrated
    Circuits," in Computer, vol. 43, no. 10, pp. 30-38, Oct. 2010.

    Parameters
    ----------
    c: circuitgraph.CircuitGraph
            Circuit to lock.
    keylen: int
            the number of bits in the key
    replacement: bool
            If True, the same line can be locked twice (resulting in a chain
            of key gates)

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    # create copy to lock
    cl = cg.copy(c)

    # randomly select gates to lock
    if replacement:
        gates = choices(tuple(cl.nodes() - cl.outputs()), k=keylen)
    else:
        gates = sample(tuple(cl.nodes() - cl.outputs()), keylen)

    # insert key gates
    key = {}
    for i, gate in enumerate(gates):
        # select random key value
        key[f'{key_prefix}{i}'] = choice([True, False])

        # create xor/xnor,input
        gate_type = 'xnor' if key[f'{key_prefix}{i}'] else 'xor'
        fanout = cl.fanout(gate)
        cl.disconnect(gate, fanout)
        cl.add(f'key_gate_{i}', gate_type, fanin=gate, fanout=fanout)
        cl.add(f'{key_prefix}{i}', 'input', fanout=f'key_gate_{i}')

    cg.lint(cl)
    return cl, key


def mux_lock(c, keylen, avoid_loops=False, key_prefix='key_'):
    """
    Locks a circuitgraph with a mux lock as outlined in
    J. Rajendran et al., "Fault Analysis-Based Logic Encryption," in IEEE
    Transactions on Computers, vol. 64, no. 2, pp. 410-424, Feb. 2015,
    doi: 10.1109/TC.2013.193.

    Parameters
    ----------
    c: circuitgraph.CircuitGraph
            Circuit to lock.
    keylen: int
            the number of bits in the key.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    # create copy to lock
    cl = cg.copy(c)

    # get 2:1 mux
    m = logic.mux(2)

    # randomly select gates
    gates = sample(tuple(cl.nodes() - cl.outputs()), keylen)
    if avoid_loops:
        decoy_gates = set()
    else:
        decoy_gates = sample(tuple(cl.nodes() - cl.outputs()), keylen)

    # insert key gates
    key = {}
    for i, gate in enumerate(gates):
        # select random key value
        key_val = choice([True, False])

        if avoid_loops:
            decoy_gate = choice(tuple(c.nodes() - c.io() - set(gates) - cl.transitive_fanout(gate) - decoy_gates))
            decoy_gates.add(decoy_gate)
        else:
            decoy_gate = decoy_gates[i]

        # create and connect mux
        fanout = cl.fanout(gate)
        cl.disconnect(gate, fanout)
        cl.add_subcircuit(m,f'mux_{i}')
        cl.connect(f'mux_{i}_out', fanout)
        key_in = cl.add(f'{key_prefix}{i}', 'input', fanout=f'mux_{i}_sel_0', uid=True)
        key[key_in] = key_val
        if key_val:
            cl.connect(gate, f'mux_{i}_in_1')
            cl.connect(decoy_gate, f'mux_{i}_in_0')
        else:
            cl.connect(gate, f'mux_{i}_in_0')
            cl.connect(decoy_gate, f'mux_{i}_in_1')

    cg.lint(cl)
    return cl, key


def random_lut_lock(c, num_gates, lut_width):
    """
    Locks a circuitgraph by replacing random gates with LUTS. This is kind of
    like applying LUT-lock with no replacement strategy.
    (H. Mardani Kamali, K. Zamiri Azar, K. Gaj, H. Homayoun and A. Sasan,
    "LUT-Lock: A Novel LUT-Based Logic Obfuscation for FPGA-Bitstream and
    ASIC-Hardware Protection," 2018 IEEE Computer Society Annual Symposium on
    VLSI (ISVLSI), Hong Kong, 2018, pp. 405-410.)

    Parameters
    ----------
    circuit: circuitgraph.CircuitGraph
            Circuit to lock.
    num_gates: int
            the number of gates to lock.
    lut_width: int
            LUT width, defines maximum fanin of locked gates.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    # create copy to lock
    cl = cg.copy(c)

    # parse mux
    m = logic.mux(2**lut_width)

    # randomly select gates
    potentialGates = set(g for g in cl.nodes() - cl.io()
                         if len(cl.fanin(g)) <= lut_width)
    gates = sample(tuple(potentialGates), num_gates)
    potentialGates -= set(gates)
    potentialGates -= cl.transitive_fanout(gates)

    # insert key gates
    key = {}
    for i, gate in enumerate(gates):

        fanout = list(cl.fanout(gate))
        fanin = list(cl.fanin(gate))
        padding = sample(tuple(potentialGates - cl.fanin(gate)),
                                lut_width - len(fanin))

        #create LUT
        cl.add_subcircuit(m, f'lut_{i}')

        # connect keys
        for j, vs in enumerate(product([False, True],
                                       repeat=len(fanin + padding))):
            assumptions = {s: v for s, v in zip(
                fanin+padding, vs[::-1]) if s in fanin}
            key_in = cl.add(f'key_{i*2**lut_width+j}', 'input',
                   fanout=f'lut_{i}_in_{j}',uid=True)
            result = sat(c, assumptions)
            if not result:
                key[key_in] = False
            else:
                key[key_in] = result[gate]

        # connect out
        cl.disconnect(gate, fanout)
        cl.connect(f'lut_{i}_out', fanout)

        # connect sel
        for j, f in enumerate(fanin + padding):
            cl.connect(f, f'lut_{i}_sel_{j}')

        # delete gate
        cl.remove(gate)
        cl = cg.relabel(cl, {f'lut_{i}_out': gate})

    cg.lint(cl)
    return cl, key


def lut_lock(c,
             num_gates,
             count_keys=False,
             skip_fi1=True,
             rank_by_shared_fanin=False,
             key_prefix="key_"):
    """
    Locks a circuitgraph with NB2-MO-HSC LUT-lock as outlined in
    H. Mardani Kamali, K. Zamiri Azar, K. Gaj, H. Homayoun and A. Sasan,
    "LUT-Lock: A Novel LUT-Based Logic Obfuscation for FPGA-Bitstream and
    ASIC-Hardware Protection," 2018 IEEE Computer Society Annual Symposium on
    VLSI (ISVLSI), Hong Kong, 2018, pp. 405-410.

    Parameters
    ----------
    circuit: circuitgraph.CircuitGraph
            Circuit to lock.
    num_gates: int
            The number of gates to lock.
    count_keys: bool 
            If true, continue locking until at least `num_gates` keys are
            added instead of `num_gates` gates.
    skip_fi1: int
            If True, nodes with a fanin of 1 (i.e. buf or inv) will not
            be considered for locking.
    rank_by_shared_fanin: bool
            If True, the output with the least shared fanin with other outputs
            will be selected for locking first. By default, the output with
            the least amount of total fanin is selected for locking first.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input

    Raises
    ------
    ValueError
            if there are not enough viable gates to lock.
    """
    # create copy to lock
    cl = cg.copy(c)

    def calc_skew(gate, cl):
        d = {False: 0, True: 0}
        fanin = list(cl.fanin(gate))

        # create subcircuit containing just gate for simulation
        simc = cg.Circuit()
        for i in fanin:
            simc.add(i, 'input')
        simc.add(gate, type=cl.type(gate), fanin=fanin)

        # simulate
        for i, vs in enumerate(product([False, True], repeat=len(fanin))):
            assumptions = {s: v for s, v in zip(fanin, vs[::-1])}
            result = sat(simc, assumptions)
            if not result:
                d[False] += 1
            else:
                d[result[gate]] += 1
        num_combos = 2**len(fanin)
        return abs(d[False] / num_combos - d[True] / num_combos)

    def replace_lut(gate, cl):
        key = dict()
        m = logic.mux(2**len(cl.fanin(gate)))
        fanout = list(cl.fanout(gate))
        fanin = list(cl.fanin(gate))

        # create LUT
        cl.add_subcircuit(m, f'lut_{gate}')

        # create subcircuit containing just gate for simulation
        simc = cg.Circuit()
        for i in fanin:
            simc.add(i, 'input')
        simc.add(gate, type=cl.type(gate), fanin=fanin)

        # connect keys
        for i, vs in enumerate(product([False, True], repeat=len(fanin))):
            assumptions = {s: v for s, v in zip(fanin, vs[::-1])}
            cl.add(f'{key_prefix}{gate}_{i}', 'input',
                   fanout=f'lut_{gate}_in_{i}')
            result = sat(simc, assumptions)
            if not result:
                key[f'{key_prefix}{gate}_{i}'] = False
            else:
                key[f'{key_prefix}{gate}_{i}'] = result[gate]

        # connect out
        cl.disconnect(gate, fanout)
        cl.connect(f'lut_{gate}_out', fanout)

        # connect sel
        for i, f in enumerate(fanin):
            cl.connect(f, f'lut_{gate}_sel_{i}')

        # delete gate
        cl.remove(gate)
        return key, [f'lut_{gate}_{n}' for n in m.nodes()], f'lut_{gate}_out'

    def continue_locking(locked_gates, num_gates, keys, count_keys):
        if count_keys:
            return len(keys) < num_gates
        else:
            return locked_gates < num_gates

    locked_gates = 0
    outputs = list(cl.outputs())
    if rank_by_shared_fanin:
        def rank_output(x):
            other_outputs = [o for o in outputs if o != x]
            other_fanin = cl.transitive_fanin(other_outputs)
            curr_fanin = cl.transitive_fanin(x)
            return len(curr_fanin & other_fanin)
    else:
        def rank_output(x):
            return len(cl.transitive_fanin(x))

    outputs.sort(key=rank_output)
    candidates = []
    forbidden_nodes = set()
    keys = dict()
    while continue_locking(locked_gates, num_gates, keys, count_keys):
        if not candidates:
            outputs = [o for o in outputs if o not in forbidden_nodes]
            try:
                candidates.append(outputs.pop(0))
            except IndexError:
                raise ValueError('Ran out of candidate gates at '
                                 f'{locked_gates} gates.')
        else:
            candidate = candidates.pop(0)
            candidate_is_output = cl.is_output(candidate)
            children = cl.fanin(candidate)
            if candidate in forbidden_nodes:
                candidates += [g for g in children if g not in forbidden_nodes]
                continue
            forbidden_nodes.add(candidate)
            if len(children) == 0:
                continue
            if skip_fi1 and len(children) == 1:
                child = children.pop()
                if child not in forbidden_nodes | set(candidates):
                    candidates.insert(0, child)
                continue
            key, nodes, output_to_relabel = replace_lut(candidate, cl)
            keys.update(key)
            forbidden_nodes.update(nodes)
            cl = cg.relabel(cl, {output_to_relabel: candidate})
            if candidate_is_output:
                cl.set_output(candidate)
            for g1 in children:
                forbidden_nodes.add(g1)
                for g2 in cl.fanin(g1):
                    if g2 not in forbidden_nodes | set(candidates):
                        candidates.append(g2)
            # Sort by least number of outputs in fanout cone, then most skew
            candidates.sort(key=lambda x: (
                len(cl.transitive_fanout(x) & cl.outputs()),
                -calc_skew(x, cl)))
            locked_gates += 1

    return cl, keys


def sfll_hd(c, width, hd):
    """
    Locks a circuitgraph with SFLL-HD as outlined in
    Muhammad Yasin, Abhrajit Sengupta, Mohammed Thari Nabeel, Mohammed Ashraf,
    Jeyavijayan (JV) Rajendran, and Ozgur Sinanoglu. 2017. Provably-Secure
    Logic Locking: From Theory To Practice. In Proceedings of the 2017 ACM
    SIGSAC Conference on Computer and Communications Security (CCS ’17).
    Association for Computing Machinery, New York, NY, USA, 1601–1618.

    Parameters
    ----------
    c: circuitgraph.CircuitGraph
            Circuit to lock.
    width: int
            key width, also the minimum fanin of the gates to lock.
    hd: int
            the hamming distance to lock with, as explained in the paper.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    # create copy to lock
    cl = cg.copy(c)

    # parse popcount
    p = logic.popcount(width)

    # find output with large enough fanin
    potential_outs = [o for o in cl.outputs()
                      if len(cl.startpoints(o)) >= width]
    if not potential_outs:
        print('input with too small')
        return None
    out = choice(tuple(potential_outs))

    # create key
    key = {f'key_{i}': choice([True, False])
           for i in range(width)}

    # instantiate and connect hd circuits
    cl.add_subcircuit(p, 'flip_pop')
    cl.add_subcircuit(p, 'restore_pop')

    # connect inputs
    for i, inp in enumerate(sample(tuple(cl.startpoints(out)), width)):
        cl.add(f'key_{i}', 'input')
        cl.add(f'hardcoded_key_{i}', '1' if key[f'key_{i}'] else '0')
        cl.add(f'restore_xor_{i}', 'xor', fanin=[f'key_{i}', inp])
        cl.add(f'flip_xor_{i}', 'xor', fanin=[f'hardcoded_key_{i}', inp])
        cl.connect(f'flip_xor_{i}', f'flip_pop_in_{i}')
        cl.connect(f'restore_xor_{i}', f'restore_pop_in_{i}')

    # connect outputs
    cl.add('flip_out', 'and')
    cl.add('restore_out', 'and')
    for i, v in enumerate(format(hd, f'0{cg.clog2(width)+1}b')[::-1]):
        cl.add(f'hd_{i}', v)
        cl.add(f'restore_out_xnor_{i}', 'xnor',
               fanin=[f'hd_{i}', f'restore_pop_out_{i}'], fanout='restore_out')
        cl.add(f'flip_out_xnor_{i}', 'xnor',
               fanin=[f'hd_{i}', f'flip_pop_out_{i}'], fanout='flip_out')

    # flip output
    out_driver = cl.fanin(out)
    old_out = cl.uid(f"{out}_pre_lock")
    cl = cg.relabel(cl, {out: old_out})
    cl.add(out, "xor", fanin=[old_out, "restore_out", "flip_out"], output=True)

    cg.lint(cl)
    return cl, key


def tt_lock(c, width):
    """
    Locks a circuitgraph with TTLock as outlined in
    M. Yasin, A. Sengupta, B. Schafer, Y. Makris, O. Sinanoglu, and
    J. Rajendran, “What to Lock?: Functional and Parametric Locking,” in
    Great Lakes Symposium on VLSI, pp. 351–356, 2017.

    Parameters
    ----------
    c: circuitgraph.CircuitGraph
            Circuit to lock.
    width: int
            the minimum fanin of the gates to lock.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    # create copy to lock
    cl = cg.copy(c)

    # find output with large enough fanin
    potential_outs = [o for o in cl.outputs()
                      if len(cl.startpoints(o)) >= width]
    if not potential_outs:
        raise ValueError(f'no output with input width >= {width}')
    out = choice(tuple(potential_outs))

    # create key
    key = {f'key_{i}': choice([True, False])
           for i in range(width)}

    # connect comparators
    cl.add('flip_out', 'and')
    cl.add('restore_out', 'and')
    for i, inp in enumerate(sample(tuple(cl.startpoints(out)),
                                          width)):
        cl.add(f'key_{i}', 'input')
        cl.add(f'hardcoded_key_{i}', '1' if key[f'key_{i}'] else '0')
        cl.add(f'restore_xor_{i}', 'xor', fanin=[f'key_{i}', inp],
               fanout='restore_out')
        cl.add(f'flip_xor_{i}', 'xor',
               fanin=[f'hardcoded_key_{i}', inp], fanout='flip_out')

    # flip output
    out_driver = cl.fanin(out)
    old_out = cl.uid(f"{out}_pre_lock")
    cl = cg.relabel(cl, {out: old_out})
    cl.add(out, "xor", fanin=[old_out, "restore_out", "flip_out"], output=True)

    cg.lint(cl)
    return cl, key


def tt_lock_sen(c, width, nsamples=10):
    """
    Locks a circuitgraph with TTLock-Sen as outlined in
    Joseph Sweeney, Marijn J.H. Heule, and Lawrence Pileggi,
    “Sensitivity Analysis of Locked Circuits,” in
    Logic for Programming, Artificial Intelligence and Reasoning
    (LPAR-23), pp. 483-497. EPiC Series in Computing 73, EasyChair.

    Parameters
    ----------
    c: circuitgraph.CircuitGraph
            Circuit to lock.
    width: int
            the minimum fanin of the gates to lock.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    # create copy to lock
    cl = cg.copy(c)

    # find output with large enough fanin
    potential_outs = [o for o in cl.outputs()
                      if len(cl.startpoints(o)) >= width]
    if not potential_outs:
        print('input with too small')
        return None

    # find average sensitivities
    A = {}
    N = {}
    S = {}
    for o in potential_outs:
        # build sensitivity circuit
        s = sensitivity_transform(c, o)
        startpoints = c.startpoints(o)
        s_out = set(o for o in s.outputs() if "difference" in o)

        # est avg sensitivity
        total = 0
        for i in range(nsamples):
            input_val = {i: randint(0, 1) for i in startpoints}
            model = sat(s, input_val)
            sen = sum(model[o] for o in s_out)
            total += sen
        A[o] = int(total/nsamples)
        N[o] = len(startpoints)
        S[o] = s

    # find output + input value with closest to avg sen
    def find_input():
        b = 0
        while b < max(N.values()):
            for o in potential_outs:
                upper = min(N[o],int(N[o]-A[o]+b))
                lower = max(0,int(N[o]-A[o]-b))
                us = cg.int_to_bin(upper, cg.clog2(N[o]))
                ls = cg.int_to_bin(lower, cg.clog2(N[o]))
                for sv in [us,ls]:
                    model = sat(S[o], {f"out_{i}": v for i, v in enumerate(sv)})
                    if model:
                        out = o
                        startpoints = c.startpoints(o)

                        key = {f'key_{i}': model[n] for i,n in enumerate(startpoints)}
                        return key, startpoints, out
            b += 1
            print(b)

    key, startpoints, out = find_input()

    # connect comparators
    cl.add('flip_out', 'and')
    cl.add('restore_out', 'and')
    for i, inp in enumerate(startpoints):
        cl.add(f'key_{i}', 'input')
        cl.add(f'hardcoded_key_{i}', '1' if key[f'key_{i}'] else '0')
        cl.add(f'restore_xor_{i}', 'xor', fanin=[f'key_{i}', inp],
               fanout='restore_out')
        cl.add(f'flip_xor_{i}', 'xor',
               fanin=[f'hardcoded_key_{i}', inp], fanout='flip_out')

    # flip output
    out_driver = cl.fanin(out)
    old_out = cl.uid(f"{out}_pre_lock")
    cl = cg.relabel(cl, {out: old_out})
    cl.add(out, "xor", fanin=[old_out, "restore_out", "flip_out"], output=True)

    cg.lint(cl)
    return cl, key


def sfll_flex(c, width, n):
    """
    Locks a circuitgraph with SFLL-flex as outlined in
    Muhammad Yasin, Abhrajit Sengupta, Mohammed Thari Nabeel, Mohammed Ashraf,
    Jeyavijayan (JV) Rajendran, and Ozgur Sinanoglu. 2017. Provably-Secure
    Logic Locking: From Theory To Practice. In Proceedings of the 2017 ACM
    SIGSAC Conference on Computer and Communications Security (CCS ’17).
    Association for Computing Machinery, New York, NY, USA, 1601–1618.

    Parameters
    ----------
    c: circuitgraph.CircuitGraph
            Circuit to lock.
    width: int
            the minimum fanin of the gates to lock.
    n: FIXME

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    # create copy to lock
    cl = cg.copy(c)

    # find output with large enough fanin
    potential_outs = [o for o in cl.outputs()
                      if len(cl.startpoints(o)) >= width]
    if not potential_outs:
        print('input with too small')
        return None
    out = choice(tuple(potential_outs))

    # create key
    key = {f'key_{i}': choice([True, False])
           for i in range(width*n)}

    # connect comparators
    cl.add('flip_out', 'or')
    cl.add('restore_out', 'or')

    for j in range(n):
        cl.add(f'flip_and_{j}', 'and', fanout='flip_out')
        cl.add(f'restore_and_{j}', 'and', fanout='restore_out')

    for i, inp in enumerate(sample(tuple(cl.startpoints(out)), width)):
        for j in range(n):
            cl.add(f'key_{i+j*width}', 'input')
            cl.add(f'hardcoded_key_{i}_{j}',
                   '1' if key[f'key_{i+j*width}'] else '0')
            cl.add(f'restore_xor_{i}_{j}', 'xor',
                   fanin=[f'key_{i+j*width}', inp],
                   fanout=f'restore_and_{j}')
            cl.add(f'flip_xor_{i}_{j}', 'xor',
                   fanin=[f'hardcoded_key_{i}_{j}', inp],
                   fanout=f'flip_and_{j}')

    # flip output
    out_driver = cl.fanin(out)
    old_out = cl.uid(f"{out}_pre_lock")
    cl = cg.relabel(cl, {out: old_out})
    cl.add(out, "xor", fanin=[old_out, "restore_out", "flip_out"], output=True)

    cg.lint(cl)
    return cl, key


def connect_banyan(cl, swb_ins, swb_outs, bw):
    I = int(2*cg.clog2(bw)-2)
    J = int(bw/2)
    for i in range(cg.clog2(J)):
        r = J/(2**i)
        for j in range(J):
            t = (j % r) >= (r/2)
            # straight
            out_i = int((i*bw)+(2*j)+t)
            in_i = int((i*bw+bw)+(2*j)+t)
            cl.connect(swb_outs[out_i], swb_ins[in_i])

            # cross
            out_i = int((i*bw)+(2*j)+(1-t)+((r-1)*((1-t)*2-1)))
            in_i = int((i*bw+bw)+(2*j)+(1-t))
            cl.connect(swb_outs[out_i], swb_ins[in_i])

            if r > 2:
                # straight
                out_i = int(((I*J*2)-((2+i)*bw))+(2*j)+t)
                in_i = int(((I*J*2)-((1+i)*bw))+(2*j)+t)
                cl.connect(swb_outs[out_i], swb_ins[in_i])

                # cross
                out_i = int(((I*J*2)-((2+i)*bw)) +
                            (2*j)+(1-t)+((r-1)*((1-t)*2-1)))
                in_i = int(((I*J*2)-((1+i)*bw))+(2*j)+(1-t))
                cl.connect(swb_outs[out_i], swb_ins[in_i])


def connect_banyan_bb(cl, swb_ins, swb_outs, bw):
    I = int(2*cg.clog2(bw)-2)
    J = int(bw/2)
    for i in range(cg.clog2(J)):
        r = J/(2**i)
        for j in range(J):
            t = (j % r) >= (r/2)
            # straight
            out_i = int((i*bw)+(2*j)+t)
            in_i = int((i*bw+bw)+(2*j)+t)
            cl.add(f"swb_{i}_{j}_straight", "buf", fanin=swb_outs[out_i], fanout=swb_ins[in_i])

            # cross
            out_i = int((i*bw)+(2*j)+(1-t)+((r-1)*((1-t)*2-1)))
            in_i = int((i*bw+bw)+(2*j)+(1-t))
            cl.add(f"swb_{i}_{j}_cross", "buf", fanin=swb_outs[out_i], fanout=swb_ins[in_i])

            if r > 2:
                # straight
                out_i = int(((I*J*2)-((2+i)*bw))+(2*j)+t)
                in_i = int(((I*J*2)-((1+i)*bw))+(2*j)+t)
                cl.add(f"swb_{i}_{j}_r_straight", "buf", fanin=swb_outs[out_i], fanout=swb_ins[in_i])

                # cross
                out_i = int(((I*J*2)-((2+i)*bw)) +
                            (2*j)+(1-t)+((r-1)*((1-t)*2-1)))
                in_i = int(((I*J*2)-((1+i)*bw))+(2*j)+(1-t))
                cl.add(f"swb_{i}_{j}_r_cross", "buf", fanin=swb_outs[out_i], fanout=swb_ins[in_i])


def full_lock(c, bw, lw):
    """
    Locks a circuitgraph with Full-Lock as outlined in
    Hadi Mardani Kamali, Kimia Zamiri Azar, Houman Homayoun, and Avesta Sasan.
    2019. Full-Lock: Hard Distributions of SAT instances for Obfuscating
    Circuits using Fully Configurable Logic and Routing Blocks. In Proceedings
    of the 56th Annual Design Automation Conference 2019 (DAC ’19).
    Association for Computing Machinery, New York, NY, USA, Article 89, 1–6.

    Parameters
    ----------
    circuit: circuitgraph.CircuitGraph
            Circuit to lock.
    banyan_width: int
            Width of Banyan network to use, must follow bw = 2**n, n>1.
    lut_width: int
            Width to use for inserted LUTs, must evenly divide bw.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    # lock with luts
    cl, key = random_lut_lock(c, int(bw/lw), lw)

    # generate switch
    m = cg.strip_io(logic.mux(2))
    s = cg.Circuit(name='switch')
    s.add_subcircuit(m, f'm0')
    s.add_subcircuit(m, f'm1')
    s.add('in_0', 'buf', fanout=['m0_in_0', 'm1_in_1'])
    s.add('in_1', 'buf', fanout=['m0_in_1', 'm1_in_0'])
    s.add('out_0', 'xor', fanin='m0_out')
    s.add('out_1', 'xor', fanin='m1_out')
    s.add('key_0', 'input', fanout=['m0_sel_0', 'm1_sel_0'])
    s.add('key_1', 'input', fanout='out_0')
    s.add('key_2', 'input', fanout='out_1')

    # generate banyan
    I = int(2*cg.clog2(bw)-2)
    J = int(bw/2)

    # add switches
    for i in range(I*J):
        cl.add_subcircuit(s, f'swb_{i}')

    # make connections
    swb_ins = [f'swb_{i//2}_in_{i%2}' for i in range(I*J*2)]
    swb_outs = [f'swb_{i//2}_out_{i%2}' for i in range(I*J*2)]
    connect_banyan(cl, swb_ins, swb_outs, bw)

    # get banyan io
    net_ins = swb_ins[:bw]
    net_outs = swb_outs[-bw:]

    # generate key
    for i in range(I*J):
        for j in range(3):
            key[f'swb_{i}_key_{j}'] = choice([True, False])

    # get banyan mapping
    mapping = {}
    polarity = {}
    orig_result = sat(cl, {**{n: False for n in net_ins}, **key})
    for net_in in net_ins:
        result = sat(
            cl, {**{n: False if n != net_in else True for n in net_ins}, **key}
        )
        for net_out in net_outs:
            if result[net_out] != orig_result[net_out]:
                mapping[net_in] = net_out
                polarity[net_in] = result[net_out]
                break

    # connect banyan io to luts
    for i in range(int(bw/lw)):
        for j in range(lw):
            driver = cl.fanin(f'lut_{i}_sel_{j}').pop()
            cl.disconnect(driver, f'lut_{i}_sel_{j}')
            net_in = net_ins[i*lw+j]
            cl.connect(mapping[net_in], f'lut_{i}_sel_{j}')
            if not polarity[net_in]:
                driver = cl.add(f'not_{net_in}', 'not', fanin=driver)
            cl.connect(driver, net_in)

    for k in key:
        cl.set_type(k, "input")

    cg.lint(cl)
    return cl, key

def full_lock_mux(c, bw, lw):
    """
    Locks a circuitgraph with Full-Lock as outlined in
    Hadi Mardani Kamali, Kimia Zamiri Azar, Houman Homayoun, and Avesta Sasan.
    2019. Full-Lock: Hard Distributions of SAT instances for Obfuscating
    Circuits using Fully Configurable Logic and Routing Blocks. In Proceedings
    of the 56th Annual Design Automation Conference 2019 (DAC ’19).
    Association for Computing Machinery, New York, NY, USA, Article 89, 1–6.

    But, uses muxes instead of the Banyan network, a relaxation that breaks symmetry
    and simplifies the model substantially. This process is outlined in
    Joseph Sweeney, Marijn J.H. Heule, and Lawrence Pileggi
    Modeling Techniques for Logic Locking. In Proceedings
    of the International Conference on Computer Aided Design 2020 (ICCAD-39).


    Parameters
    ----------
    c: circuitgraph.CircuitGraph
            Circuit to lock.
    banyan_width: int
            Width of Banyan network to use, must follow bw = 2**n, n>1.
    lut_width: int
            Width to use for inserted LUTs, must evenly divide bw.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    # first generate banyan, to get a valid mapping for the key
    b = cg.Circuit()

    # generate switch
    m = cg.strip_io(logic.mux(2))
    s = cg.Circuit(name='switch')
    s.add_subcircuit(m, f'm0')
    s.add_subcircuit(m, f'm1')
    s.add('in_0', 'buf', fanout=['m0_in_0', 'm1_in_1'])
    s.add('in_1', 'buf', fanout=['m0_in_1', 'm1_in_0'])
    s.add('out_0', 'xor', fanin='m0_out')
    s.add('out_1', 'xor', fanin='m1_out')
    s.add('key_0', 'input', fanout=['m0_sel_0', 'm1_sel_0'])
    s.add('key_1', 'input', fanout='out_0')
    s.add('key_2', 'input', fanout='out_1')

    # generate banyan
    I = int(2*cg.clog2(bw)-2)
    J = int(bw/2)

    # add switches
    for i in range(I*J):
        b.add_subcircuit(s, f'swb_{i}')

    # make connections
    swb_ins = [f'swb_{i//2}_in_{i%2}' for i in range(I*J*2)]
    swb_outs = [f'swb_{i//2}_out_{i%2}' for i in range(I*J*2)]
    connect_banyan(b, swb_ins, swb_outs, bw)

    # get banyan io
    net_ins = swb_ins[:bw]
    net_outs = swb_outs[-bw:]

    # generate key
    key = {}
    for i in range(I*J):
        for j in range(3):
            key[f'swb_{i}_key_{j}'] = choice([True, False])

    # get banyan mapping
    mapping = {}
    polarity = {}
    orig_result = sat(b, {**{n: False for n in net_ins}, **key})
    for net_in in net_ins:
        result = sat(
            b, {**{n: False if n != net_in else True for n in net_ins}, **key}
        )
        for net_out in net_outs:
            if result[net_out] != orig_result[net_out]:
                mapping[net_in] = net_out
                polarity[net_in] = result[net_out]
                break

    # lock with luts
    cl, key = random_lut_lock(c, int(bw/lw), lw)

    # generate mux
    m = cg.strip_io(logic.mux(bw))

    # add muxes and xors
    banyan_to_mux = {}
    for i in range(bw):
        cl.add_subcircuit(m, f'mux_{i}')
        for b in range(cg.clog2(bw)):
            cl.add(f'key_{i}_{b}', 'input', fanout=f'mux_{i}_sel_{b}')
        cl.add(f'mux_{i}_xor','xor',fanin=f'mux_{i}_out')
        cl.add(f'key_{i}_{cg.clog2(bw)}', 'input', fanout=f'mux_{i}_xor')
        banyan_to_mux[net_outs[i]] = f'mux_{i}_xor'

    # connect muxes to luts
    for i in range(bw):
        net_in = net_ins[i]
        xor = banyan_to_mux[mapping[net_in]]
        o = int(xor.split('_')[1])

        driver = cl.fanin(f'lut_{i//lw}_sel_{i%lw}').pop()
        cl.disconnect(driver, f'lut_{i//lw}_sel_{i%lw}')

        if not polarity[net_in]:
            driver = cl.add(f'not_{net_in}', 'not', fanin=driver)
            key[f'key_{o}_{cg.clog2(bw)}'] = True
        else:
            key[f'key_{o}_{cg.clog2(bw)}'] = False

        for b in range(bw):
            cl.connect(driver, f'mux_{b}_in_{i}')

        cl.connect(xor,f'lut_{i//lw}_sel_{i%lw}')
        for b,v in enumerate(cg.int_to_bin(i, cg.clog2(bw),True)):
            key[f'key_{o}_{b}'] = v

    cg.lint(cl)
    return cl, key


def inter_lock(c, bw):
    """
    Locks a circuitgraph with InterLock as outlined in
    Kamali, Hadi Mardani, Kimia Zamiri Azar, Houman Homayoun, and Avesta Sasan.
    "Interlock: An intercorrelated logic and routing locking."
    In 2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD),
    pp. 1-9. IEEE, 2020.

    Parameters
    ----------
    circuit: circuitgraph.CircuitGraph
            Circuit to lock.
    bw: int
            The size of the keyed rounting block. A bw of m results
            in an m x m sized KeyRB.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    cl = cg.copy(c)
    cg.lint(cl)

    # generate switch
    m = cg.strip_io(logic.mux(2))
    s = cg.Circuit(name='switch')
    s.add_subcircuit(m, f'm0')
    s.add_subcircuit(m, f'm1')
    s.add_subcircuit(m, f'm2')
    s.add_subcircuit(m, f'm3')
    s.add('in_0', 'input', fanout=['m0_in_0', 'm1_in_1'])
    s.add('in_1', 'input', fanout=['m0_in_1', 'm1_in_0'])
    s.add('ex_in_0', 'input')
    s.add('ex_in_1', 'input')
    # f1 and f2 starts as and gates, must be updated later
    s.add('f1_out', 'and', fanin=['m0_out', 'ex_in_0'], fanout='m2_in_0')
    s.add('f2_out', 'and', fanin=['m1_out', 'ex_in_1'], fanout='m3_in_1')
    s.connect('m0_out', 'm3_in_0')
    s.connect('m1_out', 'm2_in_1')
    s.add('key_0', 'input', fanout=['m0_sel_0', 'm1_sel_0'])
    s.add('key_1', 'input', fanout='m2_sel_0')
    s.add('key_2', 'input', fanout='m3_sel_0')
    s.add('out_0', 'buf', fanin='m2_out', output=True)
    s.add('out_1', 'buf', fanin='m3_out', output=True)

    sbb = cg.BlackBox("switch",
                      ["in_0", "in_1", "ex_in_0", "ex_in_1", "key_0", "key_1", "key_2"],
                      ["out_0", "out_1"])

    # Select paths to embed in the routing network
    path_length = 2 * cg.clog2(bw) - 2
    paths = []
    locked_gates = set()
    
    filtered_gates = set()
    def filter_gate(n):
        gate = n
        gates = [n]
        for _ in range(path_length):
            if (len(cl.fanin(gate)) != 2 or len(cl.fanout(gate)) != 1 or
                    gate in filtered_gates or 
                    len(cl.fanin(gate) & filtered_gates) > 0 or
                    len(cl.fanout(gate) & filtered_gates) > 0):
                return False
            gate = cl.fanout(gate).pop()
            gates.append(gate)
        filtered_gates.update(gates)
        for gate in gates:
            filtered_gates.update(cl.fanin(gate))
        return True

    candidate_gates = filter(filter_gate, cl.nodes())
    for _ in range(bw):
        try:
            gate = next(candidate_gates)
        except StopIteration:
            raise ValueError('Not enough candidate gates found for locking')
        path = [gate]
        for _ in range(path_length - 1):
            gate = cl.fanout(gate).pop()
            path.append(gate)
        paths.append(path)

    # generate banyan with J rows and I columns of SwBs
    I = path_length
    J = int(bw/2)

    for i in range(I*J):
        cl.add_blackbox(sbb,
                        f"swb_{i}")

    # make connections
    swb_ins = [f'swb_{i//2}.in_{i%2}' for i in range(I*J*2)]
    swb_outs = [f'swb_{i//2}.out_{i%2}' for i in range(I*J*2)]
    connect_banyan_bb(cl, swb_ins, swb_outs, bw)

    # get banyan io
    net_ins = swb_ins[:bw]
    net_outs = swb_outs[-bw:]

    # generate key
    # In the example from the paper, the paths in a SWB directly from an
    # input to an output are never used. Starting with that implemetation.
    # Could sometimes choose paths less than `path_length` and use these
    # connections with a decoy external input, but such a strategy is not
    # discussed in the paper.
    swaps = []
    key = {}
    for i in range(I*J):
        swaps.append(choice([True, False]))
        if swaps[-1]:
            key[f'swb_{i}_key_0'] = True
        else:
            key[f'swb_{i}_key_0'] = False
        key[f'swb_{i}_key_1'] = False
        key[f'swb_{i}_key_2'] = True

    f_gates = dict()

    # Add paths to banyan 
    # Get a random intial ordering of paths
    input_order = list(range(bw))
    shuffle(input_order)
    for i, p_idx in enumerate(input_order):
        path = paths[p_idx]
        swb_idx = (i // 2)
        i_idx = i % 2
        prev_node = cl.fanin(path[0]).pop()
        cl.connect(prev_node, f'swb_{swb_idx}.in_{i_idx}')
        for j, n in enumerate(path):
            o_idx = i_idx ^ int(swaps[swb_idx])
            ex_i = (cl.fanin(n) - {prev_node}).pop()
            cl.connect(ex_i, f'swb_{swb_idx}.ex_in_{o_idx}')
            f_gates[f"swb_{swb_idx}_f{o_idx+1}_out"] = cl.type(n)
            if j != len(path) - 1:
                next_n = cl.fanout(f'swb_{swb_idx}.out_{o_idx}').pop()
                next_n = cl.fanout(next_n).pop()
                swb_idx = int(next_n.split(".")[0].split('_')[-1])
                i_idx = int(next_n.split(".")[-1].split('_')[-1])
                prev_node = n
            else:
                for fo in cl.fanout(n):
                    cl.disconnect(n, fo)
                    try:
                        conn = cl.fanout(f"swb_{swb_idx}.out_{o_idx}").pop()
                    except KeyError:
                        conn = cl.add(f"swb_{swb_idx}.out_{o_idx}_load",
                                      "buf",
                                      fanin=f"swb_{swb_idx}.out_{o_idx}")
                    cl.connect(conn, fo)

    for path in paths:
        for node in path:
            cl.remove(node)

    for i in range(I*J):
        cl.fill_blackbox(f"swb_{i}", s)

    for k, v in f_gates.items():
        cl.set_type(k, v)

    for k in key:
        cl.set_type(k, "input")

    # cg.lint(cl)
    return cl, key


def lebl(c,bw,ng):
    """
    Locks a circuitgraph with Logic-Enhanced Banyan Locking as outlined in
    Joseph Sweeney, Marijn J.H. Heule, and Lawrence Pileggi
    Modeling Techniques for Logic Locking. In Proceedings
    of the International Conference on Computer Aided Design 2020 (ICCAD-39).

    Parameters
    ----------
    c: circuitgraph.CircuitGraph
            Circuit to lock.
    bw: int
            Width of Banyan network.
    lw: int
            Minimum number of gates mapped to network.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """
    # create copy to lock
    cl = cg.copy(c)

    # generate switch and mux
    s = cg.Circuit(name='switch')
    m2 = cg.strip_io(logic.mux(2))
    s.add_subcircuit(m2, f'm2_0')
    s.add_subcircuit(m2, f'm2_1')
    m4 = cg.strip_io(logic.mux(4))
    s.add_subcircuit(m4, f'm4_0')
    s.add_subcircuit(m4, f'm4_1')
    s.add('in_0','buf',fanout=['m2_0_in_0','m2_1_in_1'])
    s.add('in_1','buf',fanout=['m2_0_in_1','m2_1_in_0'])
    s.add('out_0','buf',fanin='m4_0_out')
    s.add('out_1','buf',fanin='m4_1_out')
    s.add('key_0','input',fanout=['m2_0_sel_0','m2_1_sel_0'])
    s.add('key_1','input',fanout=['m4_0_sel_0','m4_1_sel_0'])
    s.add('key_2','input',fanout=['m4_0_sel_1','m4_1_sel_1'])

    # generate banyan
    I = int(2*cg.clog2(bw)-2)
    J = int(bw/2)

    # add switches and muxes
    for i in range(I*J):
        cl.add_subcircuit(s, f'swb_{i}')

    # make connections
    swb_ins = [f'swb_{i//2}_in_{i%2}' for i in range(I*J*2)]
    swb_outs = [f'swb_{i//2}_out_{i%2}' for i in range(I*J*2)]
    connect_banyan(cl,swb_ins,swb_outs,bw)

    # get banyan io
    net_ins = swb_ins[:bw]
    net_outs = swb_outs[-bw:]

    # generate key
    key = {f'swb_{i//3}_key_{i%3}':choice([True,False]) for i in range(3*I*J)}

    # generate connections between banyan nodes
    bfi = {n:set() for n in swb_outs+net_ins}
    bfo = {n:set() for n in swb_outs+net_ins}
    for n in swb_outs+net_ins:
        if cl.fanout(n):
            fo_node = cl.fanout(n).pop()
            swb_i = fo_node.split('_')[1]
            bfi[f'swb_{swb_i}_out_0'].add(n)
            bfi[f'swb_{swb_i}_out_1'].add(n)
            bfo[n].add(f'swb_{swb_i}_out_0')
            bfo[n].add(f'swb_{swb_i}_out_1')

    # find a mapping of circuit onto banyan
    net_map = IDPool()
    for bn in swb_outs+net_ins:
        for cn in c:
            net_map.id(f'm_{bn}_{cn}')

    # mapping implications
    clauses = []
    for bn in swb_outs+net_ins:
        # fanin
        if bfi[bn]:
            for cn in c:
                if c.fanin(cn):
                    for fcn in c.fanin(cn):
                        clause = [-net_map.id(f'm_{bn}_{cn}')]
                        clause += [net_map.id(f'm_{fbn}_{fcn}') for fbn in bfi[bn]]
                        clause += [net_map.id(f'm_{fbn}_{cn}') for fbn in bfi[bn]]
                        clauses.append(clause)
                else:
                    clause = [-net_map.id(f'm_{bn}_{cn}')]
                    clause += [net_map.id(f'm_{fbn}_{cn}') for fbn in bfi[bn]]
                    clauses.append(clause)

        # fanout
        if bfo[bn]:
            for cn in c:
                clause = [-net_map.id(f'm_{bn}_{cn}')]
                clause += [net_map.id(f'm_{fbn}_{cn}') for fbn in bfo[bn]]
                for fcn in c.fanout(cn):
                    clause += [net_map.id(f'm_{fbn}_{fcn}') for fbn in bfo[bn]]
                clauses.append(clause)

    # no feed through
    for cn in c:
        net_map.id(f'INPUT_OR_{cn}')
        net_map.id(f'OUTPUT_OR_{cn}')
        clauses.append([-net_map.id(f'INPUT_OR_{cn}')]+[net_map.id(f'm_{bn}_{cn}') for bn in net_ins])
        clauses.append([-net_map.id(f'OUTPUT_OR_{cn}')]+[net_map.id(f'm_{bn}_{cn}') for bn in net_outs])
        for bn in net_ins:
            clauses.append([net_map.id(f'INPUT_OR_{cn}'),-net_map.id(f'm_{bn}_{cn}')])
        for bn in net_outs:
            clauses.append([net_map.id(f'OUTPUT_OR_{cn}'),-net_map.id(f'm_{bn}_{cn}')])
        clauses.append([-net_map.id(f'OUTPUT_OR_{cn}'),-net_map.id(f'INPUT_OR_{cn}')])

    # at least ngates
    for bn in swb_outs+net_ins:
        net_map.id(f'NGATES_OR_{bn}')
        clauses.append([-net_map.id(f'NGATES_OR_{bn}')]+[net_map.id(f'm_{bn}_{cn}') for cn in c])
        for cn in c:
            clauses.append([net_map.id(f'NGATES_OR_{bn}'),-net_map.id(f'm_{bn}_{cn}')])
    clauses += CardEnc.atleast(bound=ng,lits=[net_map.id(f'NGATES_OR_{bn}') for bn in swb_outs+net_ins],vpool=net_map).clauses

    # at most one mapping per out
    for bn in swb_outs+net_ins:
        clauses += CardEnc.atmost(lits=[net_map.id(f'm_{bn}_{cn}') for cn in c],vpool=net_map).clauses

    # limit number of times a gate is mapped to net outputs to fanout of gate
    for cn in c:
        lits = [net_map.id(f'm_{bn}_{cn}') for bn in net_outs]
        bound = len(c.fanout(cn))
        if len(lits)<bound: continue
        clauses += CardEnc.atmost(bound=bound,lits=lits,vpool=net_map).clauses

    # prohibit outputs from net
    for bn in swb_outs+net_ins:
        for cn in c.outputs():
            clauses += [[-net_map.id(f'm_{bn}_{cn}')]]

    # solve
    solver = Cadical(bootstrap_with=clauses)
    if not solver.solve():
        print(f'no config for width: {bw}')
        core = solver.get_core()
        print(core)
        code.interact(local=dict(globals(), **locals()))
    model = solver.get_model()

    # get mapping
    mapping = {}
    for bn in swb_outs+net_ins:
        selected_gates = [cn for cn in c if model[net_map.id(f'm_{bn}_{cn}')-1]>0]
        if len(selected_gates)>1:
            print(f'multiple gates mapped to: {bn}')
            code.interact(local=dict(globals(), **locals()))
        mapping[bn] = selected_gates[0] if selected_gates else None

    potential_net_fanins = list(c.nodes()-(c.endpoints()|set(mapping.values())|mapping.keys()|c.startpoints()))

    # connect net inputs
    for bn in net_ins:
        if mapping[bn]:
            cl.connect(mapping[bn],bn)
        else:
            cl.connect(choice(potential_net_fanins),bn)
    mapping.update({cl.fanin(bn).pop():cl.fanin(bn).pop() for bn in net_ins})
    potential_net_fanouts = list(c.nodes()-(c.startpoints()|set(mapping.values())|mapping.keys()|c.endpoints()))

    #selected_fo = {}

    # connect switch boxes
    for i,bn in enumerate(swb_outs):
        # get keys
        if key[f'swb_{i//2}_key_1'] and key[f'swb_{i//2}_key_2']:
            k = 3
        elif not key[f'swb_{i//2}_key_1'] and key[f'swb_{i//2}_key_2']:
            k = 2
        elif key[f'swb_{i//2}_key_1'] and not key[f'swb_{i//2}_key_2']:
            k = 1
        elif not key[f'swb_{i//2}_key_1'] and not key[f'swb_{i//2}_key_2']:
            k = 0
        switch_key = 1 if key[f'swb_{i//2}_key_0']==1 else 0

        mux_input = f'swb_{i//2}_m4_{i%2}_in_{k}'

        # connect inner nodes
        mux_gate_types = set()

        # constant output, hookup to a node that is already in the affected outputs fanin, not in others
        if not mapping[bn] and bn in net_outs:
            decoy_fanout_gate = choice(potential_net_fanouts)
            #selected_fo[bn] = decoy_fanout_gate
            if cl.type(decoy_fanout_gate) in ['and','nand']:
                cl.set_type(mux_input,'1')
            elif cl.type(decoy_fanout_gate) in ['or','nor','xor','xnor']:
                cl.set_type(mux_input,'0')
            elif cl.type(decoy_fanout_gate) in ['buf']:
                if randint(0,1):
                    cl.set_type(mux_input,'1')
                    cl.set_type(decoy_fanout_gate,choice(['and','xnor']))
                else:
                    cl.set_type(mux_input,'0')
                    cl.set_type(decoy_fanout_gate,choice(['or','xor']))
            elif cl.type(decoy_fanout_gate) in ['not']:
                if randint(0,1):
                    cl.set_type(mux_input,'1')
                    cl.set_type(decoy_fanout_gate,choice(['nand','xor']))
                else:
                    cl.set_type(mux_input,'0')
                    cl.set_type(decoy_fanout_gate,choice(['nor','xnor']))
            elif cl.type(decoy_fanout_gate) in ['0','1']:
                cl.set_type(mux_input,cl.type(decoy_fanout_gate))
                cl.set_type(decoy_fanout_gate,'buf')
            else:
                print('gate error')
                code.interact(local=dict(globals(), **locals()))
            cl.connect(bn,decoy_fanout_gate)
            mux_gate_types.add(cl.type(mux_input))

        # feedthrough
        elif mapping[bn] in [mapping[fbn] for fbn in bfi[bn]]:
            cl.set_type(mux_input,'buf')
            mux_gate_types.add('buf')
            if mapping[cl.fanin(f'swb_{i//2}_in_0').pop()]==mapping[bn]:
                cl.connect(f'swb_{i//2}_m2_{switch_key}_out',mux_input)
            else:
                cl.connect(f'swb_{i//2}_m2_{1-switch_key}_out',mux_input)

        # gate
        elif mapping[bn]:
            cl.set_type(mux_input,cl.type(mapping[bn]))
            mux_gate_types.add(cl.type(mapping[bn]))
            gfi = cl.fanin(mapping[bn])
            if mapping[cl.fanin(f'swb_{i//2}_in_0').pop()] in gfi:
                cl.connect(f'swb_{i//2}_m2_{switch_key}_out',mux_input)
                gfi.remove(mapping[cl.fanin(f'swb_{i//2}_in_0').pop()])
            if mapping[cl.fanin(f'swb_{i//2}_in_1').pop()] in gfi:
                cl.connect(f'swb_{i//2}_m2_{1-switch_key}_out',mux_input)

        # mapped to None, any key works
        else:
            k = None

        # fill out random gates
        for j in range(4):
            if j != k:
                t = choice(tuple(set(['buf','or','nor','and','nand','not','xor','xnor','0','1'])-mux_gate_types))
                mux_gate_types.add(t)
                mux_input = f'swb_{i//2}_m4_{i%2}_in_{j}'
                cl.set_type(mux_input,t)
                if t=='not' or t=='buf':
                    # pick a random fanin
                    cl.connect(f'swb_{i//2}_m2_{randint(0,1)}_out',mux_input)
                elif t=='1' or t=='0':
                    pass
                else:
                    cl.connect(f'swb_{i//2}_m2_0_out',mux_input)
                    cl.connect(f'swb_{i//2}_m2_1_out',mux_input)
        if [n for n in cl if cl.type(n) in ['buf','not'] and len(cl.fanin(n))>1]:
            import code
            code.interact(local=dict(globals(), **locals()))

    # connect outputs non constant outs
    rev_mapping = {}
    for bn in net_outs:
        if mapping[bn]:
            if mapping[bn] not in rev_mapping:
                rev_mapping[mapping[bn]] = set()
            rev_mapping[mapping[bn]].add(bn)

    for cn in rev_mapping.keys():
        #for fcn in cl.fanout(cn):
        #    cl.connect(sample(rev_mapping[cn],1)[0],fcn)
        for fcn,bn in zip_longest(cl.fanout(cn),rev_mapping[cn],fillvalue=list(rev_mapping[cn])[0]):
            cl.connect(bn,fcn)

    # delete mapped gates
    deleted = True
    while deleted:
        deleted = False
        for n in cl.nodes():
            # node and all fanout are in the net
            if n not in mapping and n in mapping.values():
                if all(s not in mapping and s in mapping.values() for s in cl.fanout(n)):
                    cl.remove(n)
                    deleted = True
            # node in net fanout
            if n in [mapping[o] for o in net_outs] and n in cl:
                cl.remove(n)
                deleted = True

    for k in key:
        cl.set_type(k,"input")

    cg.lint(cl)
    return cl, key

def uc_gate(w_gate):
    # LUT + input muxes
    g = cg.Circuit()

    # add mux that will function as LUT
    l = cg.mux(4)
    g.add_subcircuit(l,'lut')

    # add input muxes
    m = cg.mux(w_gate)
    g.add_subcircuit(m,f'mux0')
    g.connect('mux0_out','lut_sel_0')
    g.add_subcircuit(m,f'mux1')
    g.connect('mux1_out','lut_sel_1')

    # connect inputs to muxes
    for i in range(w_gate):
        g.add(f'in_{i}','input',fanout=[f'mux0_in_{i}',f'mux1_in_{i}'])

    # connect keys to muxes
    for i in range(cg.clog2(w_gate)):
        g.add(f'key_sel_0_{i}','input',fanout=f'mux0_sel_{i}')
        g.add(f'key_sel_1_{i}','input',fanout=f'mux1_sel_{i}')

    # connect keys to lut
    for i in range(4):
        g.add(f'key_lut_{i}','input',fanout=f'lut_in_{i}')

    # add output
    g.add('out','output',fanin='lut_out')
    return g

def gen_uc(n,m,w,d):
    uc = cg.Circuit()

    # initialize possible gate inputs as UC inputs
    gate_inputs = [uc.add(f'in_{i}','input') for i in range(n)]

    # generate uc gates for each x,y location
    for x in range(1,d+1):
        # uc gate input width is length of inputs+all previous layer outputs
        g = uc_gate(n+(x-1)*w)
        for y in range(w):
            # add gate copy
            uc.add_subcircuit(g,f'uc_gate_{x}_{y}')
            # connect all inputs
            for i,gi in enumerate(gate_inputs):
                uc.connect(gi,f'uc_gate_{x}_{y}_in_{i}')

        # add layer x outputs to possible gate inputs
        gate_inputs += [f'uc_gate_{x}_{y}_out' for y in range(w)]

    #make keys from uc_gates as primary inputs
    key = {n for n in uc if 'key' in n}
    for k in key:
        uc.set_type(k,"input")

    # connect output muxes
    mx = cg.mux(len(gate_inputs))
    for o in range(m):
        # add mux
        uc.add_subcircuit(mx,f'output_mux_{o}')
        # connect inputs
        for i,gi in enumerate(gate_inputs):
            uc.connect(gi,f'output_mux_{o}_in_{i}')
        # connect output
        uc.add(f'out_{o}','output',fanin=f'output_mux_{o}_out')
        # connect keys
        for i in range(cg.clog2(len(gate_inputs))):
            uc.add(f'key_output_mux_{o}_sel_{i}','input',
                  fanout=f'output_mux_{o}_sel_{i}')
    return uc

def uc_lock(c):
    """
    Locks a circuitgraph with a universal circuit.

    Parameters
    ----------
    c: circuitgraph.CircuitGraph
            Circuit to lock.

    Returns
    -------
    circuitgraph.CircuitGraph, dict of str:bool
            the locked circuit and the correct key value for each key input
    """

    # convert to 2 input gates
    c2 = cg.limit_fanin(c, 2)

    # determine node layers
    layers = {} # x coordinate of gate
    for n in c2.topo_sort():
        if c2.type(n) == 'output':
            continue
        fi = c2.fanin(n)
        if fi:
            layers[n] = max(layers[f] for f in fi) + 1
        else:
            layers[n] = 0
    out_layer = max(layers.values()) + 1
    for n in c2.outputs():
        layers[n] = out_layer

    # determine width, assign gate locations
    layer_counts = {} # how many gates on layer
    layer_orders = {} # y coordinate of gate
    for g,l in layers.items():
        if l not in layer_counts:
            layer_counts[l] = 1
        else:
            layer_counts[l] += 1
        layer_orders[g] = layer_counts[l]-1
    w = max(layer_counts.values()) #width
    d = len(layer_counts)-1 #doesn't include input layer
    n = len(c2.inputs()) #number of inputs
    m = len(c2.outputs()) #number of outputs

    # create UC
    uc = gen_uc(n,m,w,d)
    key = {n:False for n in uc if 'key' in n} # set all to 0

    # table for lut programming
    lut_vals = {'buf': [False,True,False,True],
                'not': [True,False,True,False],
                'xor': [False,True,True,False],
                'xnor':[True,False,False,True],
                'and': [False,False,False,True],
                'nand': [True,True,True,False],
                'nor': [True,False,False,False],
                'or': [False,True,True,True]}

    # program
    for node in c2:
        x = layers[node]
        y = layer_orders[node]

        if c2.type(node)=='output':
            # set mux keys
            fi = c2.fanin(node).pop()
            fi_y = layer_orders[fi]
            fi_x = layers[fi]
            i = fi_x*w + fi_y

            x_out = max(layers.values()) #last layer
            w_out_gate = cg.clog2(n+(x_out)*w)
            for i,v in enumerate(cg.int_to_bin(i,w_out_gate,lend=True)):
                key[f'key_output_mux_{y}_sel_{i}'] = v

            # rename out
            uc.relabel({f'out_{y}':node})

        elif c2.type(node)=='input':
            # rename to original circuit input names
            uc.relabel({f'in_{y}':node})

        else:
            # add lut vals
            for i,v in enumerate(lut_vals[c2.type(node)]):
                key[f'uc_gate_{x}_{y}_key_lut_{i}'] = v

            # set mux keys to select fanin
            w_uc_gate = cg.clog2(n+(x-1)*w) #should be n for x=1
            for fi_i,fi in enumerate(c2.fanin(node)):
                fi_x = layers[fi]
                fi_y = layer_orders[fi]
                i = fi_x*w + fi_y
                for i,v in enumerate(cg.int_to_bin(i, w_uc_gate, lend=True)):
                    key[f'uc_gate_{x}_{y}_key_sel_{fi_i}_{i}'] = v

    return uc, key
