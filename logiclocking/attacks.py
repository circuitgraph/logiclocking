"""Run attacks on logic-locked circuits."""
import code
import random
import time

import circuitgraph as cg


def _localtime():
    return time.asctime(time.localtime(time.time()))


def miter_attack(
    cl,
    key,
    timeout=None,
    key_cons=None,
    unroll_cyclic=True,
    verbose=True,
    code_on_error=False,
):
    """
    Launch a miter-based sat attack on a locked circuit.

    Parameters
    ----------
    cl: circuitgraph.Circuit
            The locked circuit to attack
    key: dict of str:bool
            The correct key, used to construct the oracle
    timeout: int
            Timeout for the attack, in seconds
    key_cons: circuitgraph.Circuit or iter of circuitgraph.Circuit
            Key conditions to satisfy during attack,
            must have output 'sat' and be a function of the key inputs
    unroll_cyclic: bool
            If True, convert cyclic circuits to acyclic versions
    verbose: bool
            If True, attack progress will be printed
    code_on_error: bool
            If True, drop into an interactive session on an error

    Returns
    -------
    dict
            A dictionary containing attack info and results

    """
    start_time = time.time()

    if cl.is_cyclic():
        if unroll_cyclic:
            cl = cg.tx.acyclic_unroll(cl)
        else:
            raise ValueError(
                "Circuit is cyclic. Set 'unroll_cyclic' to True to run sat on "
                "this circuit"
            )

    # setup vars
    keys = tuple(key.keys())
    ins = tuple(cl.startpoints() - key.keys())
    outs = tuple(cl.endpoints())

    # create simulation solver
    s_sim, v_sim = cg.sat.construct_solver(cl, key)

    # create miter solver
    m = cg.tx.miter(cl, startpoints=set(ins))
    s_miter, v_miter = cg.sat.construct_solver(m)

    # add key constraints
    if key_cons:
        if isinstance(key_cons, cg.Circuit):
            key_cons = [key_cons]
        for key_con in key_cons:
            if verbose:
                print(
                    f"[{_localtime()}] circuit: {cl.name}, "
                    f"adding constraints: {key_con.name}"
                )
            formula, v_cons = cg.sat.cnf(key_con)
            con_clauses = formula.clauses

            # add constraints circuits
            c0_offset = s_miter.nof_vars()
            c0 = cg.sat.remap(con_clauses, c0_offset)
            s_miter.append_formula(c0)
            c1_offset = s_miter.nof_vars()
            c1 = cg.sat.remap(con_clauses, c1_offset)
            s_miter.append_formula(c1)

            # encode keys connections
            clauses = [[v_cons.id("sat") + c0_offset], [v_cons.id("sat") + c1_offset]]
            clauses += [
                [-v_miter.id(f"c0_{n}"), v_cons.id(n) + c0_offset] for n in keys
            ]
            clauses += [
                [v_miter.id(f"c0_{n}"), -v_cons.id(n) - c0_offset] for n in keys
            ]
            clauses += [
                [-v_miter.id(f"c1_{n}"), v_cons.id(n) + c1_offset] for n in keys
            ]
            clauses += [
                [v_miter.id(f"c1_{n}"), -v_cons.id(n) - c1_offset] for n in keys
            ]

            s_miter.append_formula(clauses)

    # get circuit clauses
    formula, v_c = cg.sat.cnf(cl)
    clauses = formula.clauses

    # solve
    dis = []
    dos = []
    iter_times = []
    iter_keys = []
    while s_miter.solve(assumptions=[v_miter.id("sat")]):

        # get di
        model = s_miter.get_model()
        di = [model[v_miter.id(n) - 1] > 0 for n in ins]
        if tuple(di) in dis:
            if code_on_error:
                print("Error di")
                code.interact(local=dict(globals(), **locals()))
            else:
                raise ValueError("Saw same di twice")

        # get intermediate keys
        k0 = {n: model[v_miter.id(f"c0_{n}") - 1] > 0 for n in keys}
        k1 = {n: model[v_miter.id(f"c1_{n}") - 1] > 0 for n in keys}
        iter_keys.append((k0, k1))

        # get do
        s_sim.solve(assumptions=[(2 * b - 1) * v_sim.id(n) for b, n in zip(di, ins)])
        model = s_sim.get_model()
        if model is None:
            if code_on_error:
                print("Error sim")
                code.interact(local=dict(globals(), **locals()))
            else:
                raise ValueError("Could not get simulation model")
        do = [model[v_sim.id(n) - 1] > 0 for n in outs]
        dis.append(tuple(di))
        dos.append(tuple(do))
        iter_times.append(time.time() - start_time)

        # add constraints circuits
        c0_offset = s_miter.nof_vars()
        c0 = cg.sat.remap(clauses, c0_offset)
        s_miter.append_formula(c0)
        c1_offset = s_miter.nof_vars()
        c1 = cg.sat.remap(clauses, c1_offset)
        s_miter.append_formula(c1)

        # encode dis + dos
        dio_clauses = [
            [(2 * b - 1) * (v_c.id(n) + c0_offset)] for b, n in zip(di + do, ins + outs)
        ]
        dio_clauses += [
            [(2 * b - 1) * (v_c.id(n) + c1_offset)] for b, n in zip(di + do, ins + outs)
        ]
        s_miter.append_formula(dio_clauses)

        # encode keys connections
        key_clauses = [[-v_miter.id(f"c0_{n}"), v_c.id(n) + c0_offset] for n in keys]
        key_clauses += [[v_miter.id(f"c0_{n}"), -v_c.id(n) - c0_offset] for n in keys]
        key_clauses += [[-v_miter.id(f"c1_{n}"), v_c.id(n) + c1_offset] for n in keys]
        key_clauses += [[v_miter.id(f"c1_{n}"), -v_c.id(n) - c1_offset] for n in keys]
        s_miter.append_formula(key_clauses)

        # check timeout
        if timeout and (time.time() - start_time) > timeout:
            print(f"[{_localtime()}] circuit: {cl.name}, Timeout: True")
            return {
                "Time": None,
                "Iterations": len(dis),
                "Timeout": True,
                "Equivalent": False,
                "Key Found": False,
                "dis": dis,
                "dos": dos,
                "iter_times": iter_times,
                "iter_keys": iter_keys,
            }

        if verbose:
            print(
                f"[{_localtime()}] "
                f"circuit: {cl.name}, iter: {len(dis)}, "
                f"time: {time.time()-start_time}, "
                f"clauses: {s_miter.nof_clauses()}, "
                f"vars: {s_miter.nof_vars()}"
            )

    # check if a satisfying key remains
    key_found = s_miter.solve()
    if verbose:
        print(f"[{_localtime()}] circuit: {cl.name}, key found: {key_found}")
    if not key_found:
        return {
            "Time": None,
            "Iterations": len(dis),
            "Timeout": False,
            "Equivalent": False,
            "Key Found": False,
            "dis": dis,
            "dos": dos,
            "iter_times": iter_times,
            "iter_keys": iter_keys,
        }

    # get key
    model = s_miter.get_model()
    attack_key = {n: model[v_miter.id(f"c1_{n}") - 1] > 0 for n in keys}

    # check key
    assumptions = {
        **{f"c0_{k}": v for k, v in key.items()},
        **{f"c1_{k}": v for k, v in attack_key.items()},
        "sat": True,
    }
    equivalent = not cg.sat.solve(m, assumptions)
    if verbose:
        print(f"[{_localtime()}] circuit: {cl.name}, equivalent: {equivalent}")

    exec_time = time.time() - start_time
    if verbose:
        print(f"[{_localtime()}] circuit: {cl.name}, elasped time: {exec_time}")

    return {
        "Time": exec_time,
        "Iterations": len(dis),
        "Timeout": False,
        "Equivalent": equivalent,
        "Key Found": True,
        "dis": dis,
        "dos": dos,
        "iter_times": iter_times,
        "iter_keys": iter_keys,
        "attack_key": attack_key,
    }


def decision_tree_attack(c_or_cl, nsamples, key=None, verbose=True):
    """
    Launch a decision tree attack on a locked circuit.

    Attempts to capture the functionality of the oracle circuit using a
    decision tree.

    Paramters
    ---------
    c_or_cl: circuitgraph.Circuit
            The circuit to reverse engineer. Can either be
            the oracle or the locked circuit. If the locked
            circuit, must pass in the correct key using
            the `key` parameter
    nsamples: int
            The number of samples to train the decision tree on
    key: dict of str:bool
            The correct key, used to construct the oracle if
            the locked circuit is given.
    verbose: bool
            If True, attack progress will be printed

    Returns
    -------
    dict of str:sklearn.tree.DecisionTreeClassifier
            The trained classifier for each output.

    """
    from sklearn.tree import DecisionTreeClassifier

    if key:
        cl = c_or_cl
        for k, v in key.items():
            cl.set_type(k, str(int(v)))
        c = cl
    else:
        c = c_or_cl

    ins = tuple(c.startpoints())
    outs = tuple(c.endpoints())

    # generate training samples
    x = []
    y = {o: [] for o in outs}
    if verbose:
        print(f"[{_localtime()}] Generating samples")
    for i in range(nsamples):
        x += [[random.choice((True, False)) for i in ins]]
        result = cg.sat.solve(c, {i: v for i, v in zip(ins, x[-1])})
        for o in outs:
            y[o] += [result[o]]

    if verbose:
        print(f"[{_localtime()}] Training decision trees")
    estimators = {o: DecisionTreeClassifier() for o in outs}
    for o in outs:
        estimators[o].fit(x, y[o])

    if verbose:
        print(f"[{_localtime()}] Testing decision trees")
    ncorrect = 0
    for i in range(nsamples):
        x = [[random.choice((True, False)) for i in ins]]
        result = cg.sat.solve(c, {i: v for i, v in zip(ins, x[-1])})
        if all(result[o] == estimators[o].predict(x) for o in outs):
            ncorrect += 1

    if verbose:
        print(f"[{_localtime()}] Test accuracy: {ncorrect / nsamples}")
    return estimators
