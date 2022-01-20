from time import time
import code
import random
from itertools import product

import networkx as nx
from networkx.exception import NetworkXNoCycle
from circuitgraph.sat import sat, construct_solver, cnf, remap
from circuitgraph.circuit import Circuit
from circuitgraph.transform import miter
import circuitgraph as cg
from sklearn.tree import DecisionTreeClassifier


def decision_tree_attack(cl, key, nsamples):

    # setup vars
    keys = tuple(key.keys())
    ins = tuple(cl.startpoints()-key.keys())
    outs = tuple(cl.endpoints())

    # generate training samples
    x = []
    y = {o:[] for o in outs}
    for i in range(nsamples):
        x += [[random.choice((True,False)) for i in ins]]
        result = sat(cl,{**{i:v for i,v in zip(ins,x[-1])},**key})
        for o in outs:
            y[o] += [result[o]]

    estimators = {o:DecisionTreeClassifier() for o in outs}
    for o in outs:
        estimators[o].fit(x,y[o])

    # test accuracy
    ncorrect = 0
    for i in range(nsamples):
        x = [[random.choice((True,False)) for i in ins]]
        result = sat(cl,{**{i:v for i,v in zip(ins,x[-1])},**key})
        if all(result[o]==estimators[o].predict(x) for o in outs):
            ncorrect += 1

    print(ncorrect/nsamples)
    return estimators


def miter_attack(cl, key, timeout=None, key_cons=None, unroll_cyclic=True, verbose=True, code_on_error=True):
    """
    Launch a miter-based sat attack on a locked circuit

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
            If False, information on the attack will not be printed
    code_on_error: bool
            If True, drop into an interactive session on an error
    """
    start_time = time()

    if unroll_cyclic and cl.is_cyclic():
        cl = acyclic_unroll(cl)

    # setup vars
    keys = tuple(key.keys())
    ins = tuple(cl.startpoints()-key.keys())
    outs = tuple(cl.endpoints())

    # create simulation solver
    s_sim, v_sim = construct_solver(cl, key)

    # create miter solver
    m = miter(cl, startpoints=set(ins))
    s_miter, v_miter = construct_solver(m)

    # add key constraints
    if key_cons:
        if isinstance(key_cons, Circuit):
            key_cons = [key_cons]
        for key_con in key_cons:
            if verbose:
                print(f'circuit: {cl.name}, adding constraints: {key_con.name}')
            formula, v_cons = cnf(key_con)
            con_clauses = formula.clauses

            # add constraints circuits
            c0_offset = s_miter.nof_vars()
            c0 = remap(con_clauses, c0_offset)
            s_miter.append_formula(c0)
            c1_offset = s_miter.nof_vars()
            c1 = remap(con_clauses, c1_offset)
            s_miter.append_formula(c1)

            # encode keys connections
            clauses = [[v_cons.id('sat')+c0_offset], [v_cons.id('sat')+c1_offset]]
            clauses += [[-v_miter.id(f'c0_{n}'),
                         v_cons.id(n)+c0_offset] for n in keys]
            clauses += [[v_miter.id(f'c0_{n}'), -
                         v_cons.id(n)-c0_offset] for n in keys]
            clauses += [[-v_miter.id(f'c1_{n}'),
                         v_cons.id(n)+c1_offset] for n in keys]
            clauses += [[v_miter.id(f'c1_{n}'), -
                         v_cons.id(n)-c1_offset] for n in keys]

            s_miter.append_formula(clauses)

    # get circuit clauses
    formula, v_c = cnf(cl)
    clauses = formula.clauses

    # solve
    dis = []
    dos = []
    iter_times = []
    iter_keys = []
    while s_miter.solve(assumptions=[v_miter.id('sat')]):

        # get di
        model = s_miter.get_model()
        di = [model[v_miter.id(n)-1] > 0 for n in ins]
        if tuple(di) in dis:
            if verbose:
                print('error di')
            if code_on_error:
                code.interact(local=dict(globals(), **locals()))
            return {'Time': None, 'Iterations': len(dis),
                    'Timeout': False, 'Equivalent': False,
                    'Key Found': False, 'dis': dis, 'dos': dos,
                    'iter_times':iter_times, 'iter_keys':iter_keys}

        # get intermediate keys
        k0 = {n:model[v_miter.id(f'c0_{n}')-1] > 0 for n in keys}
        k1 = {n:model[v_miter.id(f'c1_{n}')-1] > 0 for n in keys}
        iter_keys.append((k0,k1))

        # get do
        s_sim.solve(assumptions=[(2*b-1)*v_sim.id(n)
                                 for b, n in zip(di, ins)])
        model = s_sim.get_model()
        if model is None:
            if code_on_error:
                print('error sim')
                code.interact(local=dict(globals(), **locals()))
            else:
                raise ValueError('Could not get simulation model')
        do = [model[v_sim.id(n)-1] > 0 for n in outs]
        dis.append(tuple(di))
        dos.append(tuple(do))
        iter_times.append(time() - start_time)

        # add constraints circuits
        c0_offset = s_miter.nof_vars()
        c0 = remap(clauses, c0_offset)
        s_miter.append_formula(c0)
        c1_offset = s_miter.nof_vars()
        c1 = remap(clauses, c1_offset)
        s_miter.append_formula(c1)

        # encode dis + dos
        dio_clauses = [[(2*b-1)*(v_c.id(n)+c0_offset)]
                       for b, n in zip(di+do, ins+outs)]
        dio_clauses += [[(2*b-1)*(v_c.id(n)+c1_offset)]
                        for b, n in zip(di+do, ins+outs)]
        s_miter.append_formula(dio_clauses)

        # encode keys connections
        key_clauses = [
            [-v_miter.id(f'c0_{n}'),
             v_c.id(n)+c0_offset] for n in keys]
        key_clauses += [[v_miter.id(f'c0_{n}'), -
                         v_c.id(n)-c0_offset] for n in keys]
        key_clauses += [[-v_miter.id(f'c1_{n}'),
                         v_c.id(n)+c1_offset] for n in keys]
        key_clauses += [[v_miter.id(f'c1_{n}'), -
                         v_c.id(n)-c1_offset] for n in keys]
        s_miter.append_formula(key_clauses)

        # check timeout
        if timeout and (time() - start_time) > timeout:
            print(f'circuit: {cl.name}, Timeout: True')
            return {'Time': None, 'Iterations': len(dis), 'Timeout': True,
                    'Equivalent': False, 'Key Found': False,
                    'dis': dis, 'dos': dos, 'iter_times':iter_times, 'iter_keys':iter_keys}

        if verbose:
            print(f'circuit: {cl.name}, iter: {len(dis)}, '
                  f'time: {time()-start_time}, '
                  f'clauses: {s_miter.nof_clauses()}, '
                  f'vars: {s_miter.nof_vars()}')

    # check if a satisfying key remains
    key_found = s_miter.solve()
    if verbose:
        print(f'circuit: {cl.name}, key found: {key_found}')
    if not key_found:
        return {'Time': None, 'Iterations': len(dis), 'Timeout': False,
                'Equivalent': False, 'Key Found': False,
                'dis': dis, 'dos': dos, 'iter_times':iter_times, 'iter_keys':iter_keys}

    # get key
    model = s_miter.get_model()
    attack_key = {n: model[v_miter.id(f'c1_{n}')-1] > 0 for n in keys}

    # check key
    assumptions = {**{f'c0_{k}': v for k, v in key.items()},
                   **{f'c1_{k}': v for k, v in attack_key.items()},
                   'sat': True}
    equivalent = not sat(m, assumptions)
    if verbose:
        print(f'circuit: {cl.name}, equivalent: {equivalent}')

    exec_time = time()-start_time
    if verbose:
        print(f'circuit: {cl.name}, elasped time: {exec_time}')

    return {'Time': exec_time, 'Iterations': len(dis), 'Timeout': False,
            'Equivalent': equivalent, 'Key Found': True, 'dis': dis,
            'dos': dos, 'iter_times':iter_times, 'iter_keys':iter_keys,
            'attack_key': attack_key}

def acyclic_unroll(c):

    if c.blackboxes:
        raise ValueError('remove blackboxes')

    # find feedback nodes
    feedback = set([e[0] for e in approx_min_fas(c.graph)])

    # get startpoints
    sp = c.startpoints()

    # create acyclic circuit
    acyc = Circuit(name=f'acyc_{c.name}')
    for n in sp:
        acyc.add(n,'input')

    # create copy with broken feedback
    c_cut = cg.copy(c)
    for f in feedback:
        fanout = c.fanout(f)
        c_cut.disconnect(f,fanout)
        c_cut.add(f'aux_in_{f}','buf',fanout=fanout)
    c_cut.set_type(c.outputs(),"buf")

    # cut feedback
    for i in range(len(feedback)+1):
        # instantiate copy
        acyc.add_subcircuit(c_cut,f'c{i}',{n:n for n in sp})

        if i > 0:
            # connect to last
            for f in feedback:
                acyc.connect(f'c{i-1}_{f}',f'c{i}_aux_in_{f}')
        else:
            # make feedback inputs
            for f in feedback:
                acyc.set_type(f'c{i}_aux_in_{f}',"input")

    # connect outputs
    for o in c.outputs():
        acyc.add(o,'output',fanin=f'c{i}_{o}')

    cg.lint(acyc)
    if acyc.is_cyclic():
        raise ValueError('circuit still cyclic')
    return acyc

def approx_min_fas(DG):
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
            n = max(DGC.nodes,
                    key=lambda x: DGC.out_degree(x)-DGC.in_degree(x))
            s1.append(n)
            DGC.remove_node(n)

    ordering = s1+list(reversed(s2))
    feedback_edges = [e for e in DG.edges if ordering.index(
        e[0]) > ordering.index(e[1])]
    feedback_edges = [(u, v) for u, v in feedback_edges
                      if u in nx.descendants(DG, v)]

    DGC = DG.copy()
    DGC.remove_edges_from(feedback_edges)
    try:
        if nx.find_cycle(DGC):
            print('cyclic')
            code.interact(local=dict(globals(), **locals()))
    except NetworkXNoCycle:
        pass

    return feedback_edges


def acyclic(cl, keys):
    # find feedback nodes
    feedback = set([e[0] for e in approx_min_fas(cl.graph)])

    acyc = cg.Circuit(name='acyc')
    acyc.add('sat', 'output')
    acyc.add('conj', 'and', fanout='sat')
    for n in keys:
        acyc.add(n, 'input')

    for i, f in enumerate(feedback):
        feedback_nodes = (cl.transitive_fanout(
            f) & cl.transitive_fanin(f)) | set([f])

        # add path broken nodes
        for d in feedback_nodes:
            acyc.add(f'f{f}_d{d}', 'or')

        # connect
        for d in feedback_nodes:
            # break here or upstream
            if d == f:
                acyc.connect(f'f{f}_d{d}', 'conj')
            if f in set(cl.fanin(d)):
                # can't break upstream
                acyc.add(f'f{f}_u{d}', '0', fanout=f'f{f}_d{d}')
            else:
                # all upstream paths are broken
                acyc.add(f'f{f}_u{d}', 'and', fanout=f'f{f}_d{d}')
                if not set(cl.fanin(d)) & feedback_nodes:
                    code.interact(local=dict(globals(), **locals()))
                for p in set(cl.fanin(d)) & feedback_nodes:
                    acyc.connect(f'f{f}_d{p}', f'f{f}_u{d}')

            # add controlling inputs
            if (cl.type(d) in ('xnor', 'xor', 'buf', 'not')
                    or not cl.fanin(d)-feedback_nodes):
                # can't break here
                acyc.add(f'f{f}_h{d}','0',fanout=f'f{f}_d{d}')

            elif cl.type(d) in ('nor', 'or'):
                acyc.add(f'f{f}_h{d}','or',fanout=f'f{f}_d{d}')
                for p in cl.fanin(d)-feedback_nodes:
                    control(cl, acyc, p, 1, f, keys)
                    acyc.connect(f'f{f}_c{p}_v1', f'f{f}_h{d}')

            elif cl.type(d) in ('nand', 'and'):
                acyc.add(f'f{f}_h{d}','or',fanout=f'f{f}_d{d}')
                for p in cl.fanin(d)-feedback_nodes:
                    control(cl, acyc, p, 0, f, keys)
                    acyc.connect(f'f{f}_c{p}_v0', f'f{f}_h{d}')

            else:
                print('huh')
                code.interact(local=dict(globals(), **locals()))

    # check all types are set
    try:
        cg.lint(acyc)
    except:
        import code
        code.interact(local=dict(globals(), **locals()))
    if not cg.sat(acyc,{'sat':True}):
        print('no satisfying key')
        import code
        code.interact(local=dict(globals(), **locals()))

    return acyc


def control(cl, acyc, n, v, f, keys):
    if f'f{f}_c{n}_v{v}' in acyc:
        pass
    elif n == f:
        acyc.add(f'f{f}_c{n}_v{v}', '0')
    elif cl.type(n) == 'input':
        if n in keys:
            acyc.add(f'f{f}_c{n}_v{v}', 'buf' if v else 'not')
            acyc.connect(n, f'f{f}_c{n}_v{v}')
        else:
            acyc.add(f'f{f}_c{n}_v{v}', '0')

    elif cl.type(n) == 'buf':
        p = list(cl.fanin(n))[0]
        acyc.add(f'f{f}_c{n}_v{v}', 'buf')
        control(cl, acyc, p, v, f, keys)
        acyc.connect(f'f{f}_c{p}_v{v}', f'f{f}_c{n}_v{v}')

    elif cl.type(n) == 'not':
        p = list(cl.fanin(n))[0]
        acyc.add(f'f{f}_c{n}_v{v}', 'buf')
        control(cl, acyc, p, 1-v, f, keys)
        acyc.connect(f'f{f}_c{p}_v{1-v}', f'f{f}_c{n}_v{v}')

    elif cl.type(n) == 'and':
        acyc.add(f'f{f}_c{n}_v{v}', 'and' if v else 'or')
        for p in cl.fanin(n):
            control(cl, acyc, p, 1 if v else 0, f, keys)
            acyc.connect(f'f{f}_c{p}_v{1 if v else 0}',f'f{f}_c{n}_v{v}')

    elif cl.type(n) == 'nand':
        acyc.add(f'f{f}_c{n}_v{v}', 'or' if v else 'and')
        for p in cl.fanin(n):
            control(cl, acyc, p, 0 if v else 1, f, keys)
            acyc.connect(f'f{f}_c{p}_v{0 if v else 1}',f'f{f}_c{n}_v{v}')

    elif cl.type(n) == 'or':
        acyc.add(f'f{f}_c{n}_v{v}', 'or' if v else 'and')
        for p in cl.fanin(n):
            control(cl, acyc, p, 1 if v else 0, f, keys)
            acyc.connect(f'f{f}_c{p}_v{1 if v else 0}',f'f{f}_c{n}_v{v}')

    elif cl.type(n) == 'nor':
        acyc.add(f'f{f}_c{n}_v{v}', 'and' if v else 'or')
        for p in cl.fanin(n):
            control(cl, acyc, p, 0 if v else 1, f, keys)
            acyc.connect(f'f{f}_c{p}_v{0 if v else 1}',f'f{f}_c{n}_v{v}')

    elif cl.type(n) == 'xor':
        acyc.add(f'f{f}_c{n}_v{v}', 'or')
        ps = list(cl.fanin(n))
        for pvs in product([0, 1], repeat=len(ps)):
            if sum(pvs) % 2 == v:
                acyc.add(f'f{f}_c{n}_v{v}_sub_{pvs}',
                         'and', fanout=f'f{f}_c{n}_v{v}')
                for p, pv in zip(ps, pvs):
                    control(cl, acyc, p, pv, f, keys)
                    acyc.connect(f'f{f}_c{p}_v{pv}',
                                 f'f{f}_c{n}_v{v}_sub_{pvs}')
    elif cl.type(n) == 'xnor':
        acyc.add(f'f{f}_c{n}_v{v}', 'or')
        ps = list(cl.fanin(n))
        for pvs in product([0, 1], repeat=len(ps)):
            if sum(pvs) % 2 != v:
                acyc.add(f'f{f}_c{n}_v{v}_sub_{pvs}',
                         'and', fanout=f'f{f}_c{n}_v{v}')
                for p, pv in zip(ps, pvs):
                    control(cl, acyc, p, pv, f, keys)
                    acyc.connect(f'f{f}_c{p}_v{pv}',
                                 f'f{f}_c{n}_v{v}_sub_{pvs}')
    elif cl.type(n) == '0':
        acyc.add(f'f{f}_c{n}_v{v}', '0' if v else '1')
    elif cl.type(n) == '1':
        acyc.add(f'f{f}_c{n}_v{v}', '1' if v else '0')
    else:
        print('gate error')
        code.interact(local=dict(globals(), **locals()))
