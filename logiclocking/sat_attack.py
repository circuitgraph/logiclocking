from time import time
from random import random

import circuitgraph as cg
from circuitgraph.sat import sat, construct_solver, cnf, remap


def miter_attack(cl, c, timeout=None):
    # assume c and cl just have 1 output, and are acyclic
    start_time = time()

    # setup vars
    keys = tuple(key.keys())
    ins = tuple(cl.startpoints() - key.keys())
    out = cl.endpoints().pop()

    # create simulation solver
    s_sim, v_sim = construct_solver(c)

    # create miter solver
    m = cg.miter(cl, startpoints=set(ins))
    s_miter, v_miter = construct_solver(m)

    # get circuit clauses
    # formula, v_c = cnf(cl)
    # clauses = formula.clauses

    # pick initial attack key
    akey = [v_miter.id(f"c0_{k}") * (-1 if random() < 0.5 else 1) for k in keys]

    dis = []
    dos = []
    # solve for disagreeing key and input
    while s_miter.solve(assumptions=akey + [v_miter.id("sat")]):

        # get input, attack key output
        miter_model = s_miter.get_model()
        di = [miter_model[v_miter.id(n) - 1] > 0 for n in ins]
        ao = miter_model[v_miter.id(f"c0_{out}") - 1] > 0
        if tuple(di) in dis:
            print("error di")
            import code

            code.interact(local=dict(globals(), **locals()))
            return {
                "Time": None,
                "Iterations": len(dis),
                "Timeout": False,
                "Equivalent": False,
                "Key Found": False,
                "dis": dis,
                "dos": dos,
            }
        print(
            f"circuit: {cl.name}, iter: {len(dis)}, "
            f"clauses: {s_miter.nof_clauses()}, "
            f"vars: {s_miter.nof_vars()}"
        )

        # get correct output from oracle
        s_sim.solve(assumptions=[(2 * b - 1) * v_sim.id(n) for b, n in zip(di, ins)])
        sim_model = s_sim.get_model()
        if sim_model is None:
            print("error sim")
            import code

            code.interact(local=dict(globals(), **locals()))
        do = sim_model[v_sim.id(out) - 1] > 0
        dis.append(tuple(di))
        dos.append(do)

        # update attack key if incorrect
        if ao != do:
            akey = [
                v_miter.id(f"c0_{n}")
                * (1 if miter_model[v_miter.id(f"c1_{n}") - 1] > 0 else -1)
                for n in keys
            ]

        # synthesize constraint
        con = cg.copy(cl)
        for b, n in zip(di, ins):
            con.set_type(n, "1" if n else "0")
        con.set_type(out, "buf")
        con_val = con.add("con_val", "1" if do else "0", uid=True)
        xnor = con.add(f"xnor_{out}", "xnor", fanin=[out, con_val], uid=True)
        con.add(f"sat", "output", fanin=xnor)
        con = cg.syn(con, engine="genus")

        # add constraints circuits
        c1_offset = s_miter.nof_vars()
        formula, v_c = cnf(con)
        clauses = formula.clauses
        c1 = remap(clauses, c1_offset)
        s_miter.append_formula(c1)

        # encode do
        s_miter.add_clause([v_c.id("sat") + c1_offset])
        # dio_clauses = [[(2*b-1)*(v_c.id(n)+c1_offset)]
        #                for b, n in zip(di+[do], ins+(out,))]
        # s_miter.append_formula(dio_clauses)

        # encode keys connections
        key_clauses = [[-v_miter.id(f"c1_{n}"), v_c.id(n) + c1_offset] for n in keys]
        key_clauses += [[v_miter.id(f"c1_{n}"), -v_c.id(n) - c1_offset] for n in keys]
        s_miter.append_formula(key_clauses)

    if timeout and (time() - start_time) > timeout:
        print(f"circuit: {cl.name}, Timeout: True")
        return {
            "Time": None,
            "Iterations": len(dis),
            "Timeout": True,
            "Equivalent": False,
            "Key Found": False,
            "dis": dis,
            "dos": dos,
        }

    # check key
    attack_key = {n: v > 0 for n, v in zip(keys, akey)}
    assumptions = {
        **{f"c0_{k}": v for k, v in key.items()},
        **{f"c1_{k}": v for k, v in attack_key.items()},
        "sat": True,
    }
    equivalent = not sat(m, assumptions)
    print(f"circuit: {cl.name}, equivalent: {equivalent}")

    exec_time = time() - start_time
    print(f"circuit: {cl.name}, elasped time: {exec_time}")

    import code

    code.interact(local=dict(globals(), **locals()))
    return {
        "Time": exec_time,
        "Iterations": len(dis),
        "Timeout": False,
        "Equivalent": equivalent,
        "Key Found": True,
        "dis": dis,
        "dos": dos,
    }


if __name__ == "__main__":
    import circuitgraph as cg
    import pickle
    from logiclocking.locks import full_lock
    from logiclocking.attacks import acyclic_unroll

    c = cg.from_lib("c432")
    try:
        with open("c432_acyc.v", "rb") as f:
            cl, key = pickle.load(f)
    except:
        cl, key = full_lock(c, 32, 2)

        # pick an output w/ keys
        out = cl.endpoints(tuple(key.keys())[0]).pop()
        for o in cl.outputs():
            if o != out:
                cl.set_type(o, "buf")
                c.set_type(o, "buf")

        # unroll
        if cl.is_cyclic():
            orig_inputs = cl.inputs()
            cl = acyclic_unroll(cl)
            cl = cg.syn(cl, engine="genus")
            for n in cl.inputs() - orig_inputs:
                c.add(n, "input")

        with open("c432_acyc.v", "wb") as f:
            pickle.dump((cl, key), f)

    # attack
    miter_attack(cl, c)
