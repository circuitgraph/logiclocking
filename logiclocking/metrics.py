"""Cacluate logic-locking metrics."""
from random import random
from statistics import mean

import circuitgraph as cg


def corruptibility(cl, key):
    """Apprixmate corruptability of a locked circuit for a specific key."""
    # set up miter
    ins = set(cl.startpoints() - key.keys())
    m = cg.miter(cl, startpoints=ins)
    set_key = {f"c0_{k}": v for k, v in key.items()}
    independent_set = {f"c1_{k}" for k in key} | ins

    # run approx
    errors = cg.approx_model_count(m, {**set_key, "sat": True}, independent_set)

    return errors / 2 ** len(independent_set)


def key_corruption(cl, key, attack_key):
    """Approximate corruption between two keys."""
    # set up miter
    ins = set(cl.startpoints() - key.keys())
    m = cg.miter(cl, startpoints=ins)
    c0_key = {f"c0_{k}": v for k, v in key.items()}
    c1_key = {f"c1_{k}": v for k, v in attack_key.items()}

    # run approx
    errors = cg.approx_model_count(m, {**c0_key, **c1_key, "sat": True}, ins)

    return errors / 2 ** len(ins)


def min_corruption(cl, key, e=0.1, min_samples=10, tol=0.1):
    """Approximate the minimum corruption."""
    # find total errors
    cor = corruptibility(cl, key)

    # get initial key corruptions
    key_corruptions = []
    for i in range(min_samples):
        sampled_key = {k: random() < 0.5 for k in key}
        key_corruptions.append(key_corruption(cl, key, sampled_key))

    # sample until distribution matches
    while abs(mean(key_corruptions) - cor) > tol:
        sampled_key = {k: random() < 0.5 for k in key}
        key_corruptions.append(key_corruption(cl, key, sampled_key))

    return len([kc for kc in key_corruptions if kc >= e]) / len(key_corruptions)


def avg_avg_sensitivity(cl, key={}):
    """Get the average average sensitivity under a given key."""
    # set key
    cl_key = cl.copy()
    for k, v in key.items():
        cl_key.set_type(k, v)

    avg_sens = []
    for o in cl.outputs():
        # estimate sen
        avg_sens.append(cl.avg_sensitivity(o, e=3, d=0.8) / len(cl.startpoints(o)))

    return mean(avg_sens)
