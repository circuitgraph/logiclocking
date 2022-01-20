import unittest
import os

import circuitgraph as cg
from logiclocking import locks, attacks, utils


class TestLocks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c17 = cg.from_lib('c17_gates')
        cls.c499 = cg.from_lib('c499')
        cls.c432 = cg.from_lib('c432')
        cls.s27 = cg.from_lib('s27')

    def test_miter_attack(self):
        c = self.c17
        cl, key = locks.xor_lock(c, 8)
        result = attacks.miter_attack(cl, key,
                                      verbose=False, code_on_error=False)
        self.assertTrue(result['Equivalent'])

        cl, key = locks.sfll_hd(c, 4, 3)
        result = attacks.miter_attack(cl, key,
                                      verbose=False, code_on_error=False)
        self.assertTrue(result['Equivalent'])

    def test_cyclic_miter_attack(self):
        c = self.c17
        cl, key = locks.mux_lock(c, 8)
        while not cl.is_cyclic():
            cl, key = locks.mux_lock(c, 8)
        result = attacks.miter_attack(cl, key,
                                      verbose=False, code_on_error=False)
        self.assertTrue(result['Equivalent'])

        c = self.c17
        cl, key = locks.full_lock(c, 8, 2)
        while not cl.is_cyclic():
            cl, key = locks.full_lock(c, 8, 2)
        result = attacks.miter_attack(cl, key,
                                      verbose=False, code_on_error=False)
        self.assertTrue(result['Equivalent'])

    def test_decision_tree_attack(self):
        c = self.c499
        cl, key = locks.xor_lock(c, 8)
        result = attacks.decision_tree_attack(cl, key, 100)

    #def test_acyclic(self):
    #    c = self.c499

    #    cl, key = locks.mux_lock(c, 8)
    #    while not cl.is_cyclic():
    #        cl, key = locks.mux_lock(c, 8)
    #    #a = attacks.acyclic_sen(cl)
    #    a = attacks.acyclic(cl, key.keys())
    #    m = cg.miter(cl)
    #    m.extend(cg.relabel(a, {g:f'acyc_{g}' for g in a if g not in cl.startpoints()}))
    #    result = cg.sat(m,{'sat':True,'acyc_sat':True})
    #    self.assertFalse(result)

    #    cl, key = locks.full_lock(c, 8, 2)
    #    while not cl.is_cyclic():
    #        cl, key = locks.full_lock(c, 8, 2)
    #    #a = attacks.acyclic_sen(cl)
    #    a = attacks.acyclic(cl, key.keys())
    #    m = cg.miter(cl)
    #    m.extend(cg.relabel(a, {g:f'acyc_{g}' for g in a if g not in cl.startpoints()}))
    #    result = cg.sat(m,{'sat':True,'acyc_sat':True})
    #    self.assertFalse(result)

    #    cl, key = locks.lut_lock(c, 1, 8)
    #    self.assertFalse(cl.is_cyclic())

