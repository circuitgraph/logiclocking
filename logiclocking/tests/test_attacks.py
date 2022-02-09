import unittest

import circuitgraph as cg
from logiclocking import locks, attacks


class TestLocks(unittest.TestCase):
    def test_miter_attack_xor_lock(self):
        c = cg.from_lib("c17_gates")
        cl, key = locks.xor_lock(c, 8)
        result = attacks.miter_attack(cl, key, verbose=False, code_on_error=False)
        self.assertTrue(result["Equivalent"])

    def test_miter_attack_sfll_hd(self):
        c = cg.from_lib("c17_gates")
        cl, key = locks.sfll_hd(c, 4, 3)
        result = attacks.miter_attack(cl, key, verbose=False, code_on_error=False)
        self.assertTrue(result["Equivalent"])

    def test_cyclic_miter_attack_mux_lock(self):
        c = cg.from_lib("c17_gates")
        cl, key = locks.mux_lock(c, 8)
        while not cl.is_cyclic():
            cl, key = locks.mux_lock(c, 8)
        with self.assertRaises(ValueError):
            result = attacks.miter_attack(
                cl, key, verbose=False, code_on_error=False, unroll_cyclic=False
            )

        result = attacks.miter_attack(cl, key, verbose=False, code_on_error=False)
        self.assertTrue(result["Equivalent"])

    def test_cyclic_miter_attack_full_lock(self):
        c = cg.from_lib("c17_gates")
        cl, key = locks.full_lock(c, 8, 2)
        while not cl.is_cyclic():
            cl, key = locks.full_lock(c, 8, 2)
        result = attacks.miter_attack(cl, key, verbose=False, code_on_error=False)
        self.assertTrue(result["Equivalent"])

    def test_decision_tree_attack(self):
        c = cg.from_lib("c499")
        cl, key = locks.xor_lock(c, 8)
        attacks.decision_tree_attack(cl, 100, key, verbose=False)
        attacks.decision_tree_attack(c, 100, verbose=False)
