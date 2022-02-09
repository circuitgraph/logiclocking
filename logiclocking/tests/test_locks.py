import unittest
import random

import circuitgraph as cg
from logiclocking import locks
from logiclocking.utils import check_for_difference


class TestLocks(unittest.TestCase):
    def lock_test(
        self,
        circuit_name,
        lock_fn,
        lock_args=(),
        lock_kwargs={},
        wrong_key_inputs="half",
    ):
        c = cg.from_lib(circuit_name)
        cl, key = lock_fn(c, *lock_args, **lock_kwargs)
        cg.lint(cl)
        self.assertSetEqual(c.outputs(), cl.outputs())
        self.assertSetEqual(c.inputs(), cl.inputs() - set(key))
        self.assertSetEqual(cl.inputs() - c.inputs(), set(key))
        self.assertFalse(check_for_difference(c, cl, key))

        wrong_key = key.copy()
        if wrong_key_inputs == "one":
            inverted_key_input = random.choice(list(key))
            wrong_key[inverted_key_input] = not wrong_key[inverted_key_input]
        elif wrong_key_inputs == "half":
            for inverted_key_input in random.sample(list(key), len(key) // 2):
                wrong_key[inverted_key_input] = not wrong_key[inverted_key_input]
        elif wrong_key_inputs == "all":
            wrong_key = {k: not v for k, v in key.items()}
        else:
            raise ValueError(f"Unkown 'wrong_key_inputs' value: '{wrong_key_inputs}'")
        self.assertTrue(check_for_difference(c, cl, wrong_key))
        return c, cl, key, wrong_key

    def test_trll(self):
        self.lock_test("c880", locks.trll, (32,))

    def test_xor_lock(self):
        self.lock_test("c880", locks.xor_lock, (32,))

    def test_mux_lock(self):
        self.lock_test("c880", locks.mux_lock, (32,))

    def test_mux_lock_avoid_loops(self):
        _, cl, _, _ = self.lock_test(
            "c880", locks.mux_lock, (32,), {"avoid_loops": True}
        )
        self.assertFalse(cl.is_cyclic())

    def test_random_lut_lock(self):
        self.lock_test("c880", locks.random_lut_lock, (8, 4))

    def test_lut_lock(self):
        self.lock_test("c5315", locks.lut_lock, (100,))

    def test_lut_lock_not_enough_gates(self):
        c = cg.from_lib("c17")
        with self.assertRaises(ValueError):
            locks.lut_lock(c, 100)

    def test_tt_lock(self):
        self.lock_test("c880", locks.tt_lock, (16,))

    def test_tt_lock_not_enough_inputs(self):
        c = cg.from_lib("c17")
        with self.assertRaises(ValueError):
            locks.tt_lock(c, len(c.inputs()) + 1)

    def test_tt_lock_sen(self):
        self.lock_test("c880", locks.tt_lock_sen, (8,))

    def test_tt_lock_sen_not_enough_inputs(self):
        c = cg.from_lib("c17")
        with self.assertRaises(ValueError):
            locks.tt_lock_sen(c, len(c.inputs()) + 1)

    def test_sfll_hd(self):
        self.lock_test("c880", locks.sfll_hd, (16, 4))

    def test_sfll_hd_not_enough_inputs(self):
        c = cg.from_lib("c17")
        with self.assertRaises(ValueError):
            locks.sfll_hd(c, len(c.inputs()) + 1, 4)

    def test_sfll_flex(self):
        self.lock_test("c880", locks.sfll_flex, (16, 4))

    def test_sfll_flex_not_enough_inputs(self):
        c = cg.from_lib("c17")
        with self.assertRaises(ValueError):
            locks.sfll_flex(c, len(c.inputs()) + 1, 4)

    def test_full_lock(self):
        self.lock_test("c499", locks.full_lock, (16, 2))

    def test_full_lock_avoid_loops(self):
        _, cl, _, _ = self.lock_test(
            "c499", locks.full_lock, (8, 2), {"avoid_loops": True}
        )
        self.assertFalse(cl.is_cyclic())

    def test_full_lock_mux(self):
        self.lock_test("c499", locks.full_lock_mux, (16, 2))

    def test_inter_lock(self):
        bw = 8
        _, _, k, _ = self.lock_test("c7552g", locks.inter_lock, (bw,))
        self.assertEqual(len(k), 4 * bw // 2 * 3)

    def test_inter_lock_reduced_swb(self):
        bw = 8
        _, _, k, _ = self.lock_test(
            "c7552g", locks.inter_lock, (bw,), {"reduced_swb": True}
        )
        self.assertEqual(len(k), 4 * bw // 2)

    def test_lebl(self):
        self.lock_test("c432", locks.lebl, (8, 8))
