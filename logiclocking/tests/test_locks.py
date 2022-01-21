import unittest

import circuitgraph as cg
from logiclocking import locks
from logiclocking.utils import check_for_difference


class TestLocks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c17 = cg.from_lib('c17_gates')
        cls.s27 = cg.from_lib('s27')
        cls.c499 = cg.from_lib('c499')
        cls.c432 = cg.from_lib('c432')
        cls.c880 = cg.from_lib('c880')
        cls.c6288 = cg.from_lib('c6288')

    def test_trll(self):
        c = self.c880
        cl, key = locks.trll(c, 32)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_xor_lock(self):
        c = self.c17
        cl, key = locks.xor_lock(c, 9)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_mux_lock(self):
        c = self.c17
        cl, key = locks.mux_lock(c, 9)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_random_lut_lock(self):
        c = self.c17
        cl, key = locks.random_lut_lock(c, 2, 2)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_lut_lock(self):
        c = self.c17
        with self.assertRaises(ValueError):
            locks.lut_lock(c, 100)
        c = cg.from_lib('c5315')
        cl, key = locks.lut_lock(c, 100)
        self.assertSetEqual(c.outputs(), cl.outputs())
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_sfll_hd(self):
        c = self.c499
        cl, key = locks.sfll_hd(c, 16, 4)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_tt_lock(self):
        c = self.c17
        cl, key = locks.tt_lock(c, 4)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    @unittest.skip("FIXME")
    def test_tt_lock_sen(self):
        c = self.c17
        cl, key = locks.tt_lock_sen(c, 4)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_sfll_flex(self):
        c = self.c17
        cl, key = locks.sfll_flex(c, 4, 2)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: v for k, v in key.items()}
        wkey['key_0'] = not wkey['key_0']
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_full_lock(self):
        c = self.c17
        cl, key = locks.full_lock(c, 8, 2)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_full_lock_mux(self):
        c = self.c17
        cl, key = locks.full_lock_mux(c, 8, 2)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))
    
    def test_inter_lock(self):
        c = cg.from_lib('c7552g')
        cl, key = locks.inter_lock(c, 8)
        cg.lint(cl)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_inter_lock_itc(self):
        c = cg.from_lib('b22_Cg')
        cl, key = locks.inter_lock(c, 16)
        cg.lint(cl)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_inter_lock_reduced_swb(self):
        c = cg.from_lib('c7552g')
        cl, key = locks.inter_lock_reduced_swb(c, 8)
        cg.lint(cl)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_lebl(self):
        c = self.c432
        cl, key = locks.lebl(c, 8, 8)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

    def test_uc_lock(self):
        uc = locks.gen_uc(5,5,3,3)
        cg.lint(uc)
        c = self.c17
        cl, key = locks.uc_lock(c)
        self.assertFalse(check_for_difference(c, cl, key))
        wkey = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(c, cl, wkey))

