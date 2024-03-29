import tempfile
import unittest

import circuitgraph as cg

from logiclocking import check_for_difference, locked_unroll, locks, read_key, write_key


class TestUtils(unittest.TestCase):
    def test_check_for_difference(self):
        c0 = cg.from_lib("c17")
        o = c0.outputs().pop()

        c1 = c0.copy()
        c1.set_output(o, False)
        o_pre = c1.uid(f"{o}_pre")
        c1.relabel({o: o_pre})
        k = c1.uid("key_0")
        c1.add(k, "input")
        c1.add(o, "xor", fanin=[o_pre, k], output=True)
        self.assertFalse(check_for_difference(c0, c1, {k: False}))
        self.assertTrue(check_for_difference(c0, c1, {k: True}))

    def test_locked_unroll(self):
        c = cg.from_lib("s27")
        cl, key = locks.trll(c, 8, shuffle_key=False)
        num_copies = 4
        clu, cu, io_map = locked_unroll(cl, key, num_copies, initial_values="0")
        self.assertEqual(len(cu.inputs()), (len(c.inputs()) - 1) * num_copies)
        self.assertEqual(len(cu.outputs()), len(c.outputs()) * num_copies)
        self.assertSetEqual(cu.inputs(), clu.inputs() - set(key))
        self.assertSetEqual(cu.outputs(), clu.outputs())
        self.assertFalse(check_for_difference(cu, clu, key))

        wrong_key = {k: not v for k, v in key.items()}
        self.assertTrue(check_for_difference(cu, clu, wrong_key))

    def test_read_write_key(self):
        key = {"k0": True, "k1": False}
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="logiclocking_test_utils_test_read_write_key"
        ) as f:
            write_key(key, f.name)
            self.assertDictEqual(key, read_key(f.name))
