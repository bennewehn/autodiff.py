import unittest
import numpy as np
from autodiff import Tensor


class TestTensorPowOperator(unittest.TestCase):

    def test_pow_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)

        out = a.pow(b)

        self.assertTrue(np.array_equal(out.data, [1, 4, 27]))

    def test_pow_broadcast(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2], requires_grad=True)

        out = a.pow(b)
        out.backward()

        self.assertTrue(np.array_equal(a.grad, [2, 4, 6]))
        self.assertAlmostEqual(b.grad[0], 12.66, 3)

    def test_pow_self(self):
        a = Tensor([1, 2, 3], requires_grad=True)

        out = a.pow(a)
        out.backward()

        self.assertTrue(np.allclose(a.grad, [1, 6.77258, 56.66253]))
