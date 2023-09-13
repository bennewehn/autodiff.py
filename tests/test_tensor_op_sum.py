import unittest
import numpy as np
from autodiff import Tensor


class TestTensorSumOperator(unittest.TestCase):

    def test_sum_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)

        out = a.sum()

        self.assertTrue(np.array_equal(out.data, [6]))

    def test_sum_2d_forward(self):
        a = Tensor([[1, 2, 3], [2, 3, 4]], requires_grad=True)

        out = a.sum()

        self.assertTrue(np.array_equal(out.data, [15]))

    def test_sum_backward(self):
        a = Tensor([[1, 2, 3], [2, 3, 4]], requires_grad=True)

        out = a.sum()
        out.backward()

        self.assertTrue(np.array_equal(a.grad, [[1, 1, 1], [1, 1, 1]]))
