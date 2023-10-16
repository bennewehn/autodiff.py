import unittest
import numpy as np
from autodiff import Tensor


class TestTensorSumOperator(unittest.TestCase):

    def test_sum_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)

        out = a.sum()

        self.assertEqual(out.data, 6)

    def test_sum_2d_forward(self):
        a = Tensor([[1, 2, 3], [2, 3, 4]], requires_grad=True)

        out = a.sum()

        self.assertEqual(out.data, 15)

    def test_sum_backward(self):
        a = Tensor([[1, 2, 3], [2, 3, 4]], requires_grad=True)

        out = a.sum()
        out.backward()

        self.assertTrue(np.array_equal(a.grad, [[1, 1, 1], [1, 1, 1]]))

    def test_sum_dim(self):
        a = Tensor([[1, 2, 1], [4, 2, 2]], requires_grad=True)

        out = a.sum(dim=-1)
        out.backward()

        self.assertTrue(np.array_equal(a.grad, [[1, 1, 1], [1, 1, 1]]))

    def test_sum_dim_keepdim(self):
        a = Tensor([[1, 2, 1], [4, 2, 2]], requires_grad=True)

        out = a.sum(dim=-1, keepdim=True)
        out.backward()

        self.assertTrue(np.array_equal(a.grad, [[1, 1, 1], [1, 1, 1]]))
