import unittest
import numpy as np
from autodiff import Tensor


class TestTensorMeanOperator(unittest.TestCase):

    def test_mean_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)

        out = a.mean()

        self.assertEqual(out.data, 2)

    def test_mean_backward(self):
        a = Tensor([1, 2, 3], requires_grad=True)

        out = a.mean()
        out.backward()

        assert a.grad is not None

        self.assertTrue(np.allclose(a.grad, [1/3, 1/3, 1/3], 0.001))

    def test_mean_backward_2d(self):
        a = Tensor([[1, 3], [1, 3]], requires_grad=True)

        out = a.mean()
        out.backward()

        assert a.grad is not None

        self.assertEqual(out.data, 2)
        self.assertTrue(np.array_equal(a.grad, [[0.25, 0.25], [0.25, 0.25]]))
