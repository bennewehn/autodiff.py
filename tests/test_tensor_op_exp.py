import unittest
import numpy as np
from autodiff import Tensor


class TestTensorExpOperator(unittest.TestCase):

    def test_exp_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)

        out = a.exp()

        self.assertTrue(np.allclose(out.data, np.exp(a.data)))

    def test_exp_backward(self):
        a = Tensor([1, 2, 3], requires_grad=True)

        out = a.exp()
        out.backward()

        self.assertTrue(np.allclose(out.grad, out.data, 10))
