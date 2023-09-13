import unittest
import numpy as np
from autodiff import Tensor


class TestTensorDivOperator(unittest.TestCase):

    def test_div_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2, 4, 6], requires_grad=True)

        out = a / b

        self.assertTrue(np.array_equal(out.data, [0.5, 0.5, 0.5]))

    def test_div_backward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2, 4, 6], requires_grad=True)

        out = a / b
        out.backward()

        print(a.grad)

        self.assertTrue(np.allclose(a.grad, [0.5, 0.25, 0.1666], 0.01))
        self.assertTrue(np.allclose(b.grad, [-0.25, -0.125, -0.08333], 0.01))

    def test_div_broadcasting(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2], requires_grad=True)

        out = a / b
        out.backward()

        self.assertTrue(np.array_equal(a.grad, [0.5, 0.5, 0.5]))
        self.assertTrue(np.array_equal(b.grad, [-1.5]))
