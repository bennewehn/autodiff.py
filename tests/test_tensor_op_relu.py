import unittest
import numpy as np
from autodiff import Tensor


class TestTensorReluOperator(unittest.TestCase):

    def test_relu_forward(self):
        a = Tensor([-1, 0, 2], requires_grad=True)

        out = a.relu()

        self.assertTrue(np.array_equal(out.data, [0, 0, 2]))

    def test_relu_backward(self):
        a = Tensor([-1, 0, 2], requires_grad=True)

        out = a.relu()
        out.backward()

        self.assertTrue(np.array_equal(a.grad, [0, 0, 1]))
