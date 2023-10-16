import unittest
import numpy as np
from autodiff import Tensor


class TestTensorNegOperator(unittest.TestCase):

    def test_neg_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)

        out = -a

        self.assertTrue(np.array_equal(out.data, [-1, -2, -3]))

    def test_neg_backward(self):
        a = Tensor([1, 2, 3], requires_grad=True)

        out = -a
        out.backward()

        assert a.grad is not None

        self.assertTrue(np.array_equal(a.grad, [-1, -1, -1]))
