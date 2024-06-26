import unittest
import numpy as np
from autodiff import Tensor


class TestTensorDotOperator(unittest.TestCase):

    def test_mul_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2, 3, 4], requires_grad=True)

        out = a @ b

        self.assertEqual(out.data, 20)

    def test_mul_backward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2, 3, 4], requires_grad=True)

        out = a @ b
        out.backward()

        assert a.grad is not None
        assert b.grad is not None

        self.assertTrue(np.array_equal(a.grad, [2, 3, 4]))
        self.assertTrue(np.array_equal(b.grad, [1, 2, 3]))

    def test_mul_backward_2d(self):
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[3, 4, 5], [1, 2, 3]], requires_grad=True)

        out = a @ b
        out.backward()

        assert a.grad is not None
        assert b.grad is not None

        self.assertTrue(np.array_equal(a.grad, [[12, 6], [12, 6]]))
        self.assertTrue(np.array_equal(b.grad, [[4, 4, 4], [6, 6, 6]]))

    def test_mul_backward_ma5_vec(self):
        a = Tensor([[1, 2, 3], [3, 4, 1]], requires_grad=True)
        b = Tensor([3, 4, 5], requires_grad=True)

        out = a @ b
        out.backward()

        assert a.grad is not None
        assert b.grad is not None

        self.assertTrue(np.array_equal(a.grad, [[3, 4, 5], [3, 4, 5]]))
        self.assertTrue(np.array_equal(b.grad, [4, 6, 4]))
