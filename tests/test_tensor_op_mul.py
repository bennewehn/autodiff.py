import unittest
import numpy as np
from autodiff import Tensor


class TestTensorMulOperator(unittest.TestCase):

    def test_mul_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2, 3, 4], requires_grad=True)

        out = a * b

        self.assertTrue(np.array_equal(out.data, [2, 6, 12]))

    def test_mul_backward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2, 3, 4], requires_grad=True)

        out = a * b
        out.backward()

        self.assertTrue(np.array_equal(a.grad, [2, 3, 4]))
        self.assertTrue(np.array_equal(b.grad, [1, 2, 3]))

    def test_mul_backwards_broadcasting(self):
        a = Tensor([4], requires_grad=True)
        b = Tensor([2, 3, 4], requires_grad=True)

        out = a * b
        out.backward()

        self.assertTrue(np.array_equal(a.grad, [9]))
        self.assertTrue(np.array_equal(b.grad, [4, 4, 4]))

    def test_mul_other(self):
        a = Tensor([2, 3, 4], requires_grad=True)

        out = a * 5
        out.backward()

        self.assertTrue(np.array_equal(out.data, [10, 15, 20]))
        self.assertTrue(np.array_equal(a.grad, [5, 5, 5]))

    def test_mul_other_self(self):
        a = Tensor([2, 3, 4], requires_grad=True)

        out = 5 * a
        out.backward()

        self.assertTrue(np.array_equal(out.data, [10, 15, 20]))
        self.assertTrue(np.array_equal(a.grad, [5, 5, 5]))

    def test_mul_2d(self):
        a = Tensor([[2, 3], [1, 2]], requires_grad=True)
        b = Tensor([[1, 2], [3, 5]], requires_grad=True)

        out = a * b
        out.backward()

        self.assertTrue(np.array_equal(out.data, [[2, 6], [3, 10]]))
        self.assertTrue(np.array_equal(a.grad, [[1, 2], [3, 5]]))
        self.assertTrue(np.array_equal(b.grad, [[2, 3], [1, 2]]))

    def test_mul_2d_broadcasting(self):
        a = Tensor([[2, 3], [1, 2]], requires_grad=True)
        b = Tensor([4, 5], requires_grad=True)

        out = a * b
        out.backward()

        self.assertTrue(np.array_equal(out.data, [[8, 15], [4, 10]]))
        self.assertTrue(np.array_equal(a.grad, [[4, 5], [4, 5]]))
        self.assertTrue(np.array_equal(b.grad, [3, 5]))

    def test_mul_self(self):
        a = Tensor([4, 5], requires_grad=True)

        out = a * a * a
        out.backward()

        print(a.grad)

        self.assertTrue(np.array_equal(out.data, [64, 125]))
        self.assertTrue(np.array_equal(a.grad, [48, 75]))
