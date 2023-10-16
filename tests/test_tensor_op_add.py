import unittest
import numpy as np
from autodiff import Tensor


class TestTensorAddOperator(unittest.TestCase):

    def test_add_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2, 3, 4], requires_grad=True)

        out = a + b

        self.assertTrue(np.array_equal(out.data, [3, 5, 7]))

    def test_add_backward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2, 3, 4], requires_grad=True)

        out = a + b
        out.backward()

        assert out.grad is not None
        assert a.grad is not None
        assert b.grad is not None

        self.assertTrue(np.array_equal(out.data, [3, 5, 7]))
        self.assertTrue(np.array_equal(out.grad, [1, 1, 1]))

        self.assertTrue(np.array_equal(a.grad, [1, 1, 1]))
        self.assertTrue(np.array_equal(b.grad, [1, 1, 1]))

    def test_requires_grad_excep(self):
        a = Tensor([1, 2, 3])
        b = Tensor([2, 3, 4])

        out = a + b

        self.assertRaises(RuntimeError, out.backward)

    def test_broadcasting_scalar(self):
        a = Tensor([2], requires_grad=True)
        b = Tensor([2, 3, 4], requires_grad=True)

        out = a + b
        out.backward()

        assert a.grad is not None
        assert b.grad is not None

        self.assertTrue(np.array_equal(out.data, [4, 5, 6]))
        self.assertTrue(np.array_equal(a.grad, [3]))
        self.assertTrue(np.array_equal(b.grad, [1, 1, 1]))

    def test_add_same(self):
        a = Tensor([2], requires_grad=True)

        out = a+a+a
        out.backward()

        assert a.grad is not None

        self.assertTrue(np.array_equal(out.data, 3*a.data))
        self.assertTrue(np.array_equal(a.grad, [3]))

    def test_add_other(self):
        a = Tensor([2, 3, 4], requires_grad=True)

        out = a+2
        out.backward()

        assert a.grad is not None

        self.assertTrue(np.array_equal(out.data, [4, 5, 6]))
        self.assertTrue(np.array_equal(a.grad, [1, 1, 1]))

    def test_add_in_place(self):
        a = Tensor([2, 3, 4], requires_grad=True)

        a += 2

        self.assertTrue(np.array_equal(a.data, [4, 5, 6]))

    def test_add_self(self):
        a = Tensor([2, 3, 4], requires_grad=True)

        out = a + a + a
        out.backward()

        self.assertTrue(np.array_equal(out.data, [6, 9, 12]))

        assert a.grad is not None

        self.assertTrue(np.array_equal(a.grad, [3, 3, 3]))
