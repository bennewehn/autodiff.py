import unittest
import numpy as np
from autodiff import Tensor


class TestTensorLogOperator(unittest.TestCase):

    def test_log_forward(self):
        a = Tensor([1, 2, 3], requires_grad=True)

        out = a.log()

        self.assertTrue(np.allclose(out.data, np.log(a.data)))

    def test_log_backward(self):
        # ignore divide by zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            a = Tensor([0, 1, 2, 3], requires_grad=True)

            out = a.log()
            out.backward()

            assert a.grad is not None

            self.assertTrue(np.allclose(a.grad, 1/a.data))
