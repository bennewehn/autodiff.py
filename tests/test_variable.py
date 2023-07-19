import numpy as np
import unittest
from autodiff import Variable


class TestVariable(unittest.TestCase):

    def test_matMul(self):
        a = Variable([1, 2, 3])
        b = Variable([2, 3, 4])
        z = a@b
        z.backward()

        self.assertEqual(z.data, np.array([20]), "Should be 20")

        self.assertTrue(np.array_equal(a.grad, b.data),
                        "Grad of a should be b")
        self.assertTrue(np.array_equal(b.grad, a.data),
                        "Grad of b should be a")

    def test_matmulSelf(self):
        a = Variable([1, 2, 3])
        z = a@a
        z.backward()

        self.assertEqual(z.data, np.array([14]), "Should be 14")
        self.assertTrue(np.array_equal(a.grad, 2*a.data), "Grad should be 2a")

    def test_matmulSelf1(self):
        a = Variable([10, 4, 4])
        z = a@a
        z.backward()

        self.assertEqual(z.data, np.array([132]), "Should be 132")
        self.assertTrue(np.array_equal(a.grad, 2*a.data), "Should be 2a")

    def test_matmulValueError(self):
        def foo():
            a = Variable([1, 2, 3])
            z = a@a@a
            z.backward()

        self.assertRaises(ValueError, foo)

    def test_add(self):
        a = Variable([1, 2, 3])
        b = Variable([2, 3, 4])
        z = a+b
        z.backward()

        self.assertTrue(np.array_equal(z.data, a.data+b.data), "Should be a+b")
        self.assertTrue(np.array_equal(
            a.grad, np.ones_like(a.data)), "Grad should be ones")
        self.assertTrue(np.array_equal(
            b.grad, np.ones_like(b.data)), "Grad should be ones")

    def test_add_same(self):
        a = Variable([2])
        z = a+a
        z.backward()

        self.assertTrue(np.array_equal(z.data, 2*a.data), "Should be 2a")
        self.assertTrue(np.array_equal(a.grad, [2]), "Grad should be 2")

    def test_add_same1(self):
        a = Variable([2])
        z = a+a+a
        z.backward()

        self.assertTrue(np.array_equal(z.data, 3*a.data), "Should be 3a")
        self.assertTrue(np.array_equal(a.grad, [3]), "Grad should be 3")

    def test_add_matmul(self):
        a = Variable([1, 2, 3])
        b = Variable([2, 3, 4])
        c = Variable([10, 4, 4])
        z = (a+b)@c
        z.backward()

        self.assertEqual(z.data, np.array([78]), "Should be 78")
        self.assertTrue(np.array_equal(a.grad, c.data), "Should be c")
        self.assertTrue(np.array_equal(b.grad, c.data), "Should be c")
        self.assertTrue(np.array_equal(c.grad, a.data+b.data), "Should be a+b")

    def test_add_matmul2(self):
        a = Variable([1, 2, 3])
        b = Variable([2, 3, 4])
        c = Variable([10, 4, 4])
        d = Variable([1, 1, 1])
        z = (a+b+c)@d
        z.backward()

        self.assertEqual(z.data, np.array([33]), "Should be 33")
        self.assertTrue(np.array_equal(a.grad, d.data), "Should be d")
        self.assertTrue(np.array_equal(b.grad, d.data), "Should be d")
        self.assertTrue(np.array_equal(
            d.grad, a.data+b.data+c.data), "Should be a+b+c")

    def test_mul(self):
        a = Variable([2])
        b = Variable([1, 2, 3])
        z = a*b
        z.backward()

        self.assertTrue(np.array_equal(z.data, 2*b.data), "Should be 2b")
        self.assertTrue(np.array_equal(a.grad, [6]), "Should be 6")
        self.assertTrue(np.array_equal(b.grad, [2, 2, 2]), "Should be twos")

    def test_matmul_mul(self):
        a = Variable([2])
        c = Variable([10, 4, 4])
        z = (a*c)@c
        z.backward()

        self.assertTrue(np.array_equal(z.data, [264]), "Should be 264")
        self.assertTrue(np.array_equal(a.grad, [132]), "Should be 132")
        self.assertTrue(np.array_equal(c.grad, 4*c.data), "Should be 4c")

    def test_funcsConcat(self):
        a = Variable([2])
        b = Variable([1, 2, 3])
        c = Variable([10, 4, 4])
        z = a*a*a+b@c
        z.backward()

        self.assertTrue(np.array_equal(z.data, [38]), "Should be 38")
        self.assertTrue(np.array_equal(a.grad, [12]), "Should be 12")
        self.assertTrue(np.array_equal(b.grad, c.data), "Should be c")
        self.assertTrue(np.array_equal(c.grad, b.data), "Should be b")

    def test_neg(self):
        a = Variable([3])
        z = -a
        z.backward()

        self.assertTrue(np.array_equal(z.data, [-3]), "Should be -3")
        self.assertTrue(np.array_equal(a.grad, [-1]), "Should be -1")

    def test_neg1(self):
        a = Variable([3, 4])
        z = -a
        z.backward()

        self.assertTrue(np.array_equal(z.data, -a.data), "Should be -a")
        self.assertTrue(np.array_equal(
            a.grad, -1*np.ones_like(a.data)), "Should be neg ones")

    def test_sum(self):
        a = Variable([2])
        b = Variable([1, 2, 3])
        x = a*b
        z = x.sum()
        z.backward()

        self.assertTrue(np.array_equal(z.data, [12]), "Should be 12")
        self.assertTrue(np.array_equal(a.grad, [6]), "Should be 6")
        self.assertTrue(np.array_equal(b.grad, [2, 2, 2]), "Should be twos")
        self.assertTrue(np.array_equal(
            x.grad, np.ones_like(b.data)), "Should be ones")

    def test_exp(self):
        a = Variable([1])
        z = a.exp()
        z.backward()

        self.assertTrue(np.array_equal(z.data, np.exp(a.data)))
        self.assertTrue(np.array_equal(a.grad, np.exp(a.data)))

    def test_exp2(self):
        a = Variable([2])
        b = Variable([1, 2, 3])
        z = (a*b).exp()
        z.backward()

        self.assertTrue(np.array_equal(z.data, np.exp(a.data*b.data)))
        self.assertTrue(np.array_equal(
            a.grad, [(np.exp(a.data*b.data)*b.data).sum()]))
        self.assertTrue(np.array_equal(b.grad, np.exp(a.data*b.data)*a.data))

    def test_pow(self):
        a = Variable([2])
        b = Variable([1, 2, 3])
        z = a**b
        z.backward()

        self.assertTrue(np.array_equal(z.data, np.power(a.data, b.data)))
        self.assertTrue(np.array_equal(a.grad, [17]))
        self.assertTrue(np.array_equal(b.grad, z.data * np.log(a.data)))

    def test_pow2(self):
        a = Variable([2])
        b = Variable([1, 2, 3])
        z = b**a
        z.backward()

        self.assertTrue(np.array_equal(z.data, np.power(b.data, a.data)))
        self.assertTrue(np.array_equal(
            a.grad, [(np.power(b.data, a.data) * np.log(b.data)).sum()]))
        self.assertTrue(np.array_equal(b.grad, 2*b.data))


if __name__ == '__main__':
    unittest.main()
