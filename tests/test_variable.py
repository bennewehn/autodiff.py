import numpy as np
import unittest
from autodiff import Variable


class TestVariable(unittest.TestCase):

    def test_matmul(self):
        a = Variable([1, 2, 3])
        b = Variable([2, 3, 4])
        z = a@b
        z.backward()

        self.assertEqual(z.data, np.array([20]), "Should be 20")

        self.assertTrue(np.array_equal(a.grad, b.data),
                        "Grad of a should be b")
        self.assertTrue(np.array_equal(b.grad, a.data),
                        "Grad of b should be a")

    def test_matmul2(self):
        a = Variable([10, 4, 4])
        b = Variable([2])
        z = a@a@b
        z.backward()

        self.assertTrue(np.array_equal(z.data, [264]), "Should be 264")
        self.assertTrue(np.array_equal(a.grad, 4*a.data), "Should be 4a")
        self.assertTrue(np.array_equal(b.grad, [132]), "Should be 132")

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

    def test_mul2(self):
        a = Variable([4, 5, 4])
        b = Variable([2, 2, 3])
        z = (a*b)*a
        z.backward()

        self.assertTrue(np.array_equal(
            z.data, a.data ** 2 * b.data), "Should be a**2*b")
        self.assertTrue(np.array_equal(
            a.grad, [16, 20, 24]))
        self.assertTrue(np.array_equal(b.grad, [16, 25, 16]))

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

    def test_sum2(self):
        a = Variable([2])
        z = a.sum()*3
        z.backward()

        self.assertTrue(np.array_equal(z.data, [6]), "Should be 6")
        self.assertTrue(np.array_equal(a.grad, [3]))

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

    def test_exp_self(self):
        a = Variable([2])
        z = (a.exp()).exp()
        z.backward()

        self.assertTrue(np.isclose(z.data, np.exp(np.exp(a.data))))
        self.assertTrue(np.isclose(a.grad, z.data*np.exp(a.data)))

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

    def test_pow3(self):
        a = Variable([2])
        b = Variable([2, 2, 3])
        z = (a**b)**a
        z.backward()

        self.assertTrue(np.array_equal(z.data, [16, 16, 64]))
        self.assertTrue(np.array_equal(
            a.grad, [(b.data*a.data**(a.data*b.data)*(np.log(a.data)+1)).sum()]))
        self.assertTrue(np.isclose(
            b.grad, np.array([a.data**(b.data*a.data+1)*np.log(a.data)])).all())

    def test_pow4(self):
        a = Variable([2])
        b = Variable([2, 2, 3])
        z = ((a**b)**a)**a
        z.backward()

        self.assertTrue(np.array_equal(z.data, [256, 256, 4096]))
        self.assertTrue(np.isclose(a.grad, [63532.7031]))
        self.assertTrue(np.isclose(
            b.grad, [709.7827, 709.7827, 11356.5234]).all())

    def test_div(self):
        a = Variable([10, 4, 4])
        b = Variable([1, 2, 3])
        z = a/b
        z.backward()

        self.assertTrue(np.array_equal(z.data, np.divide(a.data, b.data)))
        self.assertTrue(np.array_equal(a.grad, 1/b.data))
        self.assertTrue(np.array_equal(b.grad, -a.data * np.power(b.data, -2)))

    def test_div_scalar(self):
        a = Variable([10, 4, 4])
        b = Variable([2])
        z = a/b
        z.backward()

        self.assertTrue(np.array_equal(z.data, np.divide(a.data, b.data)))
        self.assertTrue(np.array_equal(
            a.grad, (1/b.data) * np.ones_like(a.data)))
        self.assertTrue(np.array_equal(
            b.grad, [(-a.data * np.power(b.data, -2)).sum()]))

    def test_div_scalar2(self):
        a = Variable([2])
        b = Variable([10, 4, 4])
        z = a/b
        z.backward()

        self.assertTrue(np.array_equal(z.data, np.divide(a.data, b.data)))
        self.assertTrue(np.array_equal(a.grad, [(1/b.data).sum()]))
        self.assertTrue(np.array_equal(
            b.grad, np.array(-a.data * np.power(b.data, -2))))

    def test_div_self(self):
        a = Variable([2])
        b = Variable([2, 2, 3])
        z = (b / a) / b
        z.backward()

        self.assertTrue(np.array_equal(
            z.data, 1/a.data * np.ones_like(b.data)))
        self.assertTrue(np.array_equal(a.grad, [-0.75]))
        self.assertTrue(np.array_equal(b.grad, np.zeros_like(b.data)))

    def test_div_self2(self):
        a = Variable([2])
        b = Variable([2, 2, 3])
        z = (a / b) / a
        z.backward()

        self.assertTrue(np.array_equal(
            z.data, 1/b.data * np.ones_like(a.data)))
        self.assertTrue(np.array_equal(a.grad, [0]))
        self.assertTrue(np.array_equal(b.grad, -b.data**-2))

    def test_div_mul(self):
        a = Variable([4, 5, 4])
        b = Variable([2, 2, 3])
        z = (a/b)*a
        z.backward()

        self.assertTrue(np.array_equal(
            z.data, np.divide(a.data, b.data)*a.data))
        self.assertTrue(np.array_equal(a.grad, 2*a.data/b.data))
        self.assertTrue(np.array_equal(b.grad, -a.data**2/b.data**2))


if __name__ == '__main__':
    unittest.main()
