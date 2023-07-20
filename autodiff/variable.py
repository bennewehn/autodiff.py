import numpy as np


class Function:
    def __init__(self) -> None:
        self.graph_label = 'not defined'

    def forward(self):
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self):
        raise NotImplementedError(f"backward not implemented for {type(self)}")


class Variable:
    def __init__(self, data: list | np.ndarray, dtype=np.float32, _children=(), _fn: Function | None = None):
        self.data: np.ndarray = np.array(
            data, dtype=dtype) if isinstance(data, list) else data
        self.dtype = dtype

        # autograd stuff
        self._prev: set[Variable] = set(_children)
        self.grad: np.ndarray = np.zeros(
            self.data.shape, dtype=self.data.dtype)
        self.fn = _fn

    def backward(self):
        self.grad.fill(1)

        if self.fn is not None:
            self.fn.backward()
        else:
            raise RuntimeError("No backward path.")

        def _backward(var: Variable):
            for child in var._prev:
                if child.fn is not None:
                    child.fn.backward()
                _backward(child)

        _backward(self)

    def __add__(self, other):
        return Add(self, self.__get_other(other)).forward()

    def __mul__(self, other):
        return Mul(self, self.__get_other(other)).forward()

    def __truediv__(self, other):
        return Div(self, self.__get_other(other)).forward()

    def __matmul__(self, other):
        return Dot(self, self.__get_other(other)).forward()

    def __pow__(self, other):
        return Pow(self, self.__get_other(other)).forward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def sum(self):
        return Sum(self).forward()

    def exp(self):
        return Exp(self).forward()

    def relu(self):
        return Relu(self).forward()

    def __repr__(self):
        return f"Variable(data={self.data}, fn={self.fn}, grad={self.grad})"

    def __get_other(self, other):
        return other if isinstance(other, Variable) else Variable(np.array([other], dtype=self.dtype))

    def copy(self):
        return Variable(self.data, self.dtype, self._prev, self.fn)


'''Binary ops'''


class Add(Function):
    def __init__(self, x: Variable, y: Variable):
        self.x, self.y = x, y
        self.graph_label = '+'

    def forward(self):
        self.out = Variable(np.add(self.x.data, self.y.data),
                            _children=(self.x, self.y), _fn=self)
        return self.out

    def backward(self):
        self.x.grad += np.ones(self.x.data.shape,
                               dtype=self.x.dtype) * self.out.grad
        self.y.grad += np.ones(self.x.data.shape,
                               dtype=self.x.dtype) * self.out.grad


class Dot(Function):
    def __init__(self, x: Variable, y: Variable):
        self.x, self.y = x, y
        self.graph_label = 'dot()'

    def forward(self):
        self.out = Variable(np.array([np.dot(
            self.x.data, self.y.data)], dtype=self.x.dtype), _children=(self.x, self.y), _fn=self)
        return self.out

    def backward(self):
        self.x.grad += self.y.data * self.out.grad
        self.y.grad += self.x.data * self.out.grad


class Mul(Function):
    def __init__(self, x: Variable, y: Variable):
        self.x, self.y = x, y
        self.graph_label = 'mul()'

    def forward(self):
        self.out = Variable(np.multiply(self.x.data, self.y.data).astype(
            self.x.dtype), _children=(self.x, self.y), _fn=self)
        return self.out

    def backward(self):
        x = self.y.data * self.out.grad
        y = self.x.data * self.out.grad
        # match scalar shape
        if self.x.grad.shape[0] == 1:
            self.x.grad += x.sum()
        else:
            self.x.grad += x
        if self.y.grad.shape[0] == 1:
            self.y.grad += y.sum()
        else:
            self.y.grad += y


class Div(Function):
    def __init__(self, x: Variable, y: Variable):
        self.x, self.y = x, y
        self.graph_label = 'div()'

    def forward(self):
        self.out = Variable(np.divide(self.x.data, self.y.data).astype(
            self.x.dtype), _children=(self.x, self.y), _fn=self)
        return self.out

    def backward(self):
        x = (1 / self.y.data) * self.out.grad
        y = np.array(
            (-self.x.data * np.power(self.y.data, -2) * self.out.grad))
        if self.x.grad.shape[0] == 1:
            self.x.grad += x.sum()
        else:
            self.x.grad += x
        if self.y.grad.shape[0] == 1:
            self.y.grad += y.sum()
        else:
            self.y.grad += y


class Pow(Function):
    def __init__(self, x: Variable, y: Variable):
        self.x, self.y = x, y
        self.graph_label = f'^{y.data}'

    def forward(self):
        self.out = Variable(np.power(self.x.data, self.y.data,
                            dtype=self.x.dtype), _children=[self.x, self.y], _fn=self)
        return self.out

    def backward(self):
        y = self.out.data * np.log(self.x.data) * self.out.grad
        x = self.y.data * np.power(self.x.data, self.y.data-1) * self.out.grad
        if self.y.grad.shape[0] == 1:
            self.y.grad += y.sum()
        else:
            self.y.grad += y
        if self.x.grad.shape[0] == 1:
            self.x.grad += x.sum()
        else:
            self.x.grad += x


'''Unary ops'''


class Exp(Function):
    def __init__(self, x: Variable):
        self.x = x
        self.graph_label = 'exp()'

    def forward(self):
        self.out = Variable(
            np.exp(self.x.data, dtype=self.x.dtype), _children=[self.x], _fn=self)
        return self.out

    def backward(self):
        self.x.grad += self.out.data * self.out.grad


class Relu(Function):
    def __init__(self, x: Variable):
        self.x = x
        self.graph_label = 'relu()'

    def forward(self):
        self.out = Variable(
            np.maximum(0, self.x.data, dtype=self.x.dtype), _children=[self.x], _fn=self)
        return self.out

    def backward(self):
        self.x.grad += np.where(self.x.data > 0, 1,
                                0).astype(self.x.dtype) * self.out.grad


'''Reduce ops'''


class Sum(Function):
    def __init__(self, x: Variable):
        self.x = x
        self.graph_label = 'sum()'

    def forward(self):
        self.out = Variable(
            [np.sum(self.x.data, dtype=self.x.dtype)], _children=[self.x], _fn=self)
        return self.out

    def backward(self):
        self.x.grad += np.ones(self.x.data.shape,
                               dtype=self.x.dtype) * self.out.grad
