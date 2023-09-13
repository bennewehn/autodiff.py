import numpy as np
from typing import SupportsFloat as Numeric


class Operator:
    def __init__(self) -> None:
        self.graph_label = 'not defined'

    def forward(self) -> 'Tensor':
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self):
        raise NotImplementedError(f"backward not implemented for {type(self)}")


class Tensor:
    def __init__(self,
                 data: list | np.ndarray,
                 requires_grad=False,
                 dtype=np.float32,
                 _children=(),
                 _fn: Operator | None = None):

        self.data: np.ndarray = np.array(
            data, dtype=dtype) if isinstance(data, list) else data
        self.dtype = dtype

        # autograd stuff
        self._prev: set[Tensor] = set(_children)
        self.grad: np.ndarray

        self.fn = _fn
        self.requires_grad = requires_grad

        if requires_grad:
            self.zero_grad()

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError(
                "Tensor doesn't require a gradient.")

        self.grad = np.ones_like(self.data)

        if self.fn is not None:
            self.fn.backward()

        else:
            return

        def _backward(var: Tensor):
            for child in var._prev:
                if child.requires_grad and child.fn is not None:
                    child.fn.backward()
                _backward(child)

        _backward(self)

    def __add__(self, other) -> 'Tensor':
        # overloading self + other
        return Add(self, self.__get_other(other)).forward()

    def __iadd__(self, other) -> 'Tensor':
        # overloading self += other
        self.data += self.__get_other(other).data
        return self

    def __radd__(self, other) -> 'Tensor':
        # overloading other + self
        return Add(self, self.__get_other(other)).forward()

    def __mul__(self, other) -> 'Tensor':
        # overloading self * other
        return Mul(self, self.__get_other(other)).forward()

    def __rmul__(self, other) -> 'Tensor':
        # overloading other * self
        return Mul(self, self.__get_other(other)).forward()

    def __truediv__(self, other) -> 'Tensor':
        # overloading self / other
        return Div(self, self.__get_other(other)).forward()

    def __neg__(self) -> 'Tensor':
        # overloading -self
        return Neg(self).forward()

    def __sub__(self, other) -> 'Tensor':
        # overloading self - other
        return self + (-other)

    def __isub__(self, other) -> 'Tensor':
        # overloading self -= other
        self.data -= self.__get_other(other).data
        return self

    def __rsub__(self, other) -> 'Tensor':
        # overloading other - self
        return self.__get_other(other) - self

    def __matmul__(self, other):
        # overloading self @ other
        return Dot(self, self.__get_other(other)).forward()

    '''
    def transpose(self):
        self.data = self.data.transpose()
        return self
    '''

    def __pow__(self, other):
        return Pow(self, self.__get_other(other)).forward()

    def pow(self, exponent) -> 'Tensor':
        return Pow(self, self.__get_other(exponent)).forward()

    def sum(self):
        return Sum(self).forward()

    def relu(self):
        return Relu(self).forward()

    def exp(self):
        return Exp(self).forward()

    def __repr__(self):
        return f"Tensor(data={self.data}, fn={self.fn}, requires_grad={self.requires_grad})"

    def __get_other(self, other) -> 'Tensor':
        if isinstance(other, np.ndarray):
            return Tensor(other, other.dtype)
        return other if isinstance(other, Tensor) else Tensor(np.array([other], dtype=self.dtype))

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def copy(self) -> 'Tensor':
        return Tensor(self.data, self.requires_grad, self.dtype,  self._prev, self.fn)


'''Binary ops'''


class Add(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        self.x, self.y = x, y
        self.graph_label = '+'

    def forward(self) -> Tensor:
        requires_grad = self.x.requires_grad or self.y.requires_grad

        self.out = Tensor(
            np.add(self.x.data, self.y.data),
            requires_grad=requires_grad,
            _children=(self.x, self.y),
            _fn=self)

        return self.out

    def backward(self):
        if self.x.requires_grad:
            grad = _handleBroadcasting(self.x.data.shape, self.out.grad)
            self.x.grad += grad

        if self.y.requires_grad:
            grad = _handleBroadcasting(self.y.data.shape, self.out.grad)
            self.y.grad += grad


class Mul(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        self.x, self.y = x, y
        self.graph_label = '*'

    def forward(self) -> Tensor:
        requires_grad = self.x.requires_grad or self.y.requires_grad

        self.out = Tensor(
            np.multiply(self.x.data, self.y.data),
            requires_grad=requires_grad,
            _children=(self.x, self.y),
            _fn=self)

        return self.out

    def backward(self):
        if self.x.requires_grad:
            grad = _handleBroadcasting(
                self.x.data.shape, self.out.grad * self.y.data)
            self.x.grad += grad

        if self.y.requires_grad:
            grad = _handleBroadcasting(
                self.y.data.shape, self.out.grad * self.x.data)
            self.y.grad += grad


class Div(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        self.x, self.y = x, y
        self.graph_label = 'div()'

    def forward(self) -> Tensor:
        requires_grad = self.x.requires_grad or self.y.requires_grad

        self.out = Tensor(
            np.divide(self.x.data, self.y.data),
            requires_grad=requires_grad,
            _children=(self.x, self.y),
            _fn=self)

        return self.out

    def backward(self):
        if self.x.requires_grad:
            grad = _handleBroadcasting(
                self.x.data.shape, (1 / self.y.data) * self.out.grad)
            self.x.grad += grad

        if self.y.requires_grad:
            grad = _handleBroadcasting(
                self.y.data.shape, -self.x.data * np.power(self.y.data, -2) * self.out.grad)
            self.y.grad += grad


def _handleBroadcasting(target_shape, data: np.ndarray) -> np.ndarray:
    # how many dimensions need to be reduced
    ndims = data.ndim - len(target_shape)

    for _ in range(ndims):
        # reduce dimensions
        data = data.sum(axis=0)

    # match desired shape
    for i, n in enumerate(target_shape):
        if n == 1:
            data = data.sum(axis=i, keepdims=True)

    return data


class Pow(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        self.x, self.y = x, y
        self.graph_label = '*'

    def forward(self) -> Tensor:
        requires_grad = self.x.requires_grad or self.y.requires_grad

        self.out = Tensor(
            np.power(self.x.data, self.y.data),
            requires_grad=requires_grad,
            _children=(self.x, self.y),
            _fn=self)

        return self.out

    def backward(self):
        if self.x.requires_grad:
            grad = _handleBroadcasting(self.x.data.shape, self.y.data *
                                       np.power(self.x.data, self.y.data - 1) * self.out.grad)
            self.x.grad += grad

        if self.y.requires_grad:
            grad = _handleBroadcasting(
                self.y.data.shape, self.out.data * np.log(self.x.data) * self.out.grad)

            self.y.grad += grad


class Dot(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        self.x, self.y = x, y
        self.graph_label = '@'

    def forward(self) -> Tensor:
        requires_grad = self.x.requires_grad or self.y.requires_grad

        self.out = Tensor(
            np.dot(self.x.data, self.y.data),
            requires_grad=requires_grad,
            _children=(self.x, self.y),
            _fn=self)

        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += np.dot(self.out.grad, self.y.data.T)

        if self.y.requires_grad:
            self.y.grad += np.dot(self.x.data.T, self.out.grad)


'''Reduce ops'''


class Sum(Operator):
    def __init__(self, x: Tensor):
        self.x = x
        self.graph_label = 'sum()'

    def forward(self) -> Tensor:
        self.out = Tensor(
            [np.sum(self.x.data, dtype=self.x.dtype)], _children=[self.x], _fn=self,
            requires_grad=self.x.requires_grad)

        return self.out

    def backward(self):
        self.x.grad += self.out.grad


'''Unary ops'''


class Neg(Operator):
    def __init__(self, x: Tensor):
        self.x = x
        self.graph_label = 'neg()'

    def forward(self) -> Tensor:

        self.out = Tensor(
            np.negative(self.x.data),
            requires_grad=self.x.requires_grad,
            _children=[self.x],
            _fn=self)

        return self.out

    def backward(self):
        self.x.grad += -1 * self.out.grad


class Exp(Operator):
    def __init__(self, x: Tensor):
        self.x = x
        self.graph_label = 'exp()'

    def forward(self):
        self.out = Tensor(
            np.exp(self.x.data, dtype=self.x.dtype),
            _children=[self.x],
            _fn=self,
            requires_grad=self.x.requires_grad)

        return self.out

    def backward(self):
        self.x.grad += self.out.data * self.out.grad


class Relu(Operator):
    def __init__(self, x: Tensor):
        self.x = x
        self.graph_label = 'relu()'

    def forward(self):
        self.out = Tensor(
            np.maximum(0, self.x.data, dtype=self.x.dtype),
            _children=[self.x], _fn=self,
            requires_grad=self.x.requires_grad)

        return self.out

    def backward(self):
        self.x.grad += np.where(self.x.data > 0, 1,
                                0).astype(self.x.dtype) * self.out.grad
