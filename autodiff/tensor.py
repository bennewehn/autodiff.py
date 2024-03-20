from numbers import Number
import numpy as np


class Operator:
    def __init__(self) -> None:
        self.graph_label = 'not defined'

    def forward(self) -> 'Tensor':
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self):
        raise NotImplementedError(f"backward not implemented for {type(self)}")


class Tensor:
    def __init__(self,
                 data: list | np.ndarray | np.float32,
                 requires_grad=False,
                 dtype=np.float32,
                 _children=(),
                 _op: Operator | None = None):

        self.data: np.ndarray = data if isinstance(
            data, np.ndarray) else np.array(data, dtype=dtype)

        self.dtype = dtype

        self._children: set[Tensor] = set(_children)
        self.grad: np.ndarray | None = None

        self.op = _op
        self.requires_grad: bool = requires_grad

        if requires_grad:
            self.zero_grad()

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError(
                "Tensor doesn't require a gradient.")

        self.grad = np.ones_like(self.data)

        if self.op is None:
            return

        visited: set[Tensor] = set()
        lst: list[Tensor] = list()

        def _topo_sort(t: Tensor):
            if t not in visited and t.requires_grad:
                for child in t._children:
                    _topo_sort(child)

                visited.add(t)

                # leaf tensor doesnt have a backward function
                if len(t._children) > 0:
                    lst.append(t)

        _topo_sort(self)

        for t in reversed(lst):
            assert t.op is not None
            t.op.backward()

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

    def __matmul__(self, other) -> 'Tensor':
        # overloading self @ other
        return Dot(self, self.__get_other(other)).forward()

    def transpose(self):
        return self.data.T

    def __pow__(self, other) -> 'Tensor':
        return Pow(self, self.__get_other(other)).forward()

    def pow(self, exponent) -> 'Tensor':
        return Pow(self, self.__get_other(exponent)).forward()

    def sum(self, dim=None, keepdim=False) -> 'Tensor':
        return Sum(self, dim, keepdim).forward()

    def mean(self) -> 'Tensor':
        return Mean(self).forward()

    def relu(self) -> 'Tensor':
        return Relu(self).forward()

    def exp(self) -> 'Tensor':
        return Exp(self).forward()

    def log(self) -> 'Tensor':
        return Log(self).forward()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self):
        return f"Tensor(data={self.data}, op={self.op}, requires_grad={self.requires_grad})"

    def __get_other(self, other) -> 'Tensor':
        if isinstance(other, np.ndarray):
            return Tensor(other, dtype=other.dtype)
        return other if isinstance(other, Tensor) else Tensor(np.array([other], dtype=self.dtype))

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def __getitem__(self, key):
        return Tensor(self.data.__getitem__(key))

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    def item(self) -> Number:
        return self.data.item()

    @property
    def shape(self):
        return self.data.shape


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
            _op=self)

        return self.out

    def backward(self):
        assert self.out.grad is not None

        if self.x.requires_grad:
            grad = _handleBroadcasting(self.x.shape, self.out.grad)
            self.x.grad += grad

        if self.y.requires_grad:
            grad = _handleBroadcasting(self.y.shape, self.out.grad)
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
            _op=self)

        return self.out

    def backward(self):
        if self.x.requires_grad:
            grad = _handleBroadcasting(
                self.x.shape, self.out.grad * self.y.data)
            self.x.grad += grad

        if self.y.requires_grad:
            grad = _handleBroadcasting(
                self.y.shape, self.out.grad * self.x.data)
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
            _op=self)

        return self.out

    def backward(self):
        assert self.out.grad is not None

        if self.x.requires_grad:
            grad = _handleBroadcasting(
                self.x.shape, (1 / self.y.data) * self.out.grad)
            self.x.grad += grad

        if self.y.requires_grad:
            grad = _handleBroadcasting(
                self.y.shape, -self.x.data * np.power(self.y.data, -2) * self.out.grad)
            self.y.grad += grad


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
            _op=self)

        return self.out

    def backward(self):
        assert self.out.grad is not None

        if self.x.requires_grad:
            grad = _handleBroadcasting(self.x.shape, self.y.data *
                                       np.power(self.x.data, self.y.data - 1) * self.out.grad)
            self.x.grad += grad

        if self.y.requires_grad:
            grad = _handleBroadcasting(
                self.y.shape, self.out.data * np.log(self.x.data) * self.out.grad)

            self.y.grad += grad


class Dot(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        self.x, self.y = x, y

        # (10, 784, ) @ (784,) => (10,)
        self.graph_label = '@'

    def forward(self) -> Tensor:
        requires_grad = self.x.requires_grad or self.y.requires_grad

        self.out = Tensor(
            np.dot(self.x.data, self.y.data),
            requires_grad=requires_grad,
            _children=(self.x, self.y),
            _op=self)

        return self.out

    def backward(self):
        assert self.out.grad is not None
        if self.x.requires_grad:
            # use outer product for matrix vector multiplication
            # don't use the outer product for vector vector multiplication
            mul = 0
            if self.y.data.ndim == 1 and len(self.out.grad.shape) > 0:
                mul = np.outer(self.out.grad, self.y.data.T)
            else:
                mul = np.dot(self.out.grad, self.y.data.T)

            self.x.grad += mul

        if self.y.requires_grad:
            self.y.grad += np.dot(self.x.data.T, self.out.grad)


'''Reduce ops'''


class Sum(Operator):
    def __init__(self, x: Tensor, dim: np.int32 | None = None, keepdim=False):
        self.x = x
        self.dim = dim
        self.graph_label = 'sum()'
        self.keepdim = keepdim

    def forward(self) -> Tensor:
        self.out = Tensor(
            np.sum(self.x.data, dtype=self.x.dtype,
                   axis=self.dim, keepdims=self.keepdim),
            _children=[self.x], _op=self,
            requires_grad=self.x.requires_grad)

        return self.out

    def backward(self):
        if self.dim is not None:
            assert self.out.grad is not None

            new_dim_arr = self.out.grad
            if not self.keepdim:
                # Add new dimension
                new_dim_arr = np.expand_dims(self.out.grad, axis=self.dim)

            # Repeat data along the new dimension
            resulting_array = np.repeat(
                new_dim_arr, self.x.shape[self.dim], axis=self.dim)

            self.x.grad += resulting_array
        else:
            assert self.out.grad is not None
            self.x.grad += self.out.grad


class Mean(Operator):
    def __init__(self, x: Tensor):
        self.x = x
        self.graph_label = 'mean()'

    def forward(self) -> Tensor:
        self.out = Tensor(
            np.mean(self.x.data, dtype=self.x.dtype), _children=[self.x], _op=self,
            requires_grad=self.x.requires_grad)

        return self.out

    def backward(self):
        assert self.out.grad is not None
        assert self.x.grad is not None

        frac = sum(self.x.shape)
        temp = (1 / (1 if frac == 0 else frac)) * self.out.grad
        self.x.grad += temp


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
            _op=self)

        return self.out

    def backward(self):
        assert self.x.grad is not None
        assert self.out.grad is not None

        self.x.grad += -1 * self.out.grad


class Exp(Operator):
    def __init__(self, x: Tensor):
        self.x = x
        self.graph_label = 'exp()'

    def forward(self):
        self.out = Tensor(
            np.exp(self.x.data, dtype=self.x.dtype),
            _children=[self.x],
            _op=self,
            requires_grad=self.x.requires_grad)

        return self.out

    def backward(self):
        assert self.x.grad is not None
        assert self.out.grad is not None

        self.x.grad += self.out.data * self.out.grad


class Log(Operator):
    def __init__(self, x: Tensor):
        self.x = x
        self.graph_label = 'log()'

    def forward(self):
        self.out = Tensor(
            np.log(self.x.data, dtype=self.x.dtype),
            _children=[self.x],
            _op=self,
            requires_grad=self.x.requires_grad)

        return self.out

    def backward(self):
        assert self.out.grad is not None
        assert self.x.grad is not None

        self.x.grad += (1 / self.x.data) * self.out.grad


class Relu(Operator):
    def __init__(self, x: Tensor):
        self.x = x
        self.graph_label = 'relu()'

    def forward(self):
        self.out = Tensor(
            np.maximum(0, self.x.data, dtype=self.x.dtype),
            _children=[self.x], _op=self,
            requires_grad=self.x.requires_grad)

        return self.out

    def backward(self):
        assert self.out.grad is not None
        assert self.x.grad is not None

        self.x.grad += np.where(self.x.data > 0, 1,
                                0).astype(self.x.dtype) * self.out.grad
