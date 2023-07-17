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
        self.data: np.ndarray = np.array(data, dtype=dtype) if isinstance(data, list) else data
        self.dtype = dtype

        # autograd stuff
        self._prev: set[Variable] = set(_children)
        self.grad: np.ndarray = np.zeros(self.data.shape, dtype=self.data.dtype)
        self.fn = _fn
    
    def backward(self):
        self.grad.fill(1)

        if self.fn != None: self.fn.backward()
        else: raise RuntimeError("No backward path.")

        def _backward(var: Variable):
            for child in var._prev:
                if child.fn != None:
                    child.fn.backward()
                # chain rule
                # scalar edge cases to match shape
                if child.grad.shape[0] == 1 and var.grad.shape[0] > 1:
                    child.grad *= var.grad[0]
                else:
                    child.grad *= var.grad
                _backward(child)
        
        _backward(self)
        
    def __add__(self, other):
        return Add(self, self.__get_other(other)).forward()
    
    def __mul__(self, other):
        return Mul(self, self.__get_other(other)).forward()

    def __matmul__(self, other):
        return Dot(self, self.__get_other(other)).forward()

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def sum(self):
        return Sum(self).forward()
    
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
        return Variable(np.add(self.x.data, self.y.data), _children=(self.x, self.y), _fn=self)
    
    def backward(self):
        self.x.grad += np.ones(self.x.data.shape, dtype=self.x.dtype)
        self.y.grad += np.ones(self.x.data.shape, dtype=self.x.dtype)

class Dot(Function):
    def __init__(self, x: Variable, y: Variable):
        self.x, self.y = x, y
        self.graph_label = 'dot'

    def forward(self):
        return Variable(np.array([np.dot(self.x.data, self.y.data)], dtype=self.x.dtype), _children=(self.x, self.y), _fn=self)
    
    def backward(self):
        self.x.grad += self.y.data
        self.y.grad += self.x.data

class Mul(Function):
    def __init__(self, x: Variable, y: Variable):
        if x is y: y = y.copy()
        self.x, self.y = x, y
        self.graph_label = '*'

    def forward(self):
        return Variable(np.multiply(self.x.data, self.y.data).astype(self.x.dtype), _children=(self.x, self.y), _fn=self)
    
    def backward(self):
        # scalar value edge cases
        if self.x.grad.shape[0] == 1:
            self.x.grad += self.y.data.sum()
        else:
            self.x.grad += self.y.data
        if self.y.grad.shape[0] == 1:
            self.y.grad += self.x.data.sum()
        else:
            self.y.grad += self.x.data # *= to fix prob


'''Reduce ops'''
class Sum(Function):
    def __init__(self, x: Variable):
        self.x = x
        self.graph_label = 'sum()'

    def forward(self):
        return Variable([np.sum(self.x.data, dtype=self.x.dtype)], _children=[self.x], _fn=self)
    
    def backward(self):
        self.x.grad += np.ones(self.x.data.shape, dtype=self.x.dtype)