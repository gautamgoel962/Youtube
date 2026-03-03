import math 

class Value: 

    def __init__(self, data, _children=(), _op = '', label = ''): 
        self.data = data 
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op 
        self.label = label 
        self.grad = 0.0
        
    def __repr__(self): 
        return f"Value(data={self.data})"

    def __add__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        # c = a + b 
        # a.grad = 1 * c.grad 
        # b.grad = 1 * c.grad
        def _backward(): 
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad 
        out._backward = _backward
        return out

    def __radd__(self, other): 
        return self + other 

    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward(): 
            # c = a * b
            # a.grad = b * c.grad 
            # b.grad = a * c.grad 
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out._backward = _backward 
        return out

    def __rmul__(self, other): 
        return self * other 

    def __pow__(self, other): 
        if not isinstance(other, (int, float)): 
            raise ValueError("exponent must be an int or float")
        # assert isinstance(other, (int, float)), "exponent must be a number/float"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward(): 
            # c = a ** b
            # a.grad = b * (a ** (b-1)) * c.grad 
            self.grad += (other * (self.data ** (other -1))) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): 
        return self * -1

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other): 
        return other + (-self)

    def __truediv__(self, other): 
        return self * (other ** -1)

    def __rtruediv__(self, other): 
        return other * (self ** -1)
    
    # bad design choice
    # def sigmoid(self): 
    #     return 1/(1 + (Value(2.71828) ** (-self.data)))

    # good design choice
    def sigmoid(self): 
        s = 1 / (1 + (math.exp(-self.data)))
        out = Value(s, (self,), 'sigmoid')

        def _backward(): 
            # c = sigmoid(a) 
            # a.grad = (c * (1 - c)) * c.grad 
            self.grad += (out.data * (1 - out.data)) * out.grad 
        out._backward = _backward 
        return out

    # bad design choice, as during backprop, we have so many intermediate values gradient to be find out, which is less eficient, as complex functions, derivate, can be sometimes, very simple.
    # def tanh(self): 
    #     return ((Value(2.71828) ** (2 * self.data)) - 1) / ((Value(2.71828) ** (2 * self.data)) + 1)

    # good design choice, as less operations, will be needed in backprop, as it is 1 - self.data^2, so 2 operations only, but for previous one, where only *, **, +, -, then 8- 10 gradients interediate have to be found out, which is inefficient. 
    def tanh(self): 
        s = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(s, (self,), 'tanh')

        def _backward(): 
            # c = tanh(a)
            # a.grad = (1 - c^2) * c.grad
            self.grad += (1 - (out.data **2)) * out.grad 
        out._backward = _backward
        return out 

    # bad design choice, prev, and label cannot be maintained 
    # def relu(self): 
    #     if self.data > 0: 
    #         return self 
    #     else: 
    #         return Value(0.0)

    
    def relu(self): 
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward(): 
            # c = relu(a)
            # a.grad = (1 if a.data > 0 else 0) * c.grad 
            self.grad += (out.data > 0) * out.grad 
        out._backward = _backward 
        return out
    
    def backward(self): 

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        self.grad = 1.0
        build_topo(self)

        for v in reversed(topo): 
            v._backward()