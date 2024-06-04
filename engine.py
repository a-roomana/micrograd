import numpy as np


class Value:
    def __init__(self, data, children=(), label="", op=""):
        self.data = data
        self.label = label
        self.grad = 0
        self.children = children
        self.op = op
        self.backward = lambda: None

    def __repr__(self):
        if self.label:
            return f"Value(data:{self.data} | label:{self.label})"
        return f"Value(data:{self.data})"

    def __add__(self, other):
        other = self.cast(other)
        out = Value(self.data + other.data, children=(self, other), op="+")

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward = backward
        return out

    def __mul__(self, other):
        other = self.cast(other)
        out = Value(self.data * other.data, children=(self, other), op="*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.backward = backward
        return out

    def __pow__(self, power):
        power = self.cast(power)
        out = Value(self.data**power.data, children=(self, power), op="**")

        def backward():
            self.grad += power.data * self.data ** (power.data - 1) * out.grad
            power.grad += np.log(self.data) * self.data**power.data * out.grad

        out.backward = backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):
        other = self.cast(other)
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def ln(self):
        out = Value(np.log(self.data), children=(self,), op="ln")

        def backward():
            self.grad += 1 / self.data * out.grad

        out.backward = backward
        return out

    def tanh(self):
        out = Value(np.tanh(self.data), children=(self,), op="tanh")

        def backward():
            self.grad += (1 - out.data**2) * out.grad

        out.backward = backward
        return out

    @staticmethod
    def cast(other):
        if not isinstance(other, Value):
            other = Value(other)
        return other

    def backward_children(self):
        # topological order all the children in the graph
        topo = []
        visited = set()

        def build_topologic(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build_topologic(child)
                topo.append(node)

        build_topologic(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v.backward()
