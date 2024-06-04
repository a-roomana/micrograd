import random

from micrograd.engine import Value


class Neuron:
    def __init__(self, input_size: int, name=""):
        self.name = name
        self.weights = [
            Value(random.uniform(-1, 1), label=f"{name}w{i}") for i in range(input_size)
        ]
        self.bias = Value(random.uniform(-1, 1), label=f"{name}b")

    def __call__(self, x: list[Value]):
        # AX + B
        out = sum(
            [wi * xi for wi, xi in zip(self.weights, x, strict=True)], start=self.bias
        )
        out = out.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, i_size: int, o_size: int, name=""):
        self.neurons = [Neuron(i_size, name=f"{name}-{i}") for i in range(o_size)]

    def __call__(self, x: list[Value]):
        out = [neuron(x) for neuron in self.neurons]
        if len(out) == 1:
            return out[0]
        return out

    def parameters(self):
        return [
            parameter for neuron in self.neurons for parameter in neuron.parameters()
        ]


class NLP:
    def __init__(
        self,
        # the sizes must be compatible with matrix multiplication
        input_layer_size: int,
        hidden_layer_size: list[int],
        output_layer_size: int,
    ):
        sizes = [input_layer_size] + hidden_layer_size + [output_layer_size]
        self.layers = [
            Layer(i_size=sizes[i - 1], o_size=sizes[i], name=f"L{i}")
            for i in range(1, len(sizes))
        ]

    def __call__(self, x: list[Value]):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def train_iter(self, x_trains: list[list[float]], y_trains: list[float], lr: float):
        y_predictions = [self(x) for x in x_trains]
        loss = sum(
            [
                (y - y_pred) ** 2
                for y, y_pred in zip(y_trains, y_predictions, strict=False)
            ],
            start=Value(0.0),
        )
        loss.backward_children()

        for p in self.parameters():
            p.data -= lr * p.grad
            p.grad = 0

        return loss

    def train(self, x_trains, y_trains, lr=0.01, epochs=100):
        loss = None
        for epoch in range(epochs):
            loss = self.train_iter(x_trains, y_trains, lr)
            if epoch % 50 == 0:
                print(f"Epoch {epoch} | Loss {loss.data}")
                yield loss
        return loss
