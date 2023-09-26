import numpy as np
from typing import List

class Tensor:
    def __init__(self, data, _producers: List["Tensor"]=[]):
        self.data = np.array(data)
        self.grad = None
        self._backward = None
        self._producers = _producers

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0), _producers=[self])
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def matmul(self, other: "Tensor") -> "Tensor":
        out = Tensor(np.matmul(self.data, other.data), _producers=[self, other])
        def _backward():
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)
        out._backward = _backward
        return out

    def sum(self) -> "Tensor":
        out = Tensor(np.sum(self.data), _producers=[self])
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def __add__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data + other.data, _producers=[self, other])
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Tensor({self.data}, grad={self.grad})"

    def backward(self):
        topo: List[Tensor] = []
        visited = set()
        def toposort(t: Tensor):
            if t not in visited:
                visited.add(t)
                for producer in t._producers:
                    toposort(producer)
                topo.append(t)
        toposort(self)

        # Loss gradient is 1. Initialise all other grads to zero.
        self.grad = 1
        for t in topo[:-1]:
            t.grad = 0

        # Backpropogate from the loss gradient
        for t in reversed(topo):
            if _b := t._backward:
                _b()

class SGD():
    def __init__(self, params: List[Tensor], lr: float=0.1):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad

if __name__ == "__main__":
    t0_data = np.random.rand(2, 2)
    t1_data = np.random.rand(2, 2)

    t0 = Tensor(t0_data)
    t1 = Tensor(t1_data)
    mm = t0.matmul(t1)
    relu = mm.relu()
    loss = relu.sum()
    loss.backward()
    optimizer = SGD([t0, t1], lr=0.1)
    optimizer.step()

    import torch
    t_t0 = torch.from_numpy(t0_data)
    t_t0.requires_grad = True
    t_t1 = torch.from_numpy(t1_data)
    t_t1.requires_grad = True
    t_mm = torch.matmul(t_t0, t_t1)
    t_relu = torch.relu(t_mm)
    t_loss = torch.sum(t_relu)
    t_loss.backward()
    optimizer = torch.optim.SGD([t_t0, t_t1], lr=0.1)
    optimizer.step()

    assert np.allclose(t0.data, t_t0.data)
    assert np.allclose(t1.data, t_t1.data)
