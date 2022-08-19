from typing import List, Callable
import numpy as np


def matmul_naive(A: List[List], B: List[List]) -> List[List]:
    """
    Size of A is (m x n)
    Size of B is (n x o)
    Size of output is (m x o)
    """
    m = len(A)
    n = len(A[0])
    o = len(B[0])
    result = [[0] * o for _ in range(m)]
    for i in range(m):
        for k in range(n):
            for j in range(o):
                result[i][j] += A[i][k] * B[k][j]
    return result


class Linear():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.stdev = 1e-4
        self.X = None
        self.w = np.random.randn(input_size, output_size) * self.stdev
        self.b = np.zeros((output_size, ))

        self.w_grads = None
        self.b_grads = None
        self.X_grads = None

    def __call__(self, X: np.array) -> np.array:
        self.X = X
        return np.dot(X, self.w) + self.b

    def backward(self, loss_grads: np.array):
        self.w_grads = np.dot(self.X.T, loss_grads)
        self.b_grads = np.sum(loss_grads, axis=0, keepdims=False)
        self.X_grads = np.dot(loss_grads, self.w.T)


class TwoLayersNet():
    def __init__(self, in_features: int=2, num_hidden: int=4, out_features: int=1):
        self.linear1 = Linear(in_features, num_hidden)
        self.linear2 = Linear(num_hidden, out_features)

    def __call__(self, X: np.array) -> np.array:
        return self.linear2(self.linear1(X))

    def backward(self, loss_grads: np.array):
        self.linear2.backward(loss_grads)
        self.linear1.backward(self.linear2.X_grads)

    def update_weight(self, optimizer: Callable):
        # Apply gradient descent on the weights
        optimizer(self.linear1.w, self.linear1.w_grads)
        optimizer(self.linear1.b, self.linear1.b_grads)
        optimizer(self.linear2.w, self.linear2.w_grads)
        optimizer(self.linear2.b, self.linear2.b_grads)
