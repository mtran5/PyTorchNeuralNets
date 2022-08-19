import numpy as np
import random


def check_norm_diff(true, pred, tolerance=1e-5):
    return np.sqrt(np.sum((true - pred)**2)) < tolerance


def generate_random_matrices():
    """
    Randomly generate two matrices of different sizes
    """
    m = random.randint(1, 10)
    n = random.randint(1, 10)
    o = random.randint(1, 10)
    A = np.random.randint(-10, 10, (m, n))
    B = np.random.randint(-10, 10, (n, o))
    return A.tolist(), B.tolist()
