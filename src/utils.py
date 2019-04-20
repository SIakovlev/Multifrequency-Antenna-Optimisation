import numpy as np


def linear_phase_shift_matrix(N, delta):
    """
    Return phase shift matrix for N element array
    :param delta:
    :return: phase shifted vector x
    """
    shift_matrix = np.zeros(shape=(N, N), dtype=np.complex128)
    Delta = 0
    for i in range(N):
        shift_matrix[i, i] = np.exp(1j * Delta)
        Delta += delta
    return shift_matrix


def coord_descent(A, x, b, idx_list):
    """
    Coordinate gradient descent algorithm
    :param A: m by n matrix
    :param x: n by 1 vector
    :param b: m by 1 vector
    :param idx_list: list of indices for Gradient Descent iterations
    :return: x
    """

    num_iterations = 10
    if len(idx_list):
        for i in range(num_iterations):
            r = b - A @ x
            for j in idx_list:
                if j == -1:
                    continue
                a_j = A[:, j]
                x[j] = x[j] + np.dot(a_j, r) / np.dot(a_j, a_j)
                r = r - a_j * np.dot(a_j, r) / np.dot(a_j, a_j)
    return x