import numpy as np
import scipy


def fg(A, x, b, weight):
    result = 0
    grad_list = []
    for A_i, x_i, b_i, w_i in zip(A, x, b, weight):

        result += w_i * np.linalg.norm(abs(A_i @ x_i) ** 2 - abs(b_i) ** 2, 2) ** 2

        grad_k = 0
        for k in range(A_i.shape[0]):
            Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
            grad_k += 4 * (x_i.T @ Q_k @ x_i - abs(b_i[k, :]) ** 2) * Q_k @ x_i * x_i
        grad_list.append(w_i * grad_k)

    return result, np.vstack(grad_list) + np.random.randn()


def g(A, x, b, weight, offset=0):
    grad_list = []
    for A_i, x_i, b_i, w_i in zip(A, x, b, weight):

        grad_k = 0
        for k in range(A_i.shape[0]):
            Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
            grad_k += 4 * (x_i.T @ Q_k @ x_i - abs(b_i[k, :]) ** 2 - offset) * Q_k @ x_i * x_i
        grad_list.append(w_i * grad_k)

    return np.vstack(grad_list).reshape(-1, )


def f(A, x, b, weight, offset=0):
    result = 0
    for A_i, x_i, b_i, w_i in zip(A, x, b, weight):
        result += w_i * np.linalg.norm(abs(A_i @ x_i) ** 2 - abs(b_i) ** 2 - offset, 2) ** 2

    return result


def h(A, x, b, weight, offset=0):
    hess_list = []
    for A_i, x_i, b_i, w_i in zip(A, x, b, weight):

        hess_k = 0
        for k in range(A_i.shape[0]):
            Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
            t0 = Q_k @ x_i * x_i
            t1 = x_i.T @ Q_k @ x_i - abs(b_i[k, :]) ** 2 - offset
            hess_k += 8 * t0 @ ((x_i.T @ Q_k) * x_i.T) \
                      + 4 * t1 * np.diag(x_i.squeeze()) @ Q_k @ np.diag(x_i.squeeze()) + 4 * t1 * np.diag(t0.squeeze())
        hess_list.append(hess_k)

    return scipy.linalg.block_diag(*hess_list)