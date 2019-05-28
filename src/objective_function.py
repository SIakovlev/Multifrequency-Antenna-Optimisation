import numpy as np
import scipy


def g(A, x, b, weight, offset=0, mask=None):
    grad_list = []
    for A_i, x_i, b_i, w_i in zip(A, x, b, weight):

        grad_k = 0
        for k in range(A_i.shape[0]):
            Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
            grad_k += 4 * (x_i.T @ Q_k @ x_i - abs(b_i[k, :]) ** 2 - offset) * Q_k @ x_i * x_i
        grad_list.append(w_i * grad_k)

    if mask is None:
        result = np.vstack(grad_list).reshape(-1, )
        return result
    else:
        grads = np.array(grad_list).reshape(len(x), -1)
        grads = np.sum(grads * mask, axis=0)
        return grads


def L(A, x, b, weight, g, lam, x0, offset=0):
    result = f_lin(A, x, b, weight, x0, offset)
    result += abs(float(lam.T @ g))
    return result


def f_temp(A, x, b, weight, offset=0):
    result = 0
    for A_i, x_i, b_i, w_i in zip(A, x, b, weight):
        result += w_i * np.linalg.norm(A_i @ x_i - abs(b_i), 2) ** 2
    return result


def f(A, x, b, weight, offset=0):
    result = 0
    for A_i, x_i, b_i, w_i in zip(A, x, b, weight):
        result += w_i * np.linalg.norm(abs(A_i @ x_i) ** 2 - abs(b_i) ** 2 - offset, 2) ** 2
    return result


def f_lin(A, x, b, weight, x0, offset=0):
    result = 0
    for A_i, x_i, b_i, w_i, x0_i in zip(A, x, b, weight, x0):
        for k in range(A_i.shape[0]):
            Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
            result += w_i * np.linalg.norm(x0_i.T @ Q_k @ x0_i + (Q_k @ x0_i * x0_i).T@(x_i - x0_i) - abs(b_i[k, :]) ** 2 - offset, 2) ** 2
    return result


def h(A, x, b, weight, offset=0, mask=None):
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

    if mask is not None:
        hess_list = [np.sum(np.array(hess_list), axis=0)]
    result = scipy.linalg.block_diag(*hess_list)
    return result


