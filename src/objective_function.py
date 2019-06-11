import numpy as np
import scipy
from scipy.optimize import minimize, LinearConstraint, Bounds, check_grad, BFGS
from numba import jit, njit

class Objective:
    """
    Base class for all the custom objective functions
    """

    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError

    def jac(self, x):
        raise NotImplementedError

    def hess(self, x):
        raise NotImplementedError

    def optimise(self, x0, constraints, bounds, options, method='trust-constr', jac='3-point', hess=BFGS(), callback=None):
        raise NotImplementedError


class L2PowerExp(Objective):

    def __init__(self, A, b, offset=0, mask=None):
        super(L2PowerExp, self).__init__()

        self.A = scipy.linalg.block_diag(*A)
        self.b = np.concatenate(b)
        self.offset = offset
        self.mask = mask

    @njit
    def __call__(self, x):
        x = np.exp(x).reshape(-1, 1)
        return np.linalg.norm(abs(self.A @ x) ** 2 - abs(self.b) ** 2 - self.offset, 2) ** 2

    @njit
    def jac(self, x):

        x = np.exp(x).reshape(-1, 1)

        grad = 0
        for k in range(self.A.shape[0]):
            Q_k = np.real(self.A[k, :].reshape(1, -1).conj().T @ self.A[k, :].reshape(1, -1))
            grad += 4 * (x.T @ Q_k @ x - abs(self.b[k, :]) ** 2 - self.offset) * Q_k @ x * x

        if self.mask is None:
            return grad.reshape(-1, )
        else:
            grads = grad.reshape(len(x), -1)
            grads = np.sum(grads * self.mask, axis=0)
            return grads

    @njit
    def hess(self, x):
        hess_list = []
        for A_i, x_i, b_i, w_i in zip(self.A, x, self.b):

            hess_k = 0
            for k in range(A_i.shape[0]):
                Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
                t0 = Q_k @ x_i * x_i
                t1 = x_i.T @ Q_k @ x_i - abs(b_i[k, :]) ** 2 - self.offset
                hess_k += 8 * t0 @ ((x_i.T @ Q_k) * x_i.T) \
                          + 4 * t1 * np.diag(x_i.squeeze()) @ Q_k @ np.diag(x_i.squeeze()) + 4 * t1 * np.diag(
                    t0.squeeze())

            hess_list.append(hess_k)

        if self.mask is not None:
            hess_list = [np.sum(np.array(hess_list), axis=0)]
        result = scipy.linalg.block_diag(*hess_list)
        return result

    def optimise(self, x0, constraints, bounds, options, fun=None, method='trust-constr', jac='3-point', hess=BFGS(), callback=None):

        if fun is None:
            fun = self

        result = minimize(fun=fun,
                          x0=x0,
                          method=method,
                          jac=jac,
                          hess=hess,
                          constraints=constraints,
                          callback=callback,
                          options=options,
                          bounds=bounds
                          )

        result.x = np.exp(result.x)

        return result


class L2PowerDecomposed(Objective):

    def __init__(self, A, b, offset=0, mask=None):
        super(L2PowerDecomposed, self).__init__()

        self.A = A
        self.b = b
        self.offset = offset
        self.mask = mask

        # list of Q matrices
        self.Q = []
        self.P = []
        for A_i in self.A:
            Q_temp = []
            P_temp = []
            for k in range(A_i.shape[0]):
                Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
                Q_temp.append(Q_k)

                eigenvalues, eigenvectors = np.linalg.eigh(Q_k)
                L_sq = np.diag(np.sqrt(np.clip(eigenvalues, 0, np.inf)))
                P_k = L_sq @ eigenvectors.T
                P_temp.append(P_k)

            self.Q.append(Q_temp)
            self.P.append(P_temp)

    def __call__(self, x):
        # x = x.reshape(-1, 1)
        result = 0
        for A_i, b_i, x_i, P_i in zip(self.A, self.b, x, self.P):
            for k in range(A_i.shape[0]):
                result += np.linalg.norm(P_i[k][-2:, :] @ x_i - np.ones((P_i[k][-2:, :].shape[0], 1))*abs(b_i[k]), 2)**2
        return result

    def jac(self, x):
        # x = x.reshape(-1, 1)
        result = 0
        for A_i, b_i, x_i, P_i in zip(self.A, self.b, x, self.P):
            for k in range(A_i.shape[0]):
                result += 2 * P_i[k] @ (P_i[k] @ x_i - np.ones((P_i[k].shape[0], 1))*b_i[k])
        return result

    def hess(self, x):
        result = 0
        for A_i, b_i, x_i, P_i in zip(self.A, self.b, x, self.P):
            for k in range(A_i.shape[0]):
                result += 2 * P_i[k].T @ P_i[k]
        return result

    def optimise(self, x0, constraints, bounds, options, fun=None, method='trust-constr', jac='3-point', hess=BFGS(), callback=None):

        if fun is None:
            fun = self
        result = minimize(fun=fun,
                          x0=x0,
                          method=method,
                          jac=jac,
                          hess=hess,
                          constraints=constraints,
                          callback=callback,
                          options=options,
                          bounds=bounds
                          )
        return result


class L2Power(Objective):

    def __init__(self, A, b, offset=0, mask=None):
        super(L2Power, self).__init__()

        self.A = A
        self.b = b
        self.offset = offset
        self.mask = mask

        self.Q_list = []
        for A_i in self.A:
            for k in range(A_i.shape[0]):
                Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
                self.Q_list.append(Q_k)

    def __call__(self, x):
        # x = x.reshape(-1, 1)
        result = 0
        for A_i, x_i, b_i in zip(self.A, x, self.b):
            result += np.linalg.norm(abs(A_i @ x_i) ** 2 - abs(b_i) ** 2 - self.offset, 2) ** 2
        return result

    def jac(self, x):
        grad_list = []
        for A_i, x_i, b_i in zip(self.A, x, self.b):

            grad_k = 0
            for k in range(A_i.shape[0]):
                Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
                grad_k += 4 * (x_i.T @ Q_k @ x_i - abs(b_i[k, :]) ** 2 - self.offset) * Q_k @ x_i * x_i
            grad_list.append(grad_k)

        if self.mask is None:
            result = np.vstack(grad_list).reshape(-1, )
            return result
        else:
            grads = np.array(grad_list).reshape(len(x), -1)
            grads = np.sum(grads * self.mask, axis=0)
            return grads

    def hess(self, x):
        hess_list = []
        for A_i, x_i, b_i in zip(self.A, x, self.b):

            hess_k = 0
            for k in range(A_i.shape[0]):
                Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
                t0 = Q_k @ x_i * x_i
                t1 = x_i.T @ Q_k @ x_i - abs(b_i[k, :]) ** 2 - self.offset
                hess_k += 8 * t0 @ ((x_i.T @ Q_k) * x_i.T) \
                          + 4 * t1 * np.diag(x_i.squeeze()) @ Q_k @ np.diag(x_i.squeeze()) + 4 * t1 * np.diag(
                    t0.squeeze())

            hess_list.append(hess_k)

        if self.mask is not None:
            hess_list = [np.sum(np.array(hess_list), axis=0)]
        result = scipy.linalg.block_diag(*hess_list)
        return result

    def optimise(self, x0, constraints, bounds, options, fun=None, method='trust-constr', jac='3-point', hess=BFGS(), callback=None):

        if fun is None:
            fun = self

        B = np.block([[0 * np.eye(100), np.eye(100)], [np.eye(100), 0 * np.eye(100)]])

        def cons(x):
            x = x.reshape(-1, 1)
            return float(x.T @ B @ x)

        constraints = [{'type': 'eq', 'fun': cons}, {'type': 'ineq', 'fun': lambda x: x}]

        result = minimize(fun=fun,
                          x0=x0,
                          method=method,
                          jac=jac,
                          hess=hess,
                          constraints=constraints,
                          callback=callback,
                          options=options,
                          bounds=bounds
                          )

        return result


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