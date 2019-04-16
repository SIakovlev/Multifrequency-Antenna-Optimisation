import numpy as np
import itertools
from scipy.optimize import minimize
from utils import plotter


def f(A, x, b, weight):
    result = 0
    grad_list = []
    for A_i, x_i, b_i, w_i in zip(A, x, b, weight):
        x_i = np.asarray(x_i, dtype=np.float128)
        b_i = np.asarray(b_i, dtype=np.float128)
        result += w_i * np.linalg.norm(abs(A_i @ x_i) ** 2 - abs(b_i) ** 2)**2

        grad_k = 0
        for k in range(A_i.shape[0]):
            Q_k = np.real(A_i[k, :].reshape(1, -1).conj().T @ A_i[k, :].reshape(1, -1))
            grad_k += 4 * (x_i.T @ Q_k @ x_i - abs(b_i[k, :]) ** 2) * Q_k @ x_i * x_i
        grad_list.append(grad_k)

    return result, np.vstack(grad_list)


class Antenna:
    def __init__(self, params):

        self.N = params["number_of_antenna_elements"]
        self.d = params["distance_between_elements"]
        self.lambdas = params["wavelengths"]
        self.beam_resolution = params["beam_resolution"]

        phi_range = np.radians(np.arange(0, 180 + self.beam_resolution, self.beam_resolution))

        afs = []
        for i, lambda_i in enumerate(self.lambdas):
            afs.append(self.array_factor(self.N, 2 * np.pi / lambda_i, self.d, phi_range))
        self.afs = afs
        self.n_currents = len(afs)

        beams = []
        for i, af_i in enumerate(self.afs):
            beams.append(self.ref_beam(self.N, af_i))
        self.beams = beams

        plotter(phi_range, self.beams, ['k', 'r'])

        self.objective = None
        self.cons = None

        self.__M = None
        self.__eps = None

    def info(self):
        pass

    def set_objective(self, weights):

        def objective(J):
            J = J.reshape(-1, 1)
            list_J = np.split(np.exp(J), self.n_currents)
            return f(self.afs, list_J, self.beams, weights)

        self.objective = objective

    def set_constraints(self, eps):

        """
        Constraint of the form Mx <= -eps (or -Mx - eps >= 0)
        :param eps: design parameter large enough to make currents small
        :return:
        """

        temp = np.array([True if i in [0, 1] else False for i in range(self.n_currents)])
        template = list(set(itertools.permutations(temp, len(temp))))
        self.__M = np.array(np.bmat([[np.eye(self.N) if elem else np.eye(self.N) * 0 for elem in template_i] for template_i in template]))
        self.__eps = eps

        def alloc_linear(J):
            return (- self.__M @ J.reshape(-1, 1) - eps).squeeze()
        self.cons = {'type': 'ineq', 'fun': alloc_linear}

    def get_optimal_current_allocation(self, params):
        """

        The main routine where optimal current distribution is calculated for multiple currents at different frequencies
        """

        if self.objective is None:
            raise ValueError("Objective function is not set!")

        if self.cons is None:
            raise ValueError("Constraints are not set!")

        x0 = np.linalg.lstsq(self.__M, - np.ones((self.__M.shape[0], 1)) * self.__eps)[0].reshape(-1, )

        print("Initial norm of residual: {}".format(self.objective(x0)[0]))
        result = minimize(fun=self.objective,
                          x0=x0,
                          constraints=self.cons,
                          **params["general"],
                          options=params["options"])
        print("Optimisation problem solved. Resulting norm of residual: {}".format(self.objective(result.x)[0]))

        return np.split(result.x, self.n_currents)

    @staticmethod
    def array_factor(N, k, d, phi):
        """
        Calculate the array factor of a given antenna array
        :param N: number of antenna elements
        :param k: wave number
        :param d: distance between elements
        :param phi: array of azimuth angles
        :return: matrix A
        """

        M = phi.shape[0]
        A = np.zeros((M, N), dtype=np.complex128)

        for i in range(M):
            a = np.zeros((1, N), dtype=np.complex128)
            for j in range(N):
                a[:, j] = np.exp(1j * k * j * d * np.cos(phi[i]))
            A[i, :] = a
        return A.astype(dtype=np.complex128)

    @staticmethod
    def ref_beam(N, A, beam_type='hamming'):

        if beam_type == 'hamming':
            weights = np.hamming(N)
            weights = weights / np.linalg.norm(weights, 2)
            return abs(A @ weights).reshape(-1, 1)




