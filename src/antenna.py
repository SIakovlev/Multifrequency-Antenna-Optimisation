import numpy as np
import itertools
import scipy
import matplotlib.colors as colors
from scipy.optimize import minimize, LinearConstraint, BFGS, check_grad
from utils import plotter, linear_phase_shift_matrix
from objective_function import f, g, h


class Antenna:
    def __init__(self, params):

        self.N = params["number_of_antenna_elements"]
        self.d = params["distance_between_elements"]
        self.lambdas = params["wavelengths"]
        self.beam_resolution = params["beam_resolution"]

        self.phi_range = np.radians(np.arange(0, 180 + self.beam_resolution, self.beam_resolution))

        afs = []
        for i, lambda_i in enumerate(self.lambdas):
            afs.append(self.array_factor(self.N, 2 * np.pi / lambda_i, self.d, self.phi_range))
        self.afs = afs
        self.n_currents = len(afs)

        beams = []
        for i, af_i in enumerate(self.afs):
            beams.append(self.hamming_ref_beam(self.N, af_i))
        self.beams = beams

        self.plot_ref_beams()

        self.objective = None
        self.jac = None
        self.hess = None
        self.cons = None
        self.callback = None

        self.__I = None

        self.__M = None
        self.__eps = None

    def info(self):
        pass

    def plot_current_distribution(self):

        colors_list = list(colors._colors_full_map.values())
        plotter([],
                [abs(I) for I in np.exp(self.__I)],
                style_mod=[colors_list[i] for i in range(len(self.afs))],
                plot_type='stem')

    def plot_ref_beams(self):

        colors_list = list(colors._colors_full_map.values())
        plotter(self.phi_range,
                [abs(beam) for beam in self.beams],
                style_mod=[colors_list[i] for i in range(len(self.afs))])

    def plot_formed_beams(self):
        colors_list = list(colors._colors_full_map.values())
        plotter(self.phi_range,
                [abs(af @ I) for af, I in zip(self.afs, np.exp(self.__I))],
                style_mod=[colors_list[i] for i in range(len(self.afs))])

    def set_hessian(self, weights):
        def hessian(J):
            J = J.reshape(-1, 1)
            list_J = np.split(np.exp(J), self.n_currents)
            return h(self.afs, list_J, self.beams, weights)
        self.hess = hessian

    def set_jacobian(self, weights):
        def jacobian(J):
            J = J.reshape(-1, 1)
            list_J = np.split(np.exp(J), self.n_currents)
            return g(self.afs, list_J, self.beams, weights)
        self.jac = jacobian

    def set_objective(self, weights):
        def objective(J):
            J = J.reshape(-1, 1)
            list_J = np.split(np.exp(J), self.n_currents)
            return f(self.afs, list_J, self.beams, weights)
        self.objective = objective

    def set_callback(self):

        def callback(J, info):
            print(f"Gradient check: {check_grad(self.objective, self.jac, J)}")

        self.callback = callback

    def set_constraints(self, eps):

        """
        Constraint of the form Mx <= -eps (or -Mx - eps >= 0)
        :param eps: design parameter large enough to make currents small
        :return:
        """

        if eps is None:
            self.cons = None
        else:
            temp = np.array([True if i in [0, 1] else False for i in range(self.n_currents)])
            template = list(set(itertools.permutations(temp, len(temp))))
            self.__M = np.array(np.bmat([[np.eye(self.N) if elem else np.eye(self.N) * 0 for elem in template_i] for template_i in template]))
            self.__eps = eps
            self.cons = LinearConstraint(self.__M, -np.inf, -eps)

    def set_phase_shift(self, val_list):
        for i, val in enumerate(val_list):
            self.afs[i] = self.afs[i] @ linear_phase_shift_matrix(self.N, val)
            self.beams[i] = self.hamming_ref_beam(self.N, self.afs[i])

    def set_mutual_coupling(self, alpha_list):
        for i, alpha in enumerate(alpha_list):
            temp = [alpha ** i for i in range(0, self.N)]
            self.afs[i] = self.afs[i] @ scipy.linalg.toeplitz(temp)
            # self.beams[i] = self.hamming_ref_beam(self.N, self.afs[i])


    def get_optimal_current_allocation(self, params):
        """

        The main routine where optimal current distribution is calculated for multiple currents at different frequencies
        """

        if self.objective is None:
            raise ValueError("Objective function is not set!")

        if self.cons is None:
            raise ValueError("Constraints are not set!")

        x0 = np.linalg.lstsq(self.__M, - np.ones((self.__M.shape[0], 1)) * self.__eps, rcond=None)[0].reshape(-1, )
        # x0 = np.ones((self.N * self.n_currents))

        # if check_grad(self.objective, self.jac, x0) > 1e-6:
        #     raise Warning("Jacobian caclulation is not accurate")

        # print("Initial norm of residual: {}".format(self.objective(x0)[0]))
        # print(f"{'Iter number':<15}| {'Function value':<25} | {'Gradient norm':<25}")
        # print(f"{'========================================================================':<50}")
        result = minimize(fun=self.objective,
                          x0=x0,
                          method='trust-constr',
                          jac=self.jac,
                          hess=self.hess,
                          constraints=self.cons,
                          callback=self.callback,
                          options=params["options"]
                          )
        print()
        print(result.v)
        print()
        print(self.__M @ result.x.reshape(-1, 1))
        # print("Optimisation problem solved. Resulting norm of residual: {}".format(self.objective(result.x)[0]))

        self.__I = np.split(result.x, self.n_currents)
        return np.split(result.x, self.n_currents), result.fun

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
    def hamming_ref_beam(N, A):

        weights = np.hamming(N)
        weights = weights / np.linalg.norm(weights, 2)
        return A @ weights.reshape(-1, 1)




