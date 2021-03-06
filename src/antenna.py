import numpy as np
import itertools
import scipy
from scipy.signal.windows import chebwin
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib2tikz
from scipy.optimize import minimize, LinearConstraint, Bounds, check_grad, BFGS
from utils import linear_phase_shift_matrix
from objective_function import f, g, h, L, f_lin


class Antenna:
    def __init__(self, params):

        self.N = params["N"]
        self.d = params["d"]
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

        # self.plot_ref_beams()

        self.objective = None
        self.objective_lin = None
        self.L = None
        self.jac = None
        self.hess = None
        self.cons = []
        self.bounds = None
        self.callback = None
        self.configuration = None

        self.I = np.zeros(shape=(self.n_currents, self.N))

        self.M = None
        self.eps = None

    def info(self):
        print(f"Antenna configuration:")
        print(f"\t- number of elements: {self.N}")
        print(f"\t- distance between elements: {self.d}")
        print(f"\t- beam wavelengths: {self.lambdas}")
        print(f"\t- beam resolution: {self.beam_resolution}")

    def plot_current_distribution(self, figsize=(10, 5), save=False):

        signals = [abs(I) for I in self.I]
        plot_names = [r'Current amplitude ($\lambda_{}$ = {}d)'.format(i, l) for i, l in enumerate(self.lambdas)]

        plt.figure(figsize=figsize)
        plt.ylabel(r"Current amplitude")
        plt.xlabel(r"Antenna element number")
        for i, (signal, label) in enumerate(zip(signals, plot_names)):
            plt.stem(signal, linefmt=f"C{i}", markerfmt=f"C{i}o", basefmt=f"C{i}", label=label, use_line_collection=True)
        plt.grid(True)
        plt.legend()
        if save:
            matplotlib2tikz.save('../results/current_distribution.tex')
        else:
            plt.show()

    def plot_ref_beams(self, figsize=(10, 5), save=False):

        signals = [abs(beam) for beam in self.beams]
        plot_names = [r'Beam ($\lambda_{}$ = {}d)'.format(i, l) for i, l in enumerate(self.lambdas)]

        plt.figure(figsize=figsize)
        plt.ylabel(r"Beam amplitude")
        plt.xlabel(r"Angle (rad)")
        for i, (signal, label) in enumerate(zip(signals, plot_names)):
            plt.plot(np.rad2deg(self.phi_range), signal, color=f"C{i}", label=label)
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        if save:
            matplotlib2tikz.save('../results/ref_beams.tex')
        else:
            plt.show()

    def plot_formed_beams(self, figsize=(10, 5), save=False):

        signals = [abs(af @ I) for af, I in zip(self.afs, self.I)]
        plot_names = [r'Beam ($\lambda_{}$ = {}d)'.format(i, l) for i, l in enumerate(self.lambdas)]

        plt.figure(figsize=figsize)
        plt.ylabel(r"Beam amplitude")
        plt.xlabel(r"Angle (rad)")
        for i, (signal, label) in enumerate(zip(signals, plot_names)):
            plt.plot(self.phi_range, signal, color=f"C{i}", label=label)
        plt.grid(True)
        plt.yscale('log')
        plt.legend()
        if save:
            matplotlib2tikz.save('../results/formed_beams.tex')
        else:
            plt.show()

    def set_hessian(self, weights, offset=0):
        def hessian(J):
            list_J = self.set_currents(J)
            return h(self.afs, list_J, self.beams, weights, offset=offset, mask=self.get_current_mask())
        self.hess = hessian

    def set_jacobian(self, weights, offset=0):
        def jacobian(J):
            list_J = self.set_currents(J)
            return g(self.afs, list_J, self.beams, weights, offset=offset, mask=self.get_current_mask())
        self.jac = jacobian

    def set_objective(self, weights, offset=0):
        def objective(J):
            list_J = self.set_currents(J)
            return f(self.afs, list_J, self.beams, weights, offset=offset)
        self.objective = objective

    def set_objective_lin(self, weights, offset=0):
        def objective(J, J0):
            list_J = self.set_currents(J)
            list_J0 = self.set_currents(J0)
            return f_lin(self.afs, list_J, self.beams, weights, list_J0, offset=offset)
        self.objective_lin = objective

    def set_lagrangian(self, weights, offset=0):
        # B = np.block([[0*np.eye(self.N), np.eye(self.N)], [np.eye(self.N), 0*np.eye(self.N)]])

        def lagrangian(J, lam, J0):
            list_J = self.set_currents(J)
            list_J0 = self.set_currents(J0)
            g = self.M @ J.reshape(-1, 1) + self.eps
            return L(self.afs, list_J, self.beams, weights, g, lam, list_J0, offset=offset)
        self.L = lagrangian

    def set_callback(self):
        def callback(J, info):
            print(f"Gradient check: {check_grad(self.objective, self.jac, J)}")
        self.callback = callback

    def set_configuration(self, configuration):
        self.configuration = configuration

    def set_currents(self, J):
        if self.configuration is None:
            return np.split(np.exp(J.reshape(-1, 1)), self.n_currents)
        # create a mask
        currents = np.zeros_like(self.I)
        for i, elem in enumerate(self.configuration):
            if elem is None:
                continue
            currents[elem, i] = np.exp(J[i])
        return np.split(currents.flatten().reshape(-1, 1), self.n_currents)

    def get_current_mask(self):
        if self.configuration is None:
            return None
        mask = np.zeros_like(self.I, dtype=bool)
        for i, elem in enumerate(self.configuration):
            if elem is None:
                continue
            mask[elem, i] = True
        return mask

    def set_allocation_constraint(self, eps, replace=True):

        """
        Constraint of the form Mx <= -eps (or -Mx - eps >= 0)
        :param eps: design parameter large enough to make currents small
        :return:
        """

        if eps is None:
            self.cons = []
        else:
            temp = np.array([True if i in [0, 1] else False for i in range(self.n_currents)])
            template = list(set(itertools.permutations(temp, len(temp))))
            self.M = np.array(np.bmat([[np.eye(self.N) if elem else np.eye(self.N) * 0 for elem in template_i]
                                       for template_i in template]))
            self.eps = eps
            if replace:
                self.cons = [LinearConstraint(self.M, -np.inf, -eps)]
            else:
                self.cons.append(LinearConstraint(self.M, -np.inf, -eps))

    def set_power_constraint(self, delta):

        """
        Constraint of the form Ix <= delta (or delta - Ix >= 0)
        :param eps: design parameter large enough to make currents small
        :return:
        """

        if delta is None:
            self.cons = None
        else:
            self.bounds = Bounds(-np.inf * np.ones(self.N * self.n_currents),
                                 np.log(delta) * np.ones(self.N * self.n_currents))

    def set_phase_shift(self, val_list):
        for i, val in enumerate(val_list):
            self.afs[i] = self.afs[i] @ linear_phase_shift_matrix(self.N, val)
            self.beams[i] = self.hamming_ref_beam(self.N, self.afs[i])

    def set_mutual_coupling(self, alpha_list):
        for i, alpha in enumerate(alpha_list):
            temp = [alpha ** i for i in range(0, self.N)]
            self.afs[i] = self.afs[i] @ scipy.linalg.toeplitz(temp)

    def get_optimal_current_allocation(self, params, x0=None, jac=True, hess=True, cons=True):
        """

        The main routine where optimal current distribution is calculated for multiple currents at different frequencies
        """

        if self.objective is None:
            raise ValueError("Objective function is not set!")

        if cons:
            if not self.cons:
                raise ValueError("Constraints are not set!")

            if x0 is None:
                x0 = np.linalg.lstsq(self.M, - np.ones((self.M.shape[0], 1)) * self.eps, rcond=None)[0].reshape(-1, )

        result = minimize(fun=self.objective,
                          x0=x0,
                          method='trust-constr',
                          jac=self.jac if jac else '3-point',
                          hess=self.hess if hess else BFGS(),
                          constraints=self.cons,
                          callback=self.callback,
                          options=params["options"],
                          bounds=self.bounds
                          )

        print()
        print(result.v)
        print()
        if cons:
            print(self.M @ result.x.reshape(-1, 1))

        self.I = self.set_currents(result.x)
        return result.fun, result.x

    def get_optimal_current_allocation_lin(self, params, x_lin, x0=None, jac=True, hess=True, cons=True):
        """

        The main routine where optimal current distribution is calculated for multiple currents at different frequencies
        """

        if self.objective is None:
            raise ValueError("Objective function is not set!")

        if cons:
            if not self.cons:
                raise ValueError("Constraints are not set!")

            if x0 is None:
                x0 = np.linalg.lstsq(self.M, - np.ones((self.M.shape[0], 1)) * self.eps, rcond=None)[0].reshape(-1, )

        result = minimize(fun=self.objective_lin,
                          x0=x0,
                          args=(x_lin,),
                          method='trust-constr',
                          jac=self.jac if jac else '3-point',
                          hess=self.hess if hess else BFGS(),
                          constraints=self.cons,
                          callback=self.callback,
                          options=params["options"],
                          bounds=self.bounds
                          )

        # print()
        # print(result.v)
        # print()
        # if cons:
        #     print(self.__M @ result.x.reshape(-1, 1))

        self.I = self.set_currents(result.x)
        return result.fun, result.x

    def get_dual_solution(self, params, x_lin, lam, x0=None):
        """

        The main routine where optimal current distribution is calculated for multiple currents at different frequencies
        """

        if self.objective is None:
            raise ValueError("Objective function is not set!")

        if x0 is None:
            x0 = np.ones(self.N * self.n_currents)

        result = minimize(fun=self.L,
                          x0=x0,
                          args=(lam, x_lin, ),
                          method='trust-constr',
                          jac='3-point',
                          hess=BFGS(),
                          callback=self.callback,
                          options=params["options"],
                          )
        return result.fun, result.x

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
    def hamming_ref_beam(N, A, scaling=1):
        # weights = np.hamming(N)
        weights = chebwin(N, 100)
        return scaling * (A @ weights.reshape(-1, 1))/max(abs(A @ weights.reshape(-1, 1)))
        # return scaling * (A @ weights.reshape(-1, 1))






