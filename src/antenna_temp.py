import numpy as np
import itertools
import scipy
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib2tikz
from scipy.optimize import minimize, Bounds, BFGS
from utils import linear_phase_shift_matrix
from objective_function import f_temp


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
        self.jac = None
        self.hess = None
        self.cons = []
        self.bounds = None
        self.callback = None
        self.configuration = None

        self.I = np.zeros(shape=(self.n_currents, self.N))

        self.__M = None
        self.__eps = None

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
            plt.stem(signal, linefmt=f"C{i}", markerfmt=f"C{i}o", basefmt=f"C{i}", label=label)
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
            plt.plot(self.phi_range, signal, color=f"C{i}", label=label)
        plt.grid(True)
        plt.legend()
        if save:
            matplotlib2tikz.save('../results/ref_beams.tex')
        else:
            plt.show()

    def plot_formed_beams(self, figsize=(10, 5), save=False):

        signals = [abs(af @ I)**2 for af, I in zip(self.afs, self.I)]
        plot_names = [r'Beam ($\lambda_{}$ = {}d)'.format(i, l) for i, l in enumerate(self.lambdas)]

        plt.figure(figsize=figsize)
        plt.ylabel(r"Beam amplitude")
        plt.xlabel(r"Angle (rad)")
        for i, (signal, label) in enumerate(zip(signals, plot_names)):
            plt.plot(self.phi_range, signal, color=f"C{i}", label=label)
        plt.grid(True)
        plt.legend()
        if save:
            matplotlib2tikz.save('../results/formed_beams.tex')
        else:
            plt.show()

    def set_objective(self, weights, offset=0):
        def objective(I):
            list_I = self.set_currents(I)
            return f_temp(self.afs, list_I, self.beams, weights, offset=offset)
        self.objective = objective

    def set_configuration(self, configuration):
        self.configuration = configuration

    def set_currents(self, I):
        if self.configuration is None:
            I = I.reshape(-1, 1)
            return np.split(I, self.n_currents)
        # create a mask
        currents = np.zeros_like(self.I)
        for i, elem in enumerate(self.configuration):
            if elem is None:
                continue
            currents[elem, i] = I[i]
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

    def set_allocation_constraint(self, eps):

        """
        Constraint of the form Mx <= -eps (or -Mx - eps >= 0)
        :param eps: design parameter large enough to make currents small
        :return:
        """

        pass

    def set_positivity_bounds(self):

        """
        :return:
        """
        if self.configuration is None:
            self.bounds = [(0, np.inf) for _ in range(self.N * self.n_currents)]
        else:
            self.bounds = [(0, np.inf) for _ in range(self.N)]

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
                x0 = np.linalg.lstsq(self.__M, - np.ones((self.__M.shape[0], 1)) * self.__eps, rcond=None)[0].reshape(-1, )

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

        # print()
        # print(result.v)
        # print()
        # if cons:
        #     print(self.__M @ result.x.reshape(-1, 1))

        self.I = self.set_currents(result.x)
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
    def hamming_ref_beam(N, A):

        weights = np.hamming(N)
        weights = weights / np.linalg.norm(weights, 2)
        return A @ weights.reshape(-1, 1)






