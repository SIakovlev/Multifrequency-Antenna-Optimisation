import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds


def im2real_matrix(A):
    return np.vstack(
        (np.concatenate((np.real(A), -np.imag(A)), axis=1), np.concatenate((np.imag(A), np.real(A)), axis=1)))


def im2real_vector(x):
    return np.hstack((np.real(x), np.imag(x)))


def real2im_vector(x):
    return np.split(x, 2)


def obj_function(A, x, b, weight=1, value_type='real_abs', norm_type=2):
    """

    :param A:
    :param x:
    :param b:
    :param weight:
    :param value_type:
    :param norm_type:
    :return:
    """

    if len(A.shape) != 2:
        print("Matrix A should have a shape: (m, n) \n")
        print("Your matrix A shape is {} \n".format(A.shape))
        return
    if len(x.shape) != 1:
        print("Vector x should have a shape: (n,) \n")
        print("Your x shape is {} \n".format(x.shape))
        return
    if len(b.shape) != 1:
        print("Vector b should have a shape: (m,) \n")
        print("Your b shape is {}".format(b.shape))
        return

    if value_type == 'real':
        real_A = im2real_matrix(A)
        real_x = im2real_vector(x)
        real_b = im2real_vector(b)
        return weight * np.linalg.norm(real_A @ real_x - real_b, ord=norm_type)
    elif value_type == 'real_abs':
        return weight * np.linalg.norm(abs(A @ x) - b, ord=norm_type)
    elif value_type == 'complex':
        return weight * np.linalg.norm(A @ x - b, ord=norm_type)


def array_factor(N, k, d, phi):
    """
    Calculate the array factor of a given antenna array
    :param N: number of antennas
    :param k: wave number
    :param d: distance between elements
    :param phi: array of azimuth angles
    :return: matrix A
    """

    M = phi.shape[0]
    A = np.zeros((M, N), dtype=np.complex_)

    for i in range(M):
        a = np.zeros((1, N), dtype=np.complex_)
        for j in range(N):
            a[:, j] = np.exp(1j * k * j * d * np.cos(phi[i]))
        A[i, :] = a
    return A


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


def phase_shift(x, delta, shift_type='linear'):
    """
    Add phase shift to the elements of vector x
    :param x:
    :param delta:
    :param shift_type:
    :return: phase shifted vector x
    """
    num_elements = x.shape[0]
    x_shifted = np.array(x, dtype=np.complex_)
    if shift_type == 'linear':
        Delta = 0
        for i in range(num_elements):
            x_shifted[i] = x[i] * np.exp(1j * Delta)
            Delta += delta
        return x_shifted
    else:
        print("Please specify phaseType value \n")


def ref_beam(N, A, position=90, beam_type='hamming', phi_step=0.1):
    phi_start = position - 90
    phi_end = position + 90
    phi_range = np.radians(np.arange(phi_start, phi_end, phi_step))
    if not A:
        lambda_ref = 1
        A = array_factor(N, 2 * np.pi / lambda_ref, lambda_ref / 2, phi_range)
    # Create weights for desired pattern
    if beam_type == 'hamming':
        weights = np.hamming(N)
        weights = weights / np.linalg.norm(weights, 2)
        return abs(A @ weights)


def plotter(data_x, data_y, style_mod, figsize=(10, 5), plot_type='normal'):
    """
    High level customised API for plotting.
    :param data_x:
    :param data_y:
    :param style_mod:
    :param figsize:
    :param plot_type:
    :return:
    """

    plt.figure(figsize=(10, 5))
    if plot_type == 'normal':
        for dataYElem, styleElem in zip(data_y, style_mod):
            plt.plot(data_x, dataYElem, styleElem) if len(data_x) else plt.plot(dataYElem, styleElem)
    elif plot_type == 'stem':
        for dataYElem, styleElem in zip(data_y, style_mod):
            plt.stem(data_x, dataYElem, styleElem) if len(data_x) else plt.stem(dataYElem, styleElem)
    plt.grid(True)
    plt.show()


def get_optimal_current_distr(AFList, beam_list, weight_list):
    """
    The main routine where optimal current distribution is calculated for multiple currents at different frequencies
    """
    # number of antenna elements corresponds to the number of columns in the array factor matrix
    n_elements = AFList[0].shape[1]
    n_currents = len(AFList)

    # define objective function
    f = lambda I: sum(map(obj_function, AFList, np.exp(np.split(I, n_currents)), beam_list, weight_list))

    # define constraints in matrix form: Cx - e >= 0
    C = np.empty(shape=(n_elements, n_elements * n_currents))
    for i in range(n_currents):
        C[:, i * n_elements: (i + 1) * n_elements] = np.eye(n_elements, dtype=np.float)
    cons = {'type': 'eq', 'fun': lambda I: -C @ I - 20}
    # cons = {'type':'eq', 'fun': lambda I: -C@I - 20}
    # bnds = Bounds(-np.inf*np.ones(numOfAntennaElements * numOfCurrents), 10*np.ones(numOfAntennaElements * numOfCurrents))

    # define initial value
    x0 = np.ones(n_elements * n_currents)

    print("Initial norm of residual: {}".format(f(x0)))

    # run optimisation solver
    # result = minimize(fun=f, x0=x0, constraints=cons, options={'ftol': 1e-8, 'disp': True, 'maxiter': 1000}, bounds=bnds)
    result = minimize(fun=f, x0=x0, constraints=cons)

    print("Optimisation problem solved. Resulting norm of residual: {}".format(f(result.x)))

    return np.split(result.x, n_currents)
