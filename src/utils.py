import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


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
            plt.stem(data_x, dataYElem, linefmt=styleElem) if len(data_x) else plt.stem(dataYElem, linefmt=styleElem)
    plt.grid(True)
    plt.show()


def m_complex2real(A):
    return np.vstack(
        (np.concatenate((np.real(A), -np.imag(A)), axis=1), np.concatenate((np.imag(A), np.real(A)), axis=1)))


def v_complex2real(x):
    return np.hstack((np.real(x), np.imag(x)))


def v_real2complex(x):
    return np.split(x, 2)


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