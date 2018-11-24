import utils
import json
import numpy as np


def main(**kwargs):
    # define the number of antenna elements
    numOfAntennaElements = kwargs["number_of_antenna_elements"]
    # define distance between two consequent elements
    d = kwargs["distance_between_elements"]
    # define two wavelengths (in nm)
    lambda1 = kwargs["lambda_1"]
    lambda2 = kwargs["lambda_2"]
    # define two array factors
    phiRange = np.radians(np.arange(0, 180, 0.1))
    A1 = utils.array_factor(numOfAntennaElements, 2 * np.pi / lambda1, d, phiRange)
    A2 = utils.array_factor(numOfAntennaElements, 2 * np.pi / lambda2, d, phiRange)

    beam1 = utils.ref_beam(numOfAntennaElements, A1)
    beam2 = utils.ref_beam(numOfAntennaElements, A2)

    utils.plotter(phiRange, [beam1, beam2], ['k', 'r'])

    J = utils.get_optimal_current_distr(AFList=[A1, A2], beam_list=[beam1, beam2], weight_list=[1.0, 1.0])

    I1_res, I2_res = np.exp(J)
    utils.plotter(phiRange, [abs(A1 @ I1_res), abs(A2 @ I2_res)], ['k', 'r'])
    utils.plotter([], [abs(I1_res), abs(I2_res)], style_mod=['k', 'r'], plot_type='stem')

if __name__ == "__main__":

    with open('../config.json') as settings:
        params = json.load(settings)

    main(**params)
