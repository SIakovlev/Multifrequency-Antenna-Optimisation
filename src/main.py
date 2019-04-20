import json
from antenna import Antenna


def main(**kwargs):

    antenna_params = kwargs["antenna_params"]

    antenna = Antenna(antenna_params)
    antenna.set_phase_shift([0.1, -0.3])
    antenna.plot_ref_beams()
    antenna.set_objective(weights=[1.0, 1.0])
    antenna.set_jacobian(weights=[1.0, 1.0])
    antenna.set_hessian(weights=[1.0, 1.0])
    antenna.set_constraints(eps=7)
    antenna.get_optimal_current_allocation(kwargs["optimisation_params"])

    antenna.plot_current_distribution()
    antenna.plot_formed_beams()


if __name__ == "__main__":

    with open('../config.json') as settings:
        params = json.load(settings)

    main(**params)
