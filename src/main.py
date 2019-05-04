import json
from antenna import Antenna
import numpy as np

def main(**kwargs):

    antenna_params = kwargs["antenna_params"]
    opt_params = kwargs["optimisation_params"]

    antenna = Antenna(antenna_params)
    antenna.set_configuration((1, 0, 1, 1, 0, 1, 0, 0, 0, 0))
    antenna.set_objective(weights=[1.0, 1.0])
    antenna.set_jacobian(weights=[1.0, 1.0])
    antenna.set_hessian(weights=[1.0, 1.0])
    antenna.get_optimal_current_allocation(opt_params, cons=False, x0=np.ones(antenna.N))

    # antenna.set_allocation_constraint(eps=7)
    # antenna.set_power_constraint(delta=0.5)
    
    antenna.plot_current_distribution()
    antenna.plot_formed_beams()


if __name__ == "__main__":

    with open('../config.json') as settings:
        params = json.load(settings)

    main(**params)
