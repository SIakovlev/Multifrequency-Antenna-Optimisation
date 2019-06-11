import json
from antenna import Antenna
from antenna_new import Antenna as Antenna_new
import numpy as np

def main(**kwargs):

    antenna_params = kwargs["antenna_params"]
    opt_params = kwargs["optimisation_params"]

    antenna = Antenna(antenna_params)
    # antenna.set_configuration([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    antenna.set_objective(weights=[1.0, 1.0])
    antenna.set_jacobian(weights=[1.0, 1.0])
    antenna.set_hessian(weights=[1.0, 1.0])
    antenna.set_allocation_constraint(7)
    antenna.get_optimal_current_allocation(opt_params, x0=np.ones(200), cons=False, jac=True, hess=True)

    antenna.plot_current_distribution()
    antenna.plot_formed_beams()

    # antenna2 = Antenna_new(antenna_params)
    # antenna2.set_objective()
    # antenna2.get_optimal_current_allocation(opt_params, x0=np.ones(200), cons=False, jac=False, hess=False)
    #
    # antenna2.plot_current_distribution()
    # antenna2.plot_formed_beams()
    # probe = np.ones(20)
    # print(f"Probe signal: \n{probe}\n")
    # print(f"Objective_f: \n{antenna2.objective(probe)}\n")
    # print(f"Objective_grad_f: \n{antenna2.jac(probe)}\n")
    # print(f"Objective_hess_f: \n{antenna2.hess(probe)}\n")

    # print(f"Objective_f: \n{antenna.objective(probe)}\n")
    # print(f"Objective_grad_f: \n{antenna.jac(probe)}\n")

if __name__ == "__main__":

    with open('../config.json') as settings:
        params = json.load(settings)

    main(**params)
