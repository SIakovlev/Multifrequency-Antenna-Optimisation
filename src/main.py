import json
from antenna import Antenna
from antenna_new import Antenna as Antenna_new
import numpy as np

def main(**kwargs):

    antenna_params = kwargs["antenna_params"]
    opt_params = kwargs["optimisation_params"]

    antenna = Antenna(antenna_params)
    # antenna.set_configuration([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    # antenna.set_phase_shift([0.2, 0.5])
    antenna.set_objective(weights=[1.0, 1.0])
    antenna.set_jacobian(weights=[1.0, 1.0])
    antenna.set_hessian(weights=[1.0, 1.0])
    antenna.plot_ref_beams()
    probe = None
    # e = 2
    # while e < 9:
    #     print(f" - eps value: {e}")
    #     antenna.set_allocation_constraint(e)
    #     _, probe = antenna.get_optimal_current_allocation(opt_params, x0=probe, cons=True, jac=True, hess=True)
    #     opt_params["options"] = {"maxiter": 5000, "verbose": 1, "initial_tr_radius": 1e6, "initial_barrier_parameter": 1e-6}
    #     # antenna.plot_current_distribution()
    #     # antenna.plot_formed_beams()
    #     e = float(min(abs(antenna.M @ probe.reshape(-1, 1)))) - 0.1
    #     print(f" - Probe: {probe}")
    print(antenna.objective(probe))
    print(np.linalg.matrix_rank(antenna.hess(probe), tol=1e-6))

    antenna.plot_current_distribution()
    antenna.plot_formed_beams()


if __name__ == "__main__":

    with open('../config.json') as settings:
        params = json.load(settings)

    main(**params)
