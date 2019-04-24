# Multifrequency-Antenna-Optimisation



Code example:

1) Read `.json` config file:
```python
    with open('config.json') as settings:
      params = json.load(settings)
```
2) Create antenna, set it, run the optimisation solver and plot the results:
```python
    antenna_params = params["antenna_params"]
    
    # create antenna object with given parameters
    antenna = Antenna(antenna_params)
    
    # set objective function, jacobian and hessian
    antenna.set_objective(weights=[1.0, 1.0])
    antenna.set_jacobian(weights=[1.0, 1.0])
    antenna.set_hessian(weights=[1.0, 1.0])
  
    # set allocation constraints and current limits
    antenna.set_allocation_constraint(eps=7)
    antenna.set_power_constraint(delta=0.5)
    
    # perform optimisation
    antenna.get_optimal_current_allocation(kwargs["optimisation_params"])
    
    # plot the results
    antenna.plot_current_distribution()
    antenna.plot_formed_beams()

```
