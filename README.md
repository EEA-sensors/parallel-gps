# parallel-gps

Companion code leveraging [GPflow](https://gpflow.readthedocs.io/en/master/) for the paper Temporal Gaussian Process Regression in Logarithmic Time.

Please cite the following paper [(arXiv preprint)](https://arxiv.org/abs/2102.09964) to use the code

```
@article{corenflos2021,
  title={Temporal {G}aussian Process Regression in Logarithmic Time},
  author={Adrien Corenflos and Zheng Zhao and Simo S\"{a}rkk\"{a}},
  journal={arXiv preprint arXiv:2102.09964},
  year={2021}
}
```

What is it?
-----------

This is an implementation of temporally parallelized and sequential state space Gaussian processes with CPU and GPU support leveraging GPflow as a framework and TensorFlow as a calculation backend.

Supported covariance functions
------------------------------

* Matern 12, 32, 52
* RBF
* Quasi-periodic
* Sum and product of the above

Installation
------------
Clone or download the project
Run `python setup.py [install|develop]` depending on the level of installation you want  


Example
-------

```python
from pssgp.kernels import RBF
from pssgp.model import StateSpaceGP
from gpflow.model import GPR

data = ...
noise_covariance = 1.
lengthscale = 1.
variance = 0.1

order = 6  # Order of the RBF approximation for (P)SSGP, will not be used if the GP model is GPR
balancing_iter = 5  # Number of balancing steps for the resulting SDE to make it more stable, will not be used if the GP model is GPR

cov_function = RBF(variance=variance, lengthscales=lengthscale, order=order, balancing_iter=balancing_iter)

gp = GPR(data=data, kernel=cov, noise_variance=noise_variance)
ssgp = StateSpaceGP(data=data, kernel=cov, noise_variance=noise_variance, parallel=False)

pssgp = StateSpaceGP(data=data, kernel=cov, noise_variance=noise_variance, parallel=True, max_parallel=1000)  
# max_parallel should be bigger than n_training + n_pred

for model in [gp, ssgp, pssgp]:
    print(model.maximum_log_likelihood_objective())

```
For more examples, see the experiments folder which reproduces the results of our paper.
