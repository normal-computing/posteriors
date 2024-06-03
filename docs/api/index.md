# API

### Extended Kalman filter (EKF)
- [`ekf.dense_fisher`](ekf/dense_fisher.md) applies an online Bayesian update based 
on a Taylor approximation of the log-likelihood. Uses the empirical Fisher
information matrix as a positive-definite alternative to the Hessian.
Natural gradient descent equivalence following [Ollivier, 2019](https://arxiv.org/abs/1703.00209).
- [`ekf.diag_fisher`](ekf/diag_fisher.md) same as `ekf.dense_fisher` but
uses the diagonal of the empirical Fisher information matrix instead.

### Laplace approximation
- [`laplace.dense_fisher`](laplace/dense_fisher.md) calculates the empirical Fisher
information matrix and uses it to approximate the posterior precision, i.e. a [Laplace
approximation](https://arxiv.org/abs/2106.14806).
- [`laplace.dense_ggn`](laplace/dense_ggn.md) calculates the Generalised
Gauss-Newton matrix which is equivalent to the non-empirical Fisher in most
neural network settings - see [Martens, 2020](https://jmlr.org/papers/volume21/17-678/17-678.pdf).
- [`laplace.diag_fisher`](laplace/diag_fisher.md) same as `laplace.dense_fisher` but
uses the diagonal of the empirical Fisher information matrix instead.
- [`laplace.diag_ggn`](laplace/diag_ggn.md) same as `laplace.dense_ggn` but
uses the diagonal of the Generalised Gauss-Newton matrix instead.

All Laplace transforms leave the parameters unmodified. Comprehensive details on Laplace approximations can be found in [Daxberger et al, 2021](https://arxiv.org/abs/2106.14806).


### Stochastic gradient Markov chain Monte Carlo (SGMCMC)
- [`sgmcmc.sgld`](sgmcmc/sgld.md) implements stochastic gradient Langevin dynamics
(SGLD) from [Welling and Teh, 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf).
- [`sgmcmc.sghmc`](sgmcmc/sghmc.md) implements the stochastic gradient Hamiltonian
Monte Carlo (SGHMC) algorithm from [Chen et al, 2014](https://arxiv.org/abs/1402.4102)
(without momenta resampling).
- [`sgmcmc.sgnht`](sgmcmc/sgnht.md) implements the stochastic gradient Nosé-Hoover
thermostat (SGNHT) algorithm from [Ding et al, 2014](https://proceedings.neurips.cc/paper/2014/file/21fe5b8ba755eeaece7a450849876228-Paper.pdf),
(SGHMC with adaptive friction coefficient).

For an overview and unifying framework for SGMCMC methods, see [Ma et al, 2015](https://arxiv.org/abs/1506.04696).


### Variational inference (VI)
- [`vi.diag`](vi/diag.md) implements a diagonal Gaussian variational distribution.
Expects a [`torchopt`](https://github.com/metaopt/torchopt) optimizer for handling the
minimization of the NELBO. Also find `vi.diag.nelbo` for simply calculating the NELBO 
with respect to a `log_posterior` and diagonal Gaussian distribution.

A review of variational inference can be found in [Blei et al, 2017](https://arxiv.org/abs/1601.00670).


### Optim
- [`optim`](optim.md) wrapper for `torch.optim` optimizers within the unified `posteriors` 
API that allows for easy swapping with UQ methods.

### TorchOpt
- [`torchopt`](torchopt.md) wrapper for [`torchopt`](https://github.com/metaopt/torchopt)
optimizers within the unified `posteriors` API that allows for easy swapping with UQ
methods.

