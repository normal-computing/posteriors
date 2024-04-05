# API

### Extended Kalman filter (EKF)
- [`ekf.diag_fisher`](ekf/diag_fisher.md) applies a sequential Bayesian update based 
on a Taylor approximation of the log-likelihood. Uses the diagonal empirical Fisher
information matrix as a positive-definite alternative to the Hessian.
Natural gradient descent equivalence following [Ollivier 2019](https://arxiv.org/abs/1901.00696).

### Laplace approximation
- [`laplace.dense_fisher`](laplace/dense_fisher.md) calculates the empirical Fisher
information matrix and uses it to approximate the posterior precision, i.e. a [Laplace
approximation](https://arxiv.org/abs/2106.14806), without modification to parameters.
- [`laplace.diag_fisher`](laplace/diag_fisher.md) same as `laplace.dense_fisher` but
uses the diagonal empirical Fisher information matrix instead.


### Stochastic gradient Markov chain Monte Carlo (SGMCMC)
- [`sgmcmc.sgld`](sgmcmc/sgld.md) implements stochastic gradient Langevin dynamics
(SGLD) from [Welling and Teh](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf).
- [`sgmcmc.sghmc`](sgmcmc/sghmc.md) implements the stochastic gradient Hamiltonian
Monte Carlo (SGHMC) algorithm from [Chen et al](https://arxiv.org/abs/1402.4102)
(without momenta resampling).


### Variational inference (VI)
- [`vi.diag`](vi/diag.md) implements a diagonal Gaussian variational distribution.
Expects a [`torchopt`](https://github.com/metaopt/torchopt) optimizer for handling the
minimization of the NELBO. Also find `vi.diag.nelbo` for simply calculating the NELBO 
with respect to a `log_posterior` and diagonal Gaussian distribution.

### Optim
- [`optim`](optim.md) wrapper for `torch.optim` optimizers within the unified `posteriors` 
API that allows for easy swapping with UQ methods.

### TorchOpt
- [`torchopt`](torchopt.md) wrapper for `torchopt` optimizers within the unified
`posteriors` API that allows for easy swapping with UQ methods.