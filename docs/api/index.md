# API

### Extended Kalman filter (EKF)
- [`ekf.diag_fisher`](/api/ekf/diag_fisher) applies a sequential Bayesian update based 
on a Taylor approximation of the log-likelihood. Uses the diagonal empirical Fisher
information matrix as a positive-definite alternative to the Hessian.
Natural gradient descent equivalence following [Ollivier 2019](https://arxiv.org/abs/1901.00696).

### Laplace approximation
- [`laplace.dense_fisher`](/api/laplace/dense_fisher) calculates the empirical Fisher
information matrix and uses it to approximate the posterior precision, i.e. a [Laplace
approximation](https://arxiv.org/abs/2106.14806), without modification to parameters.
- [`laplace.diag_fisher`](/api/laplace/diag_fisher) same as `laplace.dense_fisher` but
uses the diagonal empirical Fisher information matrix instead.


### Stochastic gradient Markov chain Monte Carlo (SGMCMC)
- [`sgmcmc.sghmc`](/api/sgmcmc/sghmc) implements the stochastic gradient Hamiltonian
Monte Carlo (SGHMC) algorithm from [Chen et al](https://arxiv.org/abs/1402.4102)
(without momenta resampling).


### Variational inference (VI)
- [`vi.diag`](/api/vi/diag) implements a diagonal Gaussian variational distribution.
Expects a [`torchopt`](https://github.com/metaopt/torchopt) optimizer for handling the
minimization of the NELBO. Also find `vi.diag.nelbo` for simply calculating the NELBO 
with respect to a `log_posterior` and diagonal Gaussian distribution.

### Optim
- [`optim`](/api/optim) wrapper for `torch.optim` optimizers within the unified `posteriors` 
API that allows for easy swapping with UQ methods.

### TorchOpt
- [`torchopt`](/api/torchopt) wrapper for `torchopt` optimizers within the unified
`posteriors` API that allows for easy swapping with UQ methods.