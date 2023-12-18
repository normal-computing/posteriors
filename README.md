# uqlib



General purpose python library for **U**ncertainy **Q**uantification (methods and benchmarks) with [PyTorch](https://github.com/pytorch/pytorch) models.

- All methods should be linear in the number of parameters, and therefore able to handle large models (e.g. transformers).
- We should support uncertainty quantification over subsets of parameters.
- We should support arbitrary loss functions.
- We should support uncertainty over some subset of the parameters - *this will take some thinking about*.
- Bayesian methods should support arbitrary priors (we just need pointwise evaluations).


## Friends

Should interface well with

- Existing optimisers in [torch.optim](https://pytorch.org/docs/stable/optim.html) (we do not need to provide gradient descent)
- [transformers](https://github.com/huggingface/transformers) for fine-tuning pre-trained models (we should make sure our uncertainty methods are also compatible in terms of inference/generation)
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning) for convenient training and logging


## Methods

- [ ] [Dropout](https://arxiv.org/abs/1506.02142)
- [ ] [Variational inference (mean-field and KFAC)](https://arxiv.org/abs/1601.00670)
    - Basic/naive NELBO added but this should be upgraded (to be optimised + KFAC) 
    and tested.
- [ ] [Laplace approximation (mean-field and KFAC)](https://arxiv.org/abs/2106.14806)
    - Currently we have a basic Hessian diagonal implementation but this should be 
    replaced with diagonal (and KFAC) Fisher information which is guaranteed to be positive definite.
- [ ] [Deep Ensemble](https://arxiv.org/abs/1612.01474)
- [ ] [SGMCMC](https://arxiv.org/abs/1506.04696)
    - v0 implementation added but needs API finalising and tests on e.g. linear 
    Gaussian models with known posterior mean + cov.
- [ ] Ensemble SGMCMC
- [ ] [SNGP](https://arxiv.org/abs/2006.10108)
- [ ] [Epistemic neural networks](https://arxiv.org/abs/2107.08924)
<!-- - [ ] [Conformal prediction](https://arxiv.org/abs/2107.07511) -->


## Benchmarks

Benchmarks should extend beyond those in [uncertainty-baselines](https://github.com/google/uncertainty-baselines). We can include classification and regression as toy examples but the leaderboard should consist of the following more practically relevant tasks:

- [ ] Generation
    - Aleatoric vs epistemic uncertainty (e.g. hallucination detection)
- [ ] Continual learning
    - Regression/classification/generation tasks but with a stream of data. Evaluate perfomance on current and historical data/tasks.
- [ ] Decision making
    - Thompson sampling effectiveness


## Contributing

You can report a bug or request a feature by [creating a new issue on GitHub](https://github.com/normal-computing/uqlib/issues).

Pull requests are welcomed! Please go through the following steps:

1. Create a new branch from `main`.
2. Run `pip install -e .` to install the package in editable mode.
3. Add your code and tests (`tests` has the same structure as `uqlib`).
4. Run `pre-commit run --all-files` and `pytest` to check your code lints and tests pass.
5. Commit your changes and push your branch to GitHub.
6. Create pull request into the `main` branch.

Feel free to open a draft PR to discuss changes or get feedback.

