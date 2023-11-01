# uqlib

**Goal**: General purpose python library for **U**ncertainy **Q**uantification (methods and benchmarks) with [PyTorch](https://github.com/pytorch/pytorch) models.

- All methods should be linear in the number of parameters, and therefore able to handle large models (e.g. transformers).
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
- [ ] [Laplace approximation (mean-field and KFAC)](https://arxiv.org/abs/2106.14806)
- [ ] [Deep Ensemble](https://arxiv.org/abs/1612.01474)
- [ ] [SGMCMC](https://arxiv.org/abs/1506.04696)
- [ ] Ensemble SGMCMC
- [ ] [SNGP](https://arxiv.org/abs/2006.10108)
- [ ] [Epistemic neural networks](https://arxiv.org/abs/2107.08924)
- [ ] [Conformal prediction](https://arxiv.org/abs/2107.07511)


