# `posteriors` examples

This directory contains examples of how to use the `posteriors` package.
- [`bayes_llama3`](bayes_llama3/): Uses `posteriors` and `lightning` to train a bayesian ensemble language model [Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
on the [TQA](https://allenai.org/data/tqa) dataset.
- [`continual_lora`](continual_lora/): Uses `posteriors.laplace.diag_fisher` to avoid
catastrophic forgetting in fine-tuning [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)
on a series of books from the [pg19](https://huggingface.co/datasets/pg19) dataset.
- [`imdb`](imdb/): Investigates [cold posterior effect](https://proceedings.mlr.press/v119/wenzel20a/wenzel20a.pdf)
for a range of approximate Bayesian methods on [IMDB](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data)
data.
- [`pyro_pima_indians`](pyro_pima_indians/): Uses `pyro` to define a Bayesian logistic
regression model for the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database).
Compares `posteriors` methods against `pyro` and `blackjax`.
See [`pyro_pima_indians_vi.ipynb`](pyro_pima_indians_vi.ipynb) for a more accessible lightweight notebook.
- [`yelp`](yelp/): Compares a host of `posteriors` methods (highlighting the easy
exchangeability) on a sentiment analysis task adapted from the [Hugging Face tutorial](https://huggingface.co/docs/transformers/training#train-in-native-pytorch).
- [`continual_regression`](continual_regression.ipynb): [Variational continual learning](https://arxiv.org/abs/1710.10628)
notebook for a simple regression task that's easy to visualize.
- [`double_well.py`](double_well.py): Compare VI and SGHMC on a two dimensional
multi-modal distribution.
- [`ekf_premier_league.py`](ekf_premier_league.py): Use an Extended Kalman Filter
to infer the skills of Premier League football teams.
- [`lightning_autoencoder.py`](lightning_autoencoder.py): Easily adapt the autoencoder
example from the [Lightning tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)
to use UQ methods with `posteriors` and logging + device handling with `lightning`.
- [`pyro_pima_indians_sghmc.ipynb`](pyro_pima_indians_sghmc.ipynb): A more accessible
notebook demonstrating the use of `posteriors` with a `pyro` defined Bayesian logistic regression model.
As well as convergence diagnostics from `pyro`.


Further information is available within the specific example directories or files.

