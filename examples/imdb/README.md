# Cold posterior effect on IMDB data

We investigate a wide range of `posteriors` methods for a [CNN-LSTM](https://proceedings.mlr.press/v119/wenzel20a/wenzel20a.pdf) model on the [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb).
We run each methods for a variety of temperatures, that is targetting the tempered
posterior distribution $p(\theta \mid \mathcal{D})^{\frac{1}{T}}$ for $T \geq 0$, thus
investigating the so-called [cold posterior effect](https://arxiv.org/abs/2008.05912),
where improved predictive performance has been observed for $T<1$.

We observe improved significantly improved performance of Bayesian techniques over
gradient descent and notably that the cold posterior effect is significantly more
prominent for Gaussian approximations than SGMCMC variants.

## Methods

We train the CNN-LSTM model with 2.7m parameters on 25k reviews from the IMDB dataset
for binary classification of positive and negative reviews. We then use `posteriors` to
swap between the following methods:

- **map**: Maximum a posteriori (MAP) estimate of the parameters. Simple optimization 
using [AdamW](https://arxiv.org/abs/1711.05101) via `posteriors.torchopt`.
- **laplace.diag_fisher**: Use the MAP estimate as the basis for a Laplace approximation
using a diagonal empirical Fisher covariance matrix. Additionally use
`posteriors.linearized_forward_diag` to asses a [linearized version](https://arxiv.org/abs/2106.14806)
of the posterior predictive distribution.
- **laplace.diag_ggn**: Same as above but using the diagonal Gauss-Newton Fisher matrix,
which is [equivalent to the conditional Fisher information matrix](https://arxiv.org/abs/1412.1193)
(integrated over labels rather than using empirical distribution). Again we assess
traditional and linearized Laplace.
- **vi.diag**: Variational inference using a diagonal Gaussian approximation. Optimized
using [AdamW](https://arxiv.org/abs/1711.05101) and also linearized variant.
- **sgmcmc.sghmc (Serial)**: [Stochastic gradient Hamiltonian Monte Carlo](https://arxiv.org/abs/1506.04696)
(SGHMC) with a Monte Carlo approximation to the posterior collected by running a single
trajectory (and removing a burn-in).
- **sgmcmc.sghmc (Parallel)**: Same as above but running 15 trajectories in parallel and
only collecting the final state of each trajectory.


All methods are run for a variety of temperatures $T \in \{0.03, 0.1, 0.3, 1.0, 3.0\}$.
Each train is over 30 epochs except serial SGHMC which uses 60 epochs to collect 27
samples from a single trajectory. Each method and temperature is run 5 times with
different random seeds, except for the parallel SGHMC in which we run over 35 seeds and
then bootstrap 5 ensembles of size 15. In all cases we use a diagonal Gaussian prior
with all variances set to 1/40.


## Results

We plot the test loss for each method and temperature in Figure 1.

<p align="center">
  <img src="https://storage.googleapis.com/posteriors/cold_posterior_laplace_loss.png" width=30%">
  <img src="https://storage.googleapis.com/posteriors/cold_posterior_vi_loss.png" width=30%">
  <img src="https://storage.googleapis.com/posteriors/cold_posterior_sghmc_loss.png" width=30%">
  <br>
  <em>Figure 1. Test loss on IMDB data (lower is better) for varying temperatures.</em>
</p>

There are a few takeaways from the results:
- Bayesian methods (VI and SGMCMC) can significantly improve over gradient descent (MAP
and we also trained for the MLE which severely overfits and was omitted from the plot
for clarity).
- Regular Laplace does not perform well, the linearization helps somewhat with GGN out
performing Empirical Fisher.
- In contrast, the linearization is detrimental for VI which we posit is due to the VI
training acting in parameter space without knowledge of linearization.
- A strong cold posterior effect for Gaussian methods (VI + Laplace), only very mild
cold posterior effect for non-Gaussian Bayes methods (SGHMC).
- Parallel SGHMC outperforms Serial SGHMC and also evidence that both out perform [deep
ensemble](https://arxiv.org/abs/1612.01474) (which is obtained with parallel SGHMC and
$T=0$).


## Data

We download the IMDB dataset using [keras.datasets.imdb](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data).


## Model
We use the CNN-LSTM model from [Wenzel et al](https://proceedings.mlr.press/v119/wenzel20a/wenzel20a.pdf)
which consists of an embedding layer, a convolutional layer, ReLU activation, max-pooling layer,
LSTM layer, and a final dense layer. In total there are 2.7m parameters. We use a diagonal
Gaussian prior with all variances set to 1/40.


## Code structure

- `lstm.py`: Simple, custom code for an LSTM layer that composes with `torch.func`.
- `model.py`: Specification of the CNN-LSTM model.
- `data.py`: Functions to load the IMDB data using `keras`.
- `train.py`: General script for training the model using `posteriors` methods 
    which can easily be swapped.
- `configs/` : Configuration files (python files) for the different methods.
- `train_runner.sh`: Bash script to run a series of training jobs.
- `combine_states_serial.py`: Combines parameter states from a single SGHMC trajectory
for ensembled forward calls.
- `combine_states_parallel.py`: Combines final parameter states from a multiple SGHMC
trajectories for ensembled forward calls.
- `test.py`: Calculate metrics on the IMDB test data for a given method.
- `test_runner.sh`: Bash script to run a series of testing jobs.
- `plot.py`: Generate Figure 1.
- `utils.py`: Small utility functions for logging.

Training code for a single train can be run from the root directory:
```bash
PYTHONPATH=. python examples/imdb/train.py --config examples/imdb/configs/laplace_diag_fisher.py --temperature 0.1 --seed 0 --device cuda:0
```
or by configuring multiple runs in `train_runner.sh` and running:
```bash
bash examples/imdb/train_runner.sh
```
Similarly testing code can be run from the root directory:
```bash
PYTHONPATH=. python examples/imdb/test.py --config examples/imdb/configs/laplace_diag_fisher.py --seed 0 --device cuda:0
```
or by configuring settings in `test_runner.sh` and running:
```bash
bash examples/imdb/test_runner.sh
```

