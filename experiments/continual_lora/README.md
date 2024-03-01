
## Laplace LoRA for Continual Learning

This experiment demonstrates how simple probablisitic strategies can be used to remedy catastophic forgetting in a continual learning setting. In particular, we consider the task of fine-tuning a language model on episodic data. We show that while continued fine-tuning catastrophically forgets all previous tasks, Bayes inspired methods avoid this outcome. We implement our experiment using the built-in functionality of `uqlib`. The api easily integrates into our involved use case.

## Methods 

Instead of simply learning the best point-estimates of the weights each episode, we implement a local Gaussian approximation to the posterior distribution of the model parameters. We first fine-tune the model to arrive at MAP estimates of the parameters, and use the empirical Fisher information at the MAP as our covariance (a [Laplace approximation](https://proceedings.neurips.cc/paper/2021/file/a7c9585703d275249f30a088cebba0ad-Paper.pdf)).

The prior for the next episode is the posterior from the previous episode, becoming a quadratic penalty in the loss function during gradient descent. Whereas the original [paper](https://www.pnas.org/doi/10.1073/pnas.1611835114) suggested using multiple penalties, we use a single penalty following this [note](https://www.inference.vc/comment-on-overcoming-catastrophic-forgetting-in-nns-are-multiple-penalties-needed-2/).

The Laplace updates are implemented in `uqlib` and easily integrated into our PyTorch training loops!

## Results 

<p align="center">
  <img src="https://storage.googleapis.com/normal-blog-artifacts/uqlib/plot_A_uqlib_laplace.png" width=75%">
  <br>
  <em>Figure 1: Validation loss by episode.</em>
</p>

Validation loss for each episode, over all four episodes. Vertical lines indicate episode breaks. Probabilistic methods (Laplace) maintain low loss, while SGD forgets immediately. The dashed line shows an offline train with access to all four training datasets concurrently; it represents the LoRA network's total capacity.


<p align="center">
  <img src="https://storage.googleapis.com/normal-blog-artifacts/uqlib/plot_B_uqlib_laplace.png" width=75%">
  <br>
  <em>Figure 2: Average validation performance.</em>
</p>

Difference of validation loss from baseline, averaged over all episodes seen thus far. 

## Data

We use a subset of [pg19](https://huggingface.co/datasets/pg19), a large corpus of books. The data can easily be downloaded using the `datasets` library from Huggingface. 

## Model

We finetune Meta's 7 billion parameter [Llama-2 model](https://huggingface.co/meta-llama/Llama-2-7b-hf), available from the `transformers` library.

## Experiments

We separate our data into `N` episodes of train and test data, and perform the following experiment. 

(Baseline SGD) For each episode: 
- Finetune the model on the `N`th train data
- Validate the model on the `N`th test data **and** all previous test data.

(Laplace LoRA) For each episode: 
- Finetune the LoRa model on the `N`th train data (using the previous posterior as the prior)
- Validate the model on the `N`th test data **and** all previous test data. 
- Update the posterior based on new MAP estimates of weights, and the latest Fisher information. (Handled by `uqlib`)

We finetune the model using LoRA on the last decoder weights, as implemented in [PEFT](https://github.com/huggingface/peft/tree/main). We use `r=8` and `alpha=32`. By setting the sequential prior scaling parameter `lambda=0`, we recover the baseline SGD method, so only one script is necessary. We stride over the texts so that all tokens have 2048 context tokens.

Hyperparameters for baseline and Laplace methods are set in `configs`. To run the experiment, use the following command: `PYTHONPATH=. python experiments/continual_lora/run_continual_experiment.py --base <path/to/config.yaml> --epochs <epochs> --device <cuda device> [optional]`

## Code structure 

- Download (`download_pg19.py`) and process the data (`load_pg19.py`)
- Download the model and prepare LoRa weights (`load_model.py`)
- Configurations (`configs/`)
- Run the continual experiment (`run_continual_experiment.py`)
- Run static offline baseline that sees all data throughout (`run_static_experiment.py`)
- Plot the results (`plot_metrics.py`)