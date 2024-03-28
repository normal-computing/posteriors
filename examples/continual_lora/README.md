
## Laplace LoRA for Continual Learning

The purpose of this benchmark is to investigate the effects of catastrophic forgetting 
in a language model fine-tuning task, where data is continuously received in an online 
setting. Continuing to apply gradient descent as new data arrives 
will result in catastrophic forgetting, where the model's performance on previous tasks 
deteriorates as it learns new ones. Exact Bayesian updating would avoid this issue, but is 
intractable. Instead, we investigate how an approximate Bayesian approach from 
`posteriors` can be successfully used to mitigate catastrophic forgetting for language models.

We observe that a applying diagonal Laplace approximation (details below) allows the model 
to retain information from previous tasks, and maintain low loss on all tasks. This is 
in contrast to the continual gradient descent baseline, which quickly forgets previous tasks.

The Laplace approximation, however, does not provide a silver bullet, and the model 
still forgets some information from previous tasks and somewhat prohibits learning of 
new tasks. This is a transparent trade-off dictated by the Bayesian forgetting 
parameter. The Laplace approximation also assumes that data is arriving in episodes 
(i.e. there are strict pre-defined boundaries between tasks). In practice the data may 
be arriving continuously and still exhibiting distribution shift which we'd like the 
model to adapt to without forgetting.


## Methods 

The baseline method for our continual learning experiment is to fine-tune the model on the new data each episode. This is a standard approach, but it will result in catastrophic forgetting.

Instead, we can approximate Bayesian updates at each episode, converting the posterior 
from the previous episode $p(\theta | D_{1:n-1})$ after receiving the dataset 
from the $n$ th episode to give a new posterior $p(\theta | D_{1:n}) \propto p(\theta | D_{1:n-1})p(D_n | \theta)$, for language model parameters $\theta$. The true Bayesian 
posterior will be highly complex and intractable. The Laplace approximation approximates the posterior as a Gaussian distribution $p(\theta | D_{1:n}) \approx N(\theta | \mu_n, F^{-1}_n)$, with the mean, $\mu_n$, at the maximum a posteriori (MAP) estimate of the parameters and the covariance as the inverse of the empirical Fisher information, $F^{-1}_n$, which we take to be diagonal. Extensive details can be found in the [Laplace Redux paper](https://proceedings.neurips.cc/paper/2021/file/a7c9585703d275249f30a088cebba0ad-Paper.pdf). The Laplace updates are implemented in `posteriors` and are easily integrated into our PyTorch training loops!

The prior for the next episode is the posterior from the previous episode, becoming a quadratic penalty in the loss function during gradient descent. Whereas the original [catastrophic forgetting paper](https://www.pnas.org/doi/10.1073/pnas.1611835114) suggested using multiple penalties (incorporating data from all previous episodes into the prior), we use a single penalty following this [note](https://www.inference.vc/comment-on-overcoming-catastrophic-forgetting-in-nns-are-multiple-penalties-needed-2/) (we use only the last epsiode's posterior, reminiscent of exact Bayesian updates).


## Data

We use a subset of [pg19](https://huggingface.co/datasets/pg19), a large corpus of books. The data can easily be downloaded using the `datasets` library from Hugging Face. An episode is represented by a single book, and we hold out the last 15% of the book for testing.


## examples

This experiment demonstrates the added control probabilistic methods provide in a setting where continued training is required. **In particular, we demonstrate the benefits of controlling the trade-off between learning new tasks and retaining old ones.** Our dataset (a subset of [pg19](https://huggingface.co/datasets/pg19)) is divided into `N` "episodes" of train and test data. In the results reported below, we use 1 book per episode, holding out the last 15% for testing. For each episode of data, we perform the following:

(Baseline SGD) 
- Fine-tune the model on the `N`th train data
- Validate the model on the `N`th test data **and** all previous test data.

(Laplace LoRA)
- Fine-tune the LoRa model on the `N`th train data (using the previous posterior as the prior)
- Validate the model on the `N`th test data **and** all previous test data. 
- Update the posterior based on new MAP estimates of weights, and the latest Fisher information. (Handled by `posteriors`)

We fine-tune the model using LoRA on the last decoder weights, as implemented in [PEFT](https://github.com/huggingface/peft/tree/main). We use `r=8` and `alpha=32`. By setting the sequential prior scaling parameter `lambda=0`, we recover the baseline SGD method, so only one script is necessary. We stride over the texts so that all tokens have 2048 context tokens.

Hyperparameters for baseline and Laplace methods are set in `configs`. To run the experiment, use the following command: `PYTHONPATH=. python examples/continual_lora/run_continual_experiment.py --base <path/to/config.yaml> --epochs <epochs> --device <cuda device> [optional]` from the root directory.

We also report results on a static offline baseline that sees all data every episode. This represents the LoRA network's total capacity, but is computationally infeasible in practice as it requires all data to available at all times.

## Results 

<p align="center">
  <img src="https://storage.googleapis.com/posteriors/plot_A_laplace.png" width=75%">
  <br>
  <em>Figure 1: Validation loss by episode.</em>
</p>

Validation loss for each episode, over all four episodes. Vertical lines indicate episode breaks. Probabilistic methods (Laplace) maintain low loss in early tasks, whilst SGD forgets. For example, in the top row, the Laplace approximation stays low throughout training demonstrating that it continues to perform well on task 0 even though it is now being trained on data from tasks 1-3. In contrast, continuing applying to gradient descent quickly decreases the performance of the model on task 0. The horizontal dashed lines show an offline train with access to all four training datasets concurrently; the LoRA network's total learning capacity - although this is not feasible in a practical online setting as the number of books increase.


<p align="center">
  <img src="https://storage.googleapis.com/posteriors/plot_B_laplace.png" width=75%">
  <br>
  <em>Figure 2: Average validation performance.</em>
</p>

Difference of validation loss from baseline, averaged over all episodes seen thus far. Where we clearly see the Laplace approximation 
maintains low loss averaged over all tasks.


## Model

We fine-tune Meta's 7 billion parameter [Llama-2 model](https://huggingface.co/meta-llama/Llama-2-7b-hf) (available from the `transformers` library), combined with last-layer LoRA (via the [PEFT](https://github.com/huggingface/peft/tree/main) library).

## Code structure 

- Download (`download_pg19.py`) and process the data (`load_pg19.py`)
- Download the model and prepare LoRa weights (`load_model.py`)
- Configurations (`configs/`)
- Run the continual experiment (`run_continual_experiment.py`)
- Run static offline baseline that sees all data throughout (`run_static_experiment.py`)
- Plot the results (`plot_metrics.py`)
