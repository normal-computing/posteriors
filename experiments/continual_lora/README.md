
## Laplace LoRA for Continual Learning

This experiment demonstrates how simple post-hoc probablisitic strategies can be used to remedy catastophic forgetting in a continual learning setting. In particular, we consider the task of fine-tuning a language model on episodic data. We show that while continued fine-tuning catastrophically forgets all previous tasks, Bayes inspired methods avoid this outcome. 

## Methods 

Instead of simply learning the best point-estimates of the weights each episode, we implement a local Gaussian approximation to the posterior distribution of the model parameters. We first fine-tune the model to arrive at MAP estimates of the parameters, and use the negative inverse Hessian of the loss at the MAP as our covariance (This is the Laplace approximation). Instead of computing the Hessian directly, we use the Fisher information.

The prior for the next episode is the posterior from the previous episode. This looks like a quadratic penalty on the loss function during gradient descent. Whereas the original [paper](https://www.pnas.org/doi/10.1073/pnas.1611835114) suggested using multiple penalties, we use a single penalty following this [note](https://www.inference.vc/comment-on-overcoming-catastrophic-forgetting-in-nns-are-multiple-penalties-needed-2/).

Lastly, rather than sampling from the weight distributions to make predictions, we approximate the predictive distribution under MAP weights using another Taylor expansion.

## Results 

![Validation loss by episode](./pictures/plot_A.png)

Validation loss for each episode, over all four episodes. Vertical lines indicate episode breaks. Probabilistic methods (Laplace) maintain low loss, while SGD forgets immediately. Early stopping is used to determine number of epochs per episode.

![Average validation performance](./pictures/plot_B.png)

Average validation perplexity over all episodes, over training time. 


## Data

We use a subset of [pg19](https://huggingface.co/datasets/pg19), a large corpus of books. The data can easily be downloaded using the `datasets` library from Huggingface.

## Model

We finetune Meta's 7 billion parameter [Llama-2 model](https://huggingface.co/meta-llama/Llama-2-7b-hf), available from the `transformers` library.

## Experiments

We separate our data into `N` episodes of train and test data, and perform the following experiment. 

(Baseline SGD) For each episode: 
- Finetune the model on the `N`th train data
- Validate the model on the `N`th test data, all previous test data.

(Laplace LoRa) For each episode: 
- Finetune the LoRa model on the `N`th train data (using the previous posterior as the prior)
- Validate the model on the `N`th test data, all previous test data. 
- Update the posterior based on new MAP estimates of weights, and the Fisher information.

We finetune the model using LoRA, as implemented in [PEFT](https://github.com/huggingface/peft/tree/main). We use `r=8` and `\alpha=32`. By setting `lambda=0`, we recover the baseline SGD method, so only one script is necessary.

Hyperparameters are set in `config.yaml`. To run the experiment, use the following command: `PYTHONPATH=. python experiments/continual_lora/run_continual_experiment.py --base experiments/continual_lora/configs/lora_laplace.yaml --epochs <epochs> --device <cuda device>`

