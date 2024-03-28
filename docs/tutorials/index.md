# Tutorials

We list here a view small examples, that are easy to run locally on a CPU, 
to show how to use `posteriors`:


- [Visualizing VI and SGHMC](visualizing_vi_sghmc.md)
[<img style="float: right; width: 5em;" src="https://storage.googleapis.com/posteriors/double_well_compare.png">](visualizing_vi_sghmc.md)
<br><br><br>
- [EKF for Premier League football](ekf_premier_league.md)
[<img style="float: right; width: 5em;" src="https://em-content.zobj.net/source/telegram/386/soccer-ball_26bd.webp">](ekf_premier_league.md)
<br><br><br>
- [Autoencoder with Lightning](lightning_autoencoder.md)
[<img style="float: right; width: 7em" src="https://storage.googleapis.com/posteriors/lightning_posteriors.png">](lightning_autoencoder.md)




Additionally, we provide more in depth examples in the [examples directory on GitHub](https://github.com/normal-computing/posteriors/tree/main/examples),
many of which use larger models that require a GPU to run:

[<img style="float: right; width: 6em" src="https://storage.googleapis.com/posteriors/plot_B_laplace.png">](https://github.com/normal-computing/posteriors/tree/main/examples/continual_lora)

- [`continual_lora`](https://github.com/normal-computing/posteriors/tree/main/examples/continual_lora):
Uses [`posteriors.laplace.diag_fisher`](/api/laplace/diag_fisher) to avoid catastrophic forgetting in fine-tuning
[Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)
on a series of books from the [pg19](https://huggingface.co/datasets/pg19) dataset.

[<img style="float: right; width: 6em" src="https://storage.googleapis.com/posteriors/yelp_spanish_uncertainty.png">](https://github.com/normal-computing/posteriors/tree/main/examples/yelp)

- [`yelp`](https://github.com/normal-computing/posteriors/tree/main/examples/yelp):
Compares a host of `posteriors` methods (highlighting the easy exchangeability) on a
sentiment analysis task adapted from the [Hugging Face tutorial](https://huggingface.co/docs/transformers/training#train-in-native-pytorch).

[<img style="float: right; width: 6em" src="https://storage.googleapis.com/posteriors/variational_continual_learning.png">](https://github.com/normal-computing/posteriors/blob/main/examples/continual_regression.ipynb)

- [`continual_regression`](https://github.com/normal-computing/posteriors/blob/main/examples/continual_regression.ipynb):
[Variational continual learning](https://arxiv.org/abs/1710.10628) notebook for a simple
regression task that's easy to visualize.

<!-- [<img style="float: right; width: 7em" src="https://storage.googleapis.com/posteriors/lightning_posteriors.png">](https://github.com/normal-computing/posteriors/blob/main/examples/lightning_autoencoder.py)

- [`lightning_autoencoder.py`](https://github.com/normal-computing/posteriors/blob/main/examples/lightning_autoencoder.py):
Easily adapt the autoencoder example from the [Lightning tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)
to use UQ methods with `posteriors` and logging + device handling with `lightning`.
 -->


