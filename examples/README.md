# `posteriors` examples

This directory contains examples of how to use the `posteriors` package.
- [`continual_lora`](continual_lora/): Uses `posteriors.laplace.diag_fisher` to avoid
catastrophic forgetting in fine-tuning [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)
on a series of books from the [pg19](https://huggingface.co/datasets/pg19) dataset.
- [`yelp`](yelp/): Compares a host of `posteriors` methods (highlighting the easy
exchangeability) on a sentiment analysis task adapted from the [Hugging Face tutorial](https://huggingface.co/docs/transformers/training#train-in-native-pytorch).
- [`continual_regression`](continual_regression.ipynb): [Variational continual learning](https://arxiv.org/abs/1710.10628)
notebook for a simple regression task that's easy to visualize.
- [`lightning_autoencoder.py`](lightning_autoencoder.py): Easily adapt the autoencoder
example from the [Lightning tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)
to use UQ methods with `posteriors` and logging + device handling with `lightning`.


Further information is available within the specific example directories or files.

