# `posteriors` examples

This directory contains examples of how to use the `posteriors` package.
- [`continual_lora`](continual_lora/): Uses `posteriors.laplace.diag_fisher` to avoid catastrophic forgetting in fine-tuning [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) on a series of books.
- [`yelp`](yelp/): Compares a host of `posteriors` methods (highlighting the easy swapability) on a sentiment analysis task adapted from the [Hugging Face tutorial](https://huggingface.co/docs/transformers/training#train-in-native-pytorch).
- [`continual_regression`](continual_regression.ipynb): Toy continual learning example with a simple regression task that's easy to visualize.
- [`lightning_autoencoder.py`](lightning_autoencoder.py): Easily adapt the autoencoder example from the [Lightning tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) to use UQ methods with `posteriors` and logging + device handling with `lightning`.

