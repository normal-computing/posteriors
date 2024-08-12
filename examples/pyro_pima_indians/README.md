# Using `posteriors` with `pyro` #

In this example, we show how to use `posteriors` with [`pyro`](https://pyro.ai/) to define a Bayesian logistic regression model for the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database). The model can then be used to automatically generate
a `log_posterior` function that can be directly passed to `posteriors`.

This specific model is small with 8 dimensions and 768 data points.


## Results


<p align="center">
    <img src="https://storage.googleapis.com/posteriors/pima_indians_full_batch_marginals.png" width=60%">
    <img src="https://storage.googleapis.com/posteriors/pima_indians_mini_batch_marginals.png" width=60%">
    <br>
    <em>Marginal posterior densities for a variety of methods and package
    implementations. Top: Full batch, bottom: mini-batched.</em>
</p>


In the above figure, we show the marginal posterior densities for a variety of methods and package implementations. We observe a broad agreement between approximations indicating all methods have converged, taking Pyro's [NUTS](http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf) implementation as a gold standard.


<p align="center">
    <img src="https://storage.googleapis.com/posteriors/pima_indians_metrics.png" width=45%">
    <br>
    <em>Kernelized Stein discrepancy (KSD) measures the distance between the samples provided by the algorithm and the true posterior via a kernel function (in this case a standard Gaussian). All results are averaged over 10 random seeds with one standard deviation displayed. †The displayed parallel SGHMC time represents the time for a single chain that could be obtained with sufficient parallel resources.</em>
</p>

In the table above we compare kernelized Stein discrepancies between methods as a quantitative
measure of the distance between the collected samples and the true posterior, confirming the
qualitative observations from the marginal posterior densities that `posteriors` methods are
competitive and suitably converged. For this small scale example the Python overheads are
significant, as demonstrated for the only minor speedup due to minibatching
(and overheads are less so for JAX in this setting although in our experience this
rapidly deteriorates for larger models).

## Code Structure

- [`model.py`](model.py): Defines the Bayesian logistic regression model using `pyro`
and loads the required functions for inference in `torch` or `jax`.
- The run files are end-to-end implementations (aside from the model loading) for the
labeled packages and methods.
- [`plot_marginals.py`](plot_marginals.py): Plots the marginal posterior densities in 
the figure above.
- [`calculate_metrics.py`](calculate_metrics.py): Calculates the metrics in the table
above using the functions in [`ksd.py`](ksd.py) to calculate the kernelized Stein
discrepancies.

