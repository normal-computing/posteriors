# Constructing Log Posteriors

!!! abstract "TL;DR"
    - `posteriors` enforces `log_posterior` or `log_likelihood` functions to have a
    `log_posterior(params, batch) -> log_prob, aux` signature, where the second element
    is a tensor valued `PyTree` containing any auxiliary information.
    - Define your `log_posterior` or `log_likelihood` to be averaged across the batch.
    - Set `temperature=1/num_data` for Bayesian methods such as 
    [`posteriors.sgmcmc.sghmc`](/api/sgmcmc/sghmc/) and [`posteriors.vi.diag`](/api/vi/diag/).
    - This ensures that hyperparameters such as learning rate are consistent across 
    batchsizes.


## Auxiliary information

Model calls can be expensive, and they might provide more information than just an output
value (and gradient). In order to avoid, losing this information `posteriors` enforces the
`log_posterior` or `log_likelihood` functions to have a
`log_posterior(params, batch) -> log_prob, aux` signature, where the second element
contains any auxiliary information, such as 
predictions or alternative metrics.

`posteriors` algorithms will store this information in `state.aux`.



## Gradient Ascent

Normally in gradient descent we minimize a loss function such as cross-entropy or 
squared error. This is equivalent to gradient ascent to maximise a log likelihood 
function e.g. cross-entropy  loss corresponds to the log likelihood of a categorical 
distribution:

\begin{aligned}
\log p(y_{1:N} \mid x_{1:N}, \theta) &= \sum_{i=1}^N \log p(y_{i} \mid x_i, \theta) \\
p(y_{i} \mid x_i, \theta) &= \text{Categorical}(y_i \mid f_\theta(x_i)) \\
\log p(y_{i} \mid x_i, \theta) &=  \sum_{k=1}^K \mathbb{I}[y_i = k] [\log f_\theta(x_i)]_k
\end{aligned}

Here $K$ is the number of classes and $\log f_\theta(x_i)$ is a vector (length  $K$) of 
logits from the model (i.e. neural network) for input $x_i$ and parameters $\theta$.


!!! example "In code"
    ```py
    import torch.nn.functional as F
    mean_log_lik = - F.cross_entropy(logits, labels.squeeze(-1))
    ```
    or equivalently
    ```py
    from torch.distributions import Categorical
    mean_log_lik = Categorical(logits=logits, validate_args=False).log_prob(labels).mean()
    ```

## Going Bayesian

To do Bayesian inference on the parameters $\theta$ we look to approximate the posterior
distribution

\begin{aligned}
p(\theta \mid y_{1:N}, x_{1:N}) &= \frac{
p(\theta) p(y_{1:N} \mid x_{1:N}, \theta)
}{Z} =  \frac{
p(\theta) \prod_{i=1}^N p(y_{i} \mid x_{i}, \theta)
}{Z} \\
\log p(\theta \mid y_{1:N}, x_{1:N}) &= \log p(\theta) + \sum_{i=1}^N \log p(y_{i} \mid x_{i}, \theta) - \log Z
\end{aligned}

Here $p(\theta)$ is some prior of the parameter which we have to define, $Z$ is a 
normalizing constant which is independent of $\theta$ and therefore we can ignore it 
(it disappears when we take the gradient).

Again we have to take minibatches giving us the stochastic log posterior
\begin{aligned}
\log p(\theta \mid y_{1:n}, x_{1:n}) = \log p(\theta) +  \frac{N}{n} \sum_{i=1}^n \log p(y_{i} \mid x_i, \theta)
\end{aligned}

But the problem here is that the value of $\log p(\theta \mid y_{1:n}, x_{1:n})$ 
will be very large in the realistic case when $N$ is very large. Instead we
should consider the averaged stochastic log posterior which remains on the same scale as
either $N$ or $n$ increaase.

\begin{aligned}
\frac{1}{N} \log p(\theta \mid y_{1:n}, x_{1:n}) = \frac1N \log p(\theta) + 
\frac{1}{n} \sum_{i=1}^n \log p(y_{i} \mid x_i, \theta)
\end{aligned}

!!! example "In code"
    ```py
    import posteriors, torch
    from optree import tree_map, tree_reduce
    from torch.distributions import Categorical

    model_function = posteriors.model_to_function(model)

    def log_posterior(params, batch):
        logits = model_function(params, **batch)
        log_prior = diag_normal_log_prob(params, sd=1., normalize=False)
        mean_log_lik = Categorical(logits=logits).log_prob(batch['labels']).mean()
        mean_log_post = log_prior / num_data + mean_log_lik
        return mean_log_post
    ```

The issue with running Bayesian methods (such as VI or SGHMC) on this mean log posterior
function is that naive application will result in approximating the tempered posterior

\begin{aligned}
p(\theta \mid y_{1:N}, x_{1:N})^{\frac1N} &= \frac{
p(\theta)^{\frac1N} p(y_{1:N} \mid x_{1:N}, \theta)^{\frac1N}
}{Z}
\end{aligned}
(a tempered distribution is $q(x, T) := p(x)^{\frac1T}/Z$ for temperature $T$).

This tempered posterior is much less concentrated than the true posterior 
$p(\theta \mid y_{1:N}, x_{1:N})$. To correct for this we can either supply our Bayesian
inference algorithm with:
```py
temperature=1/num_data
```
Note that with this support, optimization can often be obtained by simply setting
`temperature=0`.

!!! example
    ```py
    import torchopt
    # Define log_posterior as above
    # Load dataloader
    num_data = len(dataloader.dataset)

    vi_transform = posteriors.vi.diag.build(
        log_posterior=log_posterior,
        optimizer = torchopt.adam(lr=1e-3),
        temperature=1/num_data
    )
    
    vi_state = vi_transform.init(params)

    for batch in dataloader:
        vi_state = vi_transform.update(vi_state, batch)
    ```

Alternatively, we can rescale the log posterior  `log_post = mean_log_post * num_data`
but this may not scale well as `log_post` values become extremely large resulting
in e.g. the need for an extremely small learning rate.


## Prior Hyperparameters
Observe the mean log posterior function
\begin{aligned}
\frac{1}{N} \log p(\theta \mid y_{1:n}, x_{1:n}) = \frac1N \log p(\theta) +  \frac{1}{n} \sum_{i=1}^n \log p(y_{i} \mid x_i, \theta)
\end{aligned}

Typically the prior $p(\theta)$ will have some scale hyperparameter $\sigma^2$:
$$
p(\theta) = e^{\frac{1}{\sigma^2}\gamma(\theta)} / Z(\sigma^2)
$$
(such as a normal distribution). The mean log posterior becomes
\begin{aligned}
\frac{1}{N} \log p(\theta \mid y_{1:n}, x_{1:n}) = \frac{1}{N\sigma^2} \log \gamma(\theta) +  \frac{1}{n} \sum_{i=1}^n \log p(y_{i} \mid x_i, \theta)
\end{aligned}
We are free to choose $\sigma^2$ and indeed it controls the strength of the prior vs the
likelihood. In most cases we probably want the prior to be quite weak and therefore the 
variance $\sigma^2$ quite large. As we can see if $\sigma^2$ is large then the prior 
term becomes very small. We can ignore the normalising constant $Z(\sigma^2)$ because it
does not depend on $\theta$, in fact this often recomended to keep the `log_posterior` 
values on a nice scale comparable to loss functions we are accustomed to, this can be 
achieved for a normal prior with [`posteriors.diag_normal_log_prob(x, normalize=False)`](/api/utils/#posteriors.utils.diag_normal_log_prob).
