!!! abstract "TL;DR"
    - Define your `log_posterior` or `log_likelihood` to be averaged across the batch.
    - Set `temperature=1/num_data` for Bayesian methods such as 
    [`uqlib.sgmcmc.sghmc`](/api/sgmcmc/sghmc/) and [`uqlib.vi.diag`](/api/vi/diag/).
    - This ensures that hyperparameters such as learning rate are consistent across 
    batchsizes.



## Gradient Ascent

Normally in gradient descent we minimize a loss function such as cross-entropy or 
squared error. This is equivalent to gradient ascent to maximise a log likelihood 
function. In `uqlib` we prefer the probabilistic interpretation, e.g. cross-entropy 
loss corresponds to the log likelihood of a categorical distribution:

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
    mean_log_lik = Categorical(logits=logits).log_prob(labels).mean()
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

But the problem here is that this number will be very large in the realistic case $N$ 
is very large. Thus instead we consider the mean stochastic log posterior which is 
stable as either $N$ or $n$ grow really large.

\begin{aligned}
\frac{1}{N} \log p(\theta \mid y_{1:n}, x_{1:n}) = \frac1N \log p(\theta) + 
\frac{1}{n} \sum_{i=1}^n \log p(y_{i} \mid x_i, \theta)
\end{aligned}

!!! example "In code"
    ```py
    import uqlib, torch
    from optree import tree_map, tree_reduce
    from torch.distributions import Categorical

    log_prior = diag_normal_log_prob(params, sd=1., normalize=False)
    mean_log_lik = Categorical(logits=logits).log_prob(labels).mean()
    mean_log_post = log_prior / num_data + mean_log_lik
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
inference algorithm with 
```py
temperature=1/num_data
```
in the case that it supports this argument (which is how we should approach it imo, 
we should strive to make all code have a temperature argument at the highest level and 
handle very small temperatures in a stable way). Note that with this support, the user 
can also do MAP optimisation simply by setting temperature = 0.

Or we do a simple adjustment to rescale the log posterior 
`log_post = mean_log_post * num_data` but this might not scale well as `log_post` values 
could be extremely large and the user might have to use an extremely small learning rate.


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
achieved for a normal prior with [`uqlib.diag_normal_log_prob(x, normalize=False)`](/api/utils/#uqlib.utils.diag_normal_log_prob).
