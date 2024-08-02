import torch
import pandas as pd
import pyro
import pyro.distributions as dist
import jax


def load_data():
    # Load the Pima Indians Diabetes dataset
    data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = [
        "num_pregnant",
        "glucose_concentration",
        "blood_pressure",
        "skin_thickness",
        "serum_insulin",
        "bmi",
        "diabetes_pedigree",
        "age",
        "class",
    ]
    data = pd.read_csv(data_url, header=None, names=column_names)

    # Preprocess the data
    X_all = data.drop(columns=["class"]).values
    y_all = data["class"].values

    # Normalize the data
    X_mean = X_all.mean(axis=0)
    X_std = X_all.std(axis=0)
    X_all = (X_all - X_mean) / X_std

    # Convert to torch tensors
    X_all = torch.tensor(X_all, dtype=torch.float)
    y_all = torch.tensor(y_all, dtype=torch.float)
    return X_all, y_all


def load_model(num_data):
    # Define the logistic regression model with pyro
    def model(data):
        X, y = data

        batchsize = X.shape[0]

        # Define the priors
        w = pyro.sample(
            "w",
            dist.Normal(torch.zeros(X.shape[1]), scale=(num_data / batchsize) ** 0.5),
        )  # Scale to ensure the prior variance is 1 for all batch sizes

        # Define the logistic regression model
        logits = torch.matmul(X, w)
        y_pred = torch.sigmoid(logits)

        return pyro.sample("obs", dist.Bernoulli(y_pred), obs=y)

    # Define the log posterior function using Pyro's tracing utilities
    def log_posterior_normalized(params, batch):
        X, y = batch
        batchsize = X.shape[0]
        conditioned_model = pyro.condition(model, data={"w": params})
        model_trace = pyro.poutine.trace(conditioned_model).get_trace((X, y))
        log_joint = model_trace.log_prob_sum()
        return log_joint / batchsize, torch.tensor([])

    return model, log_posterior_normalized


def load_jax_model(num_data):
    def jax_log_posterior_normalized(params, batch):
        X, y = batch
        batch_size = X.shape[0]
        logits = jax.numpy.matmul(X, params)
        y_pred = jax.nn.sigmoid(logits)
        return (
            jax.numpy.sum(jax.numpy.log(y_pred * y + (1 - y_pred) * (1 - y)), axis=0)
            / batch_size
            + jax.scipy.stats.norm.logpdf(params).sum() / num_data
        )

    return jax_log_posterior_normalized
