import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def forward_multiple(model, parameter_vectors: torch.tensor, X) -> torch.tensor:
    """Evaluates multiple forward passes of a model with different parameter vectors.

    Does not use torch.inference_mode() by default
    (although this should be considered in most cases).

    Does not squeeze output, output is guranateed to be 3D
    (even if only one parameter vector or one input is passed).

    Args:
        model: torch.nn.Module
        parameter_vectors: torch.tensor
            Shape: (n_parameter_vectors, dim_parameter)
        X: torch.tensor
            Shape: (n_samples, dim_input)

    Returns:
        torch.tensor
            Shape: (n_samples, n_parameter_vectors, dim_output)
    """
    parameter_vectors = torch.atleast_2d(parameter_vectors).to(model.device)

    # This assumes that X is a tensor, is this a fair assumption?
    X = torch.atleast_2d(X).to(model.device)

    fs = list()

    orig_params = parameters_to_vector(model.parameters())

    for vec in parameter_vectors:
        vector_to_parameters(vec, model.parameters())
        fs.append(model(X))

    vector_to_parameters(orig_params, model.parameters())
    return torch.stack(fs).transpose(0, 1)
