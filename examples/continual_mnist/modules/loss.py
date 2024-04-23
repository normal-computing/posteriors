import posteriors
from torch import func
import torch.nn.functional as F


def log_posterior(model, num_data):
    def fn_call(params, batch):
        images, labels = batch
        output = func.functional_call(model, params, images)
        log_post_val = (
            -F.cross_entropy(output, labels)
            + posteriors.diag_normal_log_prob(params) / num_data
        )
        return log_post_val, output

    return fn_call
