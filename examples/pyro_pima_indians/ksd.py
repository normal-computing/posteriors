import torch


def gaussian_kernel(x, y):
    return torch.exp(-0.5 * (x - y).pow(2).sum(-1))


def gaussian_kernel_dx(x, y):
    return (x - y) * gaussian_kernel(x, y)


def gaussian_kernel_dy(x, y):
    return -gaussian_kernel_dx(x, y)


def gaussian_kernel_diag_dxdy(x, y):
    return (1 - (x - y).pow(2)) * gaussian_kernel(x, y)


def ksd(samples, gradients, batchsize=None):
    n = samples.shape[0]
    if batchsize is None:

        def get_batch_inds():
            return torch.arange(n)
    else:

        def get_batch_inds():
            return torch.randint(n, size=(batchsize,))

    def k0(sampsi, sampsj, gradsi, gradsj):
        return (
            torch.sum(gaussian_kernel_diag_dxdy(sampsi, sampsj))
            + torch.dot(gaussian_kernel_dx(sampsi, sampsj), gradsj)
            + torch.dot(gaussian_kernel_dy(sampsi, sampsj), gradsi)
            + torch.dot(gradsi, gradsj) * gaussian_kernel(sampsi, sampsj)
        )

    def v_k_0(sampsi, gradsi):
        batch_inds = get_batch_inds()
        return torch.vmap(k0, in_dims=(None, 0, None, 0))(
            sampsi, samples[batch_inds], gradsi, gradients[batch_inds]
        ).mean()

    return torch.sqrt(
        torch.vmap(v_k_0, randomness="different")(samples, gradients).mean()
    )
