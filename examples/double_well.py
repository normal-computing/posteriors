import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torchopt
import uqlib

torch.manual_seed(42)


# 2D multi_modal log posterior function (double well)
def log_posterior(x, batch):
    log_prob = -torch.sum(x**4, axis=-1) / 10.0 + torch.sum(x**2, axis=-1)
    return log_prob, torch.tensor([])


# Variational inference
vi_transform = uqlib.vi.diag.build(
    log_posterior, optimizer=torchopt.adam(lr=1e-2), init_log_sds=-2.0
)
n_vi_steps = 2000
vi_state = vi_transform.init(torch.zeros(2))

nelbos = []
for _ in range(n_vi_steps):
    vi_state = vi_transform.update(vi_state, None)
    nelbos.append(vi_state.nelbo.item())

# Plot the negative ELBO
plt.plot(nelbos)
plt.ylabel("NELBO")
plt.tight_layout()


# SGHMC
sghmc_transform = uqlib.sgmcmc.sghmc.build(log_posterior, lr=5e-2, alpha=1.0)
n_sghmc_steps = 10000
sghmc_state = sghmc_transform.init(torch.zeros(2))

samples = torch.zeros(1, 2)
log_posts = []
for _ in range(n_sghmc_steps):
    sghmc_state = sghmc_transform.update(sghmc_state, None)
    samples = torch.cat([samples, sghmc_state.params.unsqueeze(0)], axis=0)
    log_posts.append(sghmc_state.log_posterior.item())

# Plot SGHMC log posterior values
plt.plot(log_posts)
plt.ylabel("SGHMC Log Posterior")
plt.tight_layout()

# Plot the 2D multi-modal log posterior function
lim = 4
x = torch.linspace(-lim, lim, 1000)
X, Y = torch.meshgrid(x, x)
Z = torch.vmap(log_posterior, in_dims=(0, None))(torch.stack([X, Y], axis=-1), None)[0]
plt.contourf(X, Y, Z, levels=50, cmap="Purples", alpha=0.5, zorder=-1)

# Plot VI Gaussian and SGHMC samples
mean = vi_state.params
sd_diag = torch.exp(vi_state.log_sd_diag)
Z_gauss = torch.vmap(
    lambda z: -torch.sum(torch.square((z - mean) / sd_diag), axis=-1) / 2.0,
)(torch.stack([X, Y], axis=-1))

plt.contour(X, Y, Z_gauss, levels=5, colors="black", alpha=0.5)
sghmc_samps = plt.scatter(
    samples[:, 0], samples[:, 1], c="r", s=0.5, alpha=0.5, label="SGHMC Samples"
)

vi_legend_line = mlines.Line2D(
    [], [], color="black", label="VI", alpha=0.5, linestyle="--"
)
plt.legend(handles=[vi_legend_line, sghmc_samps])
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.tight_layout()
