from posteriors import ekf
from posteriors import laplace
from posteriors import sgmcmc
from posteriors import types
from posteriors import optim

from posteriors.utils import CatchAuxError
from posteriors.utils import model_to_function
from posteriors.utils import linearized_forward_diag
from posteriors.utils import hvp
from posteriors.utils import fvp
from posteriors.utils import empirical_fisher
from posteriors.utils import diag_normal_log_prob
from posteriors.utils import diag_normal_sample
from posteriors.utils import tree_size
from posteriors.utils import tree_extract
from posteriors.utils import tree_insert
from posteriors.utils import tree_insert_
from posteriors.utils import extract_requires_grad
from posteriors.utils import insert_requires_grad
from posteriors.utils import insert_requires_grad_
from posteriors.utils import extract_requires_grad_and_func
from posteriors.utils import inplacify
from posteriors.utils import tree_map_inplacify_
from posteriors.utils import flexi_tree_map
from posteriors.utils import per_samplify
from posteriors.utils import is_scalar

import logging

logger = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
logger.setLevel(logging.ERROR)

from posteriors import vi
from posteriors import torchopt

del logging
del logger
