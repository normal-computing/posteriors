from uqlib import ekf
from uqlib import laplace
from uqlib import sgmcmc
from uqlib import types
from uqlib import optim


from uqlib.utils import model_to_function
from uqlib.utils import linearized_forward_diag
from uqlib.utils import hvp
from uqlib.utils import diag_normal_log_prob
from uqlib.utils import diag_normal_sample
from uqlib.utils import tree_size
from uqlib.utils import tree_extract
from uqlib.utils import tree_insert
from uqlib.utils import tree_insert_
from uqlib.utils import extract_requires_grad
from uqlib.utils import insert_requires_grad
from uqlib.utils import insert_requires_grad_
from uqlib.utils import extract_requires_grad_and_func
from uqlib.utils import inplacify
from uqlib.utils import tree_map_inplacify_
from uqlib.utils import flexi_tree_map
from uqlib.utils import per_samplify


import logging

logger = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
logger.setLevel(logging.ERROR)

from uqlib import vi
from uqlib import torchopt

del logging
del logger
