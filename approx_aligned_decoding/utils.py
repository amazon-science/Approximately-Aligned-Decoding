import hashlib
import random

import numpy as np
import torch


def set_seed(obj):
    seed_val = int.from_bytes(hashlib.sha256(str(obj).encode('utf-8')).digest()[:8], "big")
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val % (2 ** 32))
    random.seed(seed_val)
