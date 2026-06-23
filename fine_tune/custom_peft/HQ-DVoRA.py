"""
Experimental Hybrid-Quantum DVoRA
"""

import torch
import math
import logging
import pennylane as qml
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)