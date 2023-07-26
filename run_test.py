# import numpy as np
import copy
import random
import json
import math
import itertools

import logging
# logger = logging.get_logger(__name__)

from utils.config import Config

# from tools.videocomposer.test_unet_temporal1 import *
from tools.videocomposer.test_unet_temporal_profile import *

if __name__ == '__main__':
    cfg = Config(load=True)
    inference_single(cfg.cfg_dict)
    print('run test!')