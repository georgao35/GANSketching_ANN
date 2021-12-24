import sys
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import jittor as jt
import multiprocessing as mp
from cleanfid import fid
import torch

from eval.ppl import compute_ppl
from training.networks.stylegan2 import Generator


def convert_generator(ckpt_path, w_shift=False):
    g = Generator(256, 512, 8, w_shift=w_shift)
    ckpt = jt.load(ckpt_path)
    g.load_state_dict(ckpt)
    torch.save(g.state_dict(to="torch"), ckpt_path[:-4] + ".pth")


if __name__ == "__main__":
    convert_generator(sys.argv[1])
