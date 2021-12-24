import sys
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import jittor as jt
import multiprocessing as mp
from cleanfid import fid
import torch
from jittor import nn
import jittor as jt

from eval.ppl import compute_ppl
from training.networks.stylegan2 import Generator


def triangle_distance(pretrained, ours, ref):
    print("name | ours-pretrained | ref-pretrained | ours-ref")
    for key, val_pretrained in pretrained:
        val_ours = ours[key]
        val_ref = ref[key]
        assert val_pretrained.shape == val_ours.shape
        assert val_pretrained.shape == val_ref.shape
        dist1 = (val_ours - val_pretrained).sqr().mean()
        dist2 = (val_ref - val_pretrained).sqr().mean()
        dist3 = (val_ours - val_ref).sqr().mean()
        print(key, dist1, dist2, dist3)


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", type=str)
parser.add_argument("--ckpt", type=str)
parser.add_argument("--ref_ckpt", type=str)
args = parser.parse_args()

triangle_distance(jt.load(args.pretrained), jt.load(args.ckpt), jt.load(args.ref_ckpt))
