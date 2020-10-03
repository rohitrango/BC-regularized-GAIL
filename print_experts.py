import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import argparse
import torch
import os
import sys
from baselines.common import plot_util as pu
from baselines.common.plot_util import COLORS
import warnings
import matplotlib as mplot
mplot.rcParams.update({'font.size': 12})

def main():
    """
    Plot the plots inside the folder given
    """
    res = dict()
    filelist = []
    for r, d, f in os.walk('./gail_experts'):
        f = list(filter(lambda x: x.endswith('pt'), f))
        f = list(map(lambda x: os.path.join(r, x), f))
        filelist.extend(f)

    for file in filelist:
        a = torch.load(file)['rewards']
        rew = a.sum(1)
        res[file] = rew

    for k in sorted(res.keys()):
        v = res[k]
        print('{} {:.2f} \\pm {:.2f}'.format(k, v.mean(), v.std()))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
