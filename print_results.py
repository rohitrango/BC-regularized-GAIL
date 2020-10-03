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


parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', required=True,
            help='Get filenames')
args = parser.parse_args()

def main():
    """
    Plot the plots inside the folder given
    """
    res = dict()
    filelist = []
    for dirs in args.files:
        for r, d, f in os.walk(dirs):
            f = list(filter(lambda x: x.endswith('txt'), f))
            f = list(map(lambda x: os.path.join(r, x), f))
            filelist.extend(f)

    for f in filelist:
        key = f.split('/')[-1].split('_')[0]
        if key == 'random':
            key = f.split('_')[0]
        with open(f) as fi:
            rews = fi.read().split('\n')
            rews = filter(lambda x: x != '', rews)
            rews = list(map(lambda x: float(x), rews))
            lis = res.get(key, [])
            lis.extend(rews)
            res[key] = lis

    #for k, v in res.items():
    keys = sorted(list(res.keys()))
    for k in keys:
        v = res[k]
        print('{} {:.2f} \\pm {:.2f}'.format(k, np.mean(v), np.std(v)))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
