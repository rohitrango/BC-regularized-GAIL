import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse
import torch
import os
import sys
from baselines.common import plot_util as pu
from baselines.common.plot_util import COLORS
import warnings
import matplotlib as mplot
mplot.rcParams.update({'font.size': 15})

parser = argparse.ArgumentParser()
parser.add_argument('--dirs', type=str, required=True,
            help='Get directories')
args = parser.parse_args()


def main():
    """
    Plot the plots inside the folder given
    """
    # Now plot the common things
    print(args.dirs)
    envname  = args.dirs.split('/')[-1].split('*')[0]
    dirs = glob.glob(args.dirs)
    allfiles = []
    for d in dirs:
        for r, _, files in os.walk(d):
            files = list(filter(lambda x: 'txt' in x and 'length' in x, files))
            files = list(map(lambda x: os.path.join(r, x), files))
            allfiles.extend(files)
    allfiles = sorted(allfiles)
    # given all the files, plot them
    vals = []
    for fil in allfiles:
        with open(fil, 'r') as fi:
            v = float(fi.read().replace('\n', ''))
            vals.append(v)
    # Make a bar graph
    plt.bar(np.arange(len(vals)), vals, color=['red', 'green', 'lightgreen', 'blue', 'lightblue'], alpha=0.6)
    print([x.split('/')[-1] for x in allfiles])
    plt.xticks(range(5), ['']*5)
    #plt.xticks(np.arange(5), ['BC', 'Ours', 'Ours (noDisTr)', 'GAIL', 'GAIL (noDisTr)'], rotation=30)
    method = ['BC', 'Ours', 'Ours (no disc. training)', 'GAIL', 'GAIL (no disc. training)']
    for i in range(5):
        plt.text(i, 0, str(" "  + method[i]), rotation=90, ha='center', va='bottom', fontsize=18)
    plt.xlabel('Method')
    plt.ylabel('Avg episode length')
    plt.title(envname)
    plt.tight_layout()
    plt.savefig('{}_barplot.png'.format(envname))



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
