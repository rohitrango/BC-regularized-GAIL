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
parser.add_argument('--files', nargs='+', required=True,
            help='Get filenames')
parser.add_argument('--legend', nargs='+', default=[],
            help='Legend values')
parser.add_argument('--max_steps', default=0, type=int)
parser.add_argument('--yscale', default='linear', type=str)
parser.add_argument('--bcpath', default='', type=str)
args = parser.parse_args()

def check_last_name(result):
    path = result.dirname
    splits = path.split('/')
    for sp in splits[::-1]:
        if sp == '':
            continue
        try:
            sp = int(sp)
        except:
            return sp
    return ''


def main():
    """
    Plot the plots inside the folder given
    """
    # Now plot the common things
    splits = args.files[0].split('/')
    if splits[-1] == '':
        splits = splits[-2]
    else:
        splits = splits[-1]
    env = splits
    results = pu.load_results(args.files, )
    fig = pu.plot_results(results, average_group=True,
            shaded_err=False,
            shaded_std=True,
            group_fn=lambda _: check_last_name(_),
            split_fn=lambda _: '', figsize=(10, 10))

    # Add results for behaviour cloning if present
    allbcfiles = [args.bcpath]
    allfiles = []
    allrandomfiles = []  # For random agent behavior

    for file in allbcfiles:
        for r, dirs, files in os.walk(file):
            print(files)
            txtfiles = list(filter(lambda x: 'BC_' in x and '.txt' in x, files))
            rndfiles = list(filter(lambda x: 'random_' in x and '.txt' in x, files))
            allfiles.extend(list(map(lambda x: os.path.join(r, x), txtfiles)))
            allrandomfiles.extend(list(map(lambda x: os.path.join(r, x), rndfiles)))

    ## Show all files for BC and plot
    print(allfiles)
    if allfiles != []:
        bcreward = []
        for file in allfiles:
            with open(file, 'r') as fi:
                rews = fi.read().split('\n')
                rews = filter(lambda x: x != '', rews)
                rews = list(map(lambda x: float(x), rews))
                bcreward.extend(rews)

        # Get mean and std
        #print(bcreward)
        mean = np.mean(bcreward)
        std = np.std(bcreward)
        idxcolor=10
        plt.plot([0, args.max_steps], [mean, mean], label='BC', color=COLORS[idxcolor])
        plt.fill_between([0, args.max_steps], [mean - std, mean - std], [mean + std, mean + std], alpha=0.2, color=COLORS[idxcolor])

    ## Get random policy
    if allrandomfiles != []:
        rndreward = []
        for file in allrandomfiles:
            with open(file, 'r') as fi:
                rews = fi.read().split('\n')
                rews = filter(lambda x: x != '', rews)
                rews = list(map(lambda x: float(x), rews))
                rndreward.extend(rews)

        # Get mean and std
        #print(bcreward)
        mean = np.mean(rndreward)
        plt.plot([0, args.max_steps], [mean, mean], label='random', color='gray', linestyle='dashed')

    plt.xlabel('# environment interactions', fontsize=20)
    envnamehere = 'ant'
    if env.lower().startswith(envnamehere):
        plt.ylim(ymin=-5000, ymax=5000)
    if env.lower().startswith(''):
        plt.ylabel('Reward', fontsize=30)
    plt.yscale(args.yscale)
    plt.title(env.replace('BC','').replace('GAIL', '').replace('no', '').replace('alph', ''), \
            fontsize=50)

    if env.lower().startswith(envnamehere):
        if args.legend != []:
            if allfiles != []:
                args.legend.append('BC')
            plt.legend(args.legend, fontsize=30, loc='bottom right')
    else:
        plt.legend().set_visible(False)
    #plt.ticklabel_format(useOffset=1)
    plt.savefig('{}.png'.format(env), bbox_inches='tight', )
    print("saved ", env)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
