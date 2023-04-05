### This creates the 

import pickle as pkl
import pandas as pd
import numpy as np
import os
import seaborn as sns
import pandas as pd
import numpy as np
import pickle as pkl
from itertools import product
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl

def save_figure(fname, fig, palette):
    if palette == "binary_r":
        fname += "-bw.pdf"
    fig.savefig(fname)


def plot_trace(df, palette):
    ylim, xlim = (0.001, 1), (0.925, 0.9475)
    df_mean = pkl.load(open("asymptotic-distribution-mean.pkl", "rb"))
    estimate = df_mean[0]
    se = np.sqrt(
    pkl.load(open("asymptotic-distribution-cov.pkl", "rb"))[
            0, 0
        ]
    )
    confi_upper, confi_lower = (
        estimate - 1.645 * se,
        estimate + 1.645 * se,
    )

    x, y,z = df_trace["Delta"], df_trace["Impact"],df_trace["ratio_ll_base"]

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots(1,1,figsize=(13, 8))
    #fig,ax = plt.subplots()
    # make a plot
    ax.plot(x, y, linewidth = 5)
    kwargs = {
        "ec": "black",
        "color": "grey",
        "alpha": 0.2,
        "label": r"$U_\delta(0.1)$",
    }
    args = ([confi_lower, 0], confi_upper - confi_lower, ylim[1] * 0.2)
    rect = mpatches.Rectangle(*args, **kwargs)
    ax.add_patch(rect)

    ax.axvline(x=estimate, label="Point estimate", linestyle="--", linewidth = 5)
    ax.set_xlabel(r"$\Delta$ Schooling")
    ax.legend(loc="upper left", frameon = False)

    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\Delta$ Schooling")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xticks(
        [
            estimate - 0.01,
            confi_lower,
            estimate - 0.005,
            estimate,
            estimate + 0.005,
            confi_upper,
            estimate + 0.01,
        ]
    )

    ax.set_xticklabels(
        [
            round(estimate - 0.01, 3),
            r"$\delta_U$",
            round(estimate - 0.005, 3),
            r"$\hat{\delta}$",
            round(estimate + 0.005, 3),
            r"$\delta_L$",
            round(estimate + 0.01, 3),
        ],
    )

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(x, z,color="gray",linestyle='dotted', linewidth = 5)
    ax2.set_ylabel("Scaled neg. log Likelihood")
#    plt.rcParams['figure.figsize'] = [14, 10]
    ax.spines[['top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)


    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.usetex"] = True
    #plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.rcParams["axes.labelsize"] = 30
    plt.rcParams["xtick.labelsize"] = 26
    plt.rcParams["ytick.labelsize"] = 26
    plt.rcParams["legend.fontsize"] = 22
   
    save_figure("fig-trace-delta", fig, palette)

df_trace = pkl.load(open("df-delta-trace.pkl", "rb"))


for palette in ["tab10", "binary_r"]:
    sns.set_palette(palette)
   
    if palette == "binary_r":
    
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
        mpl.rcParams["axes.labelsize"] = 30
        mpl.rcParams["xtick.labelsize"] = 26
        mpl.rcParams["ytick.labelsize"] = 26
        mpl.rcParams["legend.fontsize"] = 22
   
   
    plot_trace(df_trace, palette)


#    if palette == "binary_r":
#
#        mpl.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
#        mpl.rcParams["text.usetex"] = True
#        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
#        mpl.rcParams["axes.labelsize"] = 30
#        mpl.rcParams["xtick.labelsize"] = 26
#        mpl.rcParams["ytick.labelsize"] = 26
#        mpl.rcParams["legend.fontsize"] = 22
        
        
    
  
   



