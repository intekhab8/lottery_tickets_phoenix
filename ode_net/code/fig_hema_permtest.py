import sys
import os
import numpy as np
import csv 

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

from datahandler import DataHandler
from odenet import ODENet
from read_config import read_arguments_from_file
from visualization_inte import *

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as colors
import numpy as np

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)


if __name__ == "__main__":

    sys.setrecursionlimit(3000)
    save_file_name = "just_plots"

    output_root_dir = '{}/{}/'.format("output", save_file_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)

    neuron_dict = {"sim350": 40, "sim690": 50}
    #models = ["phoenix"]
    datasets = ["hema_data"]


    #Plotting setup
    fig_hema_data = plt.figure(figsize=(2.5,7), tight_layout=True)
    gs = fig_hema_data.add_gridspec(10, 6, hspace = 0.03, wspace = 0.02) #
    border_width = 1.5
    tick_lab_size = 12
    ax_lab_size = 13
    all_genes = ['blind_pruning','pathreg','lambda_pruning']
    prune_labels = {'blind_pruning': 'IMP',
                    'pathreg':'PathReg',
                    'lambda_pruning': 'DASH',
                    'lambda2_pruning': 'DASH',
                    'cnode': "C-NODE",
                    'L0': r"$L_0$"}


    print("......")

    this_data = "hema_data"

    ax = fig_hema_data.add_subplot(gs[0:10, 0:5])

    ax.spines['bottom'].set_linewidth(border_width)
    ax.spines['left'].set_linewidth(border_width)
    ax.spines['top'].set_linewidth(border_width)
    ax.spines['right'].set_linewidth(border_width)
    ax.cla()

    analysis_type = "reactome"
    lineage = "B"
    
    print("making heatmap")
    wide_file = "/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/hema_data_{}/all_permtests_".format(lineage) + analysis_type +"_wide.csv"
    z = np.loadtxt(open(wide_file, "rb"),  dtype = "str",delimiter=",", skiprows=1)[:,1:]
    num_tops = z.shape[0]
    print("The analysis contains", num_tops, "pathways.")
    num_models = z.shape[1]
    z[z == ""] = "0"                
    z = z.astype(float)
    z[ z < 0] = 0 #set negative values to 0
    z = z.transpose()
    
    pval_file = "/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/hema_data_{}/all_permtests_".format(lineage) + analysis_type +"_wide_pval.csv"
    p_vals = np.loadtxt(open(pval_file, "rb"),  dtype = "str",delimiter=",", skiprows=1)[:,1:]
    p_vals[p_vals == ""] = "1"
    p_vals = p_vals.astype(float)

    ind = np.arange(num_models) +0.5
    ax.set_xlim(0,num_models)
    ax.set_ylim(0,num_tops)

    path_names = np.loadtxt(open(wide_file, "rb"), 
        dtype = "str",delimiter=",", skiprows=1, usecols = (0))
    path_names = np.char.strip(path_names, '"')

    z_min, z_max = np.nanmin(z), np.nanmax(z)
    print(z.transpose())
    c = ax.pcolormesh(z.transpose(), cmap='Reds', #z.transpose() #vmin=0, vmax= z_max, #120
            norm = colors.PowerNorm(gamma = 1))  #, #gamma = 2 shading = "nearest"

     # Overlay stars based on p-values
    for i in range(num_tops):
        for j in range(num_models):
            if p_vals[i, j] < 0.05/1000:  # Adjust the significance threshold as needed
                ax.text(j + 0.5, i + 0.4, "*", ha='center', va='center', color='black', fontsize = 20)

#    fig_hema_data.savefig('{}/manuscript_fig_hema_data_permtest_{}.png'.format(output_root_dir, analysis_type), bbox_inches='tight')
    
    for idx in range(num_tops):
        ax.axhline(y=idx, xmin=0, xmax=num_models, linestyle='dotted', color = "black", alpha = 0.3)

    for idx in range(num_models):
        ax.axvline(x=idx, ymin=0, ymax=num_tops, linestyle='dotted', color = "black", alpha = 0.3)    

        

    print("......")
    ax1 = fig_hema_data.add_subplot(gs[0, 0:5], sharex = ax)
    #ax1.spines['bottom'].set_linewidth(border_width)
    #ax1.spines['left'].set_linewidth(border_width)
    #ax1.spines['top'].set_linewidth(border_width)
    #ax1.spines['right'].set_linewidth(border_width)
    ax1.set_frame_on(False)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.cla()
    
    
    width= 0.17  # the width of the bars
    deltas = [-0.5,0.5]
    cost_shrinker = 1.8
    '''
    for this_metric in metrics:
        this_delta = deltas[metrics.index(this_metric)] 
        this_perf_vals = [perf_info[this_gene][this_metric]/cost_shrinker if this_metric == "runtime_cost" else perf_info[this_gene][this_metric] for this_gene in all_genes ]
        ax1.bar(ind  + width*this_delta, this_perf_vals, width = width, 
                        capsize = 5, color = metric_cols[this_metric], alpha = 0.7,  edgecolor = "black", 
                        linewidth = 1.5, align = 'center')
        
    for this_metric in metrics:
        this_delta = deltas[metrics.index(this_metric)] 
        this_perf_vals = [perf_info[this_gene][this_metric] for this_gene in all_genes]
        for this_idx in range(len(this_perf_vals)):
            this_val = this_perf_vals[this_idx]
            this_text = this_val
            this_height = this_val
            this_text = "{:.2f}".format(this_val)
            if this_metric == "runtime_cost":
                this_height = this_val/cost_shrinker
            
            ax1.text(this_idx + 0.5 + width*this_delta, this_height + 0.1 , this_text, 
                     ha="center", va="bottom", color="black",size = 20, rotation = 90)
    
    ax.legend(handles = leg_general_info, loc='center left', prop={'size': tick_lab_size+3}, 
                        ncol = 1,  handleheight=1.5, frameon = False, bbox_to_anchor = (-0.4,1.1),
                        title = "Performance", title_fontsize=20) #

    print("......")

    '''

    ax.set_yticks(np.arange(num_tops)+0.5)    
    ax.set_yticklabels(path_names)
    ax.tick_params(axis='y', labelsize= tick_lab_size+4, rotation = 0)
    
    ax.set_xticks(ind)    
    ax.set_xticklabels([ prune_labels[this_tot_gene] for this_tot_gene in all_genes], rotation=25, ha = "center")
    for tick, pad_value in zip(ax.xaxis.get_major_ticks(), [0,-8,0]): #-8,
        tick.set_pad(pad_value)
    
    ax.tick_params(axis='x', labelsize= tick_lab_size+3, length=8)
    
    

    cbar =  fig_hema_data.colorbar(c, ax= ax, use_gridspec = False,  
                            shrink=0.8, orientation = "horizontal", pad = 0.11)
    
    cbar.set_ticks([0 , 3, 7])
    cbar.set_ticklabels([r"$z$-score =0", '3', '7'], ha='right')
    cbar.ax.tick_params(labelsize = tick_lab_size+3) 
    cbar.outline.set_linewidth(2)
    
    
    fig_hema_data.savefig('{}/manuscript_fig_hema_data_permtest_{}_icml_straight.png'.format(output_root_dir, lineage),  bbox_inches='tight') #

