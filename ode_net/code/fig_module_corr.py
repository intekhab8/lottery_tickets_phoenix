import sys
import os
import numpy as np
import csv 
import math 

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
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def get_effect_matrix(this_data, this_model, noise_string):

    model_loc = '/home/ubuntu/lottery_tickets_phoenix/models_for_plots/{}/{}/{}/'.format(this_data, this_model, noise_string)
    Wo_sums = np.genfromtxt(model_loc + "wo_sums.csv", delimiter=",")
    
    Wo_prods = np.genfromtxt(model_loc + "wo_prods.csv", delimiter=",")
    alpha_comb = np.genfromtxt(model_loc + "alpha_comb.csv", delimiter=",")
    gene_mult = np.genfromtxt(model_loc + "gene_mult.csv", delimiter=",")
    gene_mult = np.transpose(np.maximum(gene_mult, 0))
    effects_mat = np.matmul(np.hstack((Wo_sums,Wo_prods)),alpha_comb)
    effects_mat =   effects_mat *gene_mult.T 

    return(effects_mat)


def get_mutual_info_matrix(this_data, this_model, noise_string):
    model_loc = '/home/ubuntu/lottery_tickets_phoenix/models_for_plots/{}/{}/{}/'.format(this_data, this_model, noise_string)
    Wo_sums = np.genfromtxt(model_loc + "wo_sums.csv", delimiter=",")
    num_neurons = Wo_sums.shape[1]
    mutual_info_matrix = np.zeros((num_neurons, num_neurons))
    # Calculate mutual information between each pair of variables
    for i in range(num_neurons):
        for j in range(num_neurons):
            mutual_info_matrix[i, j] = mutual_info_regression(Wo_sums[:, i+i+1], Wo_sums[:, j:j+1])

    #MI_matrix = np.corrcoef(Wo_sums, rowvar=False)
    return(mutual_info_matrix)

def binarize_matrix(matrix, threshold=0.5):
    # Binarize the matrix based on the threshold
    binary_matrix = (matrix > threshold).astype(int)
    return binary_matrix


def get_module_corr_matrix(this_data, this_model, noise_string):
    model_loc = '/home/ubuntu/lottery_tickets_phoenix/models_for_plots/{}/{}/{}/'.format(this_data, this_model, noise_string)
    Wo_sums = np.genfromtxt(model_loc + "wo_sums.csv", delimiter=",")
    Wo_prods = np.genfromtxt(model_loc + "wo_prods.csv", delimiter=",")
    stacked_matrix = np.hstack((np.abs(Wo_sums), np.abs(Wo_prods)))
    inner_product_matrix = np.dot(stacked_matrix, stacked_matrix.T)
    
    return(inner_product_matrix)

def row_normalize(matrix):
    row_sums = np.sum(np.abs(matrix), axis=1, keepdims=True)     # Calculate the sum of absolute values for each row
    row_sums[row_sums == 0] = 1.0 # Avoid division by zero
    normalized_matrix = matrix / row_sums     # Row-normalize the matrix
    print(row_sums.shape)
    return normalized_matrix

def column_normalize(matrix):
    col_sums = np.sum(np.abs(matrix), axis=0, keepdims=True) # Calculate the sum of absolute values for each column
    col_sums[col_sums == 0] = 1.0 # Avoid division by zero
    print(col_sums.shape)
    normalized_matrix = matrix / col_sums # Column-normalize the matrix
    return normalized_matrix


if __name__ == "__main__":

    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format('val_config_inte.cfg'))
    settings = read_arguments_from_file('val_config_inte.cfg')
    save_file_name = "just_plots"

    output_root_dir = '{}/{}/'.format(settings['output_dir'], save_file_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    
    neuron_dict = {"sim350": 40, "sim690": 50}
    models = ["phoenix_ground_truth","pathreg", "phoenix_blind_prune", "phoenix_full_bio_prune", "phoenix_lambda_prune"]
    datasets = ["sim350"]
    noises = [0.025]
    
    
    datahandler_dim = {"sim350": 350}
    model_labels = {
        "phoenix_ground_truth": "Ground truth",
        "phoenix_blind_prune": "IMP",  # Unicode for lambda is \u03BB
        "phoenix_full_bio_prune": "Bio Pruning",
        "phoenix_lambda_prune": "DASH",
        "pathreg": "PathReg",
    }
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")    
    #Plotting setup
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)
    fig_heat_sparse = plt.figure(figsize=(27,5*len(noises))) # tight_layout=True
    axes_heat_sparse = fig_heat_sparse.subplots(ncols= len(models), nrows=len(noises), 
    sharex=False, sharey=False, 
    subplot_kw={'frameon':True})
    #fig_heat_sparse.subplots_adjust(hspace=0, wspace=0)
    border_width = 1.5
    tick_lab_size = 16
    ax_lab_size = 24
    color_mult = 0.02#0.025
    
    plt.grid(True)
    
    print("......")
    
    for this_data in datasets:
        this_neurons = neuron_dict[this_data]
        for this_noise in noises:
            noise_string = "noise_{}".format(this_noise)
            for this_model in models:
                print("Now on model = {}, noise = {}".format(this_model, this_noise))
                
                row_num = noises.index(this_noise)
                if len(axes_heat_sparse.shape) == 1:
                    this_row_plots = [axes_heat_sparse[col_num] for col_num in range(len(axes_heat_sparse))]
                else:    
                    this_row_plots = axes_heat_sparse[row_num]
                col_num = models.index(this_model)
                ax = this_row_plots[col_num]
                ax.spines['bottom'].set_linewidth(border_width)
                ax.spines['left'].set_linewidth(border_width)
                ax.spines['top'].set_linewidth(border_width)
                ax.spines['right'].set_linewidth(border_width)
                ax.cla()

                y, x = np.meshgrid(np.linspace(1, datahandler_dim[this_data], datahandler_dim[this_data]), np.linspace(1, datahandler_dim[this_data], datahandler_dim[this_data]))
                if this_model == "phoenix_ground_truth":
                    z = np.genfromtxt('/home/ubuntu/lottery_tickets_phoenix/models_for_plots/{}/edge_prior_matrix_chalmers_350_noise_0.0.csv'.format(this_data), delimiter=",")
                    z = np.dot(np.abs(z), np.abs(z).T)
                    #print(z.shape)
                else:    
                    z = get_module_corr_matrix(this_data, this_model, noise_string)


                #z = row_normalize(z)
                #row_sums =  np.abs(z).sum(axis=1)
                #z = z / row_sums[:, np.newaxis]
                #ax.axvline(x=0.50, color='black', linewidth=2)
               
                z_min, z_max = color_mult*np.abs(z).min(), color_mult*np.abs(z).max()
                c = ax.pcolormesh(x, y, z, cmap='inferno', vmin=z_min, vmax=z_max) 
                ax.axis([x.min(), x.max(), y.min(), y.max()]) 
                
                if row_num == 0 and col_num == 0:
                    fig_heat_sparse.canvas.draw()
                    labels_y = [item.get_text() for item in ax.get_yticklabels()]
                    labels_y_mod = [(r"$g$"+item).translate(SUB) for item in labels_y]
                    labels_x = [item.get_text() for item in ax.get_xticklabels()]
                    labels_x_mod = [(r'$g$'+item).translate(SUB) for item in labels_x]
                
                #ax.set_xticklabels(labels_x_mod)
                #ax.set_yticklabels(labels_y_mod)
                ax.tick_params(axis='x', labelsize= tick_lab_size)
                ax.tick_params(axis='y', labelsize= tick_lab_size)
                    
                if row_num == 0:
                    ax.set_title(model_labels[this_model], fontsize=ax_lab_size, pad = 10)
                if col_num == 0:
                    if len(axes_heat_sparse.shape) == 1:
                        ax.set_ylabel("Coregulatory effects", fontsize = ax_lab_size) 
                    else:    
                        ax.set_ylabel("Noise level = {:.0%}".format(this_noise/0.5), fontsize = ax_lab_size) 
                 
    cbar =  fig_heat_sparse.colorbar(c, ax=axes_heat_sparse.ravel().tolist(), 
                                        shrink=0.50, orientation = "vertical", pad = 0.01)
    cbar.set_ticks([0.02, 0.30])
    cbar.set_ticklabels(['None', 'Strong'])
    cbar.ax.tick_params(labelsize = tick_lab_size+5) 
    #cbar.set_label(r'$\widetilde{D_{ij}}$= '+'Estimated coregulation of '+ r'$g_i$'+ ' and ' +r"$g_j$" +' in SIM350', size = ax_lab_size)
    cbar.outline.set_linewidth(2)

    
    fig_heat_sparse.savefig('{}/manuscript_fig_module_corr_5.png'.format(output_root_dir), bbox_inches='tight')
    