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

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)


def read_gene_names(file_path):
    with open(file_path, 'r') as file:
        # Read the first line (column names)
        file.readline()
        # Read and return the gene names
        gene_names = [line.strip().replace('_input', '') for line in file]
    return gene_names

# New function to obtain degree distribution from adjacency matrix
def get_degree_dist(data, model, noise_string):
    if model == "phoenix_ground_truth":
        effects_matrix = np.genfromtxt('/home/ubuntu/lottery_tickets_phoenix/models_for_plots/{}/edge_prior_matrix_chalmers_350_noise_0.0.csv'.format(this_data), delimiter=",")
        binary_matrix = np.where(np.abs(effects_matrix) > 0 , 1, 0)
        out_degrees = np.sum(binary_matrix, axis=1)
    else:
        effects_matrix = get_effect_matrix(this_data, this_model, noise_string)
        threshold = np.percentile(np.abs(effects_matrix), 99)
        binary_matrix = np.where(np.abs(effects_matrix) >= threshold , 1, 0) # Convert to binary matrix using a threshold
        out_degrees = np.sum(binary_matrix, axis=1)         

    
   
    return out_degrees


def get_effect_matrix(this_data, this_model, noise_string):

    model_loc = '/home/ubuntu/lottery_tickets_phoenix/models_for_plots/{}/{}/{}/'.format(this_data, this_model, noise_string)
    Wo_sums = np.genfromtxt(model_loc + "wo_sums.csv", delimiter=",")
    
    Wo_prods = np.genfromtxt(model_loc + "wo_prods.csv", delimiter=",")
    alpha_comb = np.genfromtxt(model_loc + "alpha_comb.csv", delimiter=",")
    gene_mult = np.genfromtxt(model_loc + "gene_mult.csv", delimiter=",")
    gene_mult = np.transpose(np.maximum(gene_mult, 0))
    effects_mat = np.matmul(np.hstack((Wo_sums,Wo_prods)),alpha_comb)
    print(np.mean(Wo_sums == 0), np.mean(effects_mat == 0))
    effects_mat =   effects_mat *gene_mult.T 


    return(effects_mat)

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
    
    gene_names = read_gene_names('/home/ubuntu/lottery_tickets_phoenix/models_for_plots/sim350/gene_names.csv')

    neuron_dict = {"sim350": 40, "sim690": 50}
    models = ["phoenix_ground_truth","phoenix_blind_prune", "phoenix_full_bio_prune", "phoenix_lambda_prune"]
    datasets = ["sim350"]
    noises = [0, 0.025]
    
    
    datahandler_dim = {"sim350": 350}
    model_labels = {
        "phoenix_ground_truth" : "Ground truth", 
        "phoenix_blind_prune": "Blind pruning (\u03BB = 0)",  # Unicode for lambda is \u03BB
        "phoenix_full_bio_prune": "Only biological pruning (\u03BB = 1)",
        "phoenix_lambda_prune": "Our pruning (\u03BB chosen)",
    }
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")    
    #Plotting setup
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)
    fig_heat_sparse = plt.figure(figsize=(18,10)) # tight_layout=True
    axes_heat_sparse = fig_heat_sparse.subplots(ncols= len(models), nrows=len(noises), 
    sharex=False, sharey=True, 
    subplot_kw={'frameon':True})
    #fig_heat_sparse.subplots_adjust(hspace=0, wspace=0)
    border_width = 1.5
    tick_lab_size = 14
    ax_lab_size = 15
    color_mult = 0.12#0.25
    
    plt.grid(True)
    
    print("......")
    
    for this_data in datasets:
        this_neurons = neuron_dict[this_data]
        for this_noise in noises:
            noise_string = "noise_{}".format(this_noise)
            for this_model in models:
                print("Now on model = {}, noise = {}".format(this_model, this_noise))
                
                row_num = noises.index(this_noise)
                this_row_plots = axes_heat_sparse[row_num]
                col_num = models.index(this_model)
                ax = this_row_plots[col_num]
                ax.spines['bottom'].set_linewidth(border_width)
                ax.spines['left'].set_linewidth(border_width)
                ax.spines['top'].set_linewidth(border_width)
                ax.spines['right'].set_linewidth(border_width)
                ax.cla()

                
                degrees = get_degree_dist(this_data, this_model, noise_string)
                n, bins, patches = ax.hist(degrees, bins = 20)

                max_degree_value = np.max(degrees)
                #max_degree_value = np.partition(degrees, -1)[-1]
                
                # Find indices of degrees in the bin with the highest x-axis value
                tolerance = 1e-10
                indices_in_max_bin = np.where(degrees >= max_degree_value)[0]
                
                # Replace indices with corresponding gene names
                gene_names_in_max_bin = [gene_names[idx] for idx in indices_in_max_bin]
                
                # Add a diagonal arrow and display indices of degrees in the annotation text
                ax.annotate(f'Genes: {gene_names_in_max_bin}',
                            xy=(max_degree_value, 0),
                            xytext=(20, 80),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5', color='red'),
                            ha='left')  # Adjust horizontal alignment
            
                        
                if row_num == 0 and col_num == 0:
                    fig_heat_sparse.canvas.draw()
                    labels_y = [item.get_text() for item in ax.get_yticklabels()]
                    labels_y_mod = [(r"$g'$"+item).translate(SUB) for item in labels_y]
                    labels_x = [item.get_text() for item in ax.get_xticklabels()]
                    labels_x_mod = [(r'$g$'+item).translate(SUB) for item in labels_x]
                
                ax.tick_params(axis='x', labelsize= tick_lab_size)
                ax.tick_params(axis='y', labelsize= tick_lab_size)
                    
                if row_num == 0:
                    ax.set_title(model_labels[this_model], fontsize=ax_lab_size, pad = 10)
                if col_num == 0:
                    ax.set_ylabel("Noise level = {:.0%}".format(this_noise/0.5), fontsize = ax_lab_size) 
                 

    
    fig_heat_sparse.savefig('{}/manuscript_fig_degree_dist_icml.png'.format(output_root_dir), bbox_inches='tight')
    