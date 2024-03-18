# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np
from tqdm import tqdm
from math import ceil
from time import perf_counter, process_time

import torch
import torch.optim as optim

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

#from datagenerator import DataGenerator
from datahandler import DataHandler
from odenet import ODENet
from read_config import read_arguments_from_file
from visualization_inte import *

#torch.set_num_threads(16) #CHANGE THIS!


def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

MODEL_TYPE = 'L0'

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_breast.cfg')
clean_name =  "desmedt_11165genes_1sample_186T" 
parser.add_argument('--data', type=str, default='/home/ubuntu/lottery_tickets_phoenix/breast_cancer_data/clean_data/{}.csv'.format(clean_name))

args = parser.parse_args()
device = "cpu"
# Main function
if __name__ == "__main__":
    print("Getting influence scores for {}".format(MODEL_TYPE))
    print("----------------")
    print('Setting recursion limit to 3000')
    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format(args.settings))
    settings = read_arguments_from_file(args.settings)
    
    data_handler = DataHandler.fromcsv(args.data,device , settings['val_split'], normalize=settings['normalize_data'], 
                                        batch_type=settings['batch_type'], batch_time=settings['batch_time'], 
                                        batch_time_frac=settings['batch_time_frac'],
                                        noise = settings['noise'],
                                        img_save_dir = "NULL",
                                        scale_expression = settings['scale_expression'],
                                        log_scale = settings['log_scale'],
                                        init_bias_y = settings['init_bias_y'])


    pretrained_model_file = '/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/breast_cancer/{}/best_val_model.pt'.format(MODEL_TYPE)
    pathreg_model = torch.load(pretrained_model_file)
    print(pathreg_model)


    time_pts_to_project = torch.from_numpy(np.arange(0,1,0.1))
    pathreg_model[1].set_times(time_pts_to_project)
    n_random_inputs_per_gene = 3
    all_scores = []
    #Read in the prior matrix
    for this_gene in tqdm(range(data_handler.dim), desc="Genes Progress"):#
        this_gene_score_sum = 0
        for random_input_idx in range(n_random_inputs_per_gene):
            #print(this_gene, random_input_idx)
            this_init = 1*(torch.rand(1,1,data_handler.dim, device = data_handler.device) - 0.5)
            unpert_out = pathreg_model(this_init)
            this_pert_col =  1*(torch.rand(1,device = data_handler.device) - 0.5)
            this_init[:,0, this_gene] = this_pert_col 
            pert_out = pathreg_model(this_init) 
            all_other_genes = [idx for idx in range(data_handler.dim) if idx != this_gene]
            this_gene_score_sum += torch.mean(abs(unpert_out[1:,:,all_other_genes] - pert_out[1:,:,all_other_genes])).item()
        
        all_scores.append(this_gene_score_sum/n_random_inputs_per_gene)
        if this_gene % 1000 == 0: 
            np.savetxt('/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/breast_cancer/inferred_influences/inferred_influence_{}_first_{}.csv'.format(MODEL_TYPE, this_gene+1), 
                        all_scores, delimiter=',') 


    print("done, saving now!")
    np.savetxt('/home/ubuntu/lottery_tickets_phoenix/all_manuscript_models/breast_cancer/inferred_influences/inferred_influence_{}.csv'.format(MODEL_TYPE), 
                    all_scores, delimiter=',') 