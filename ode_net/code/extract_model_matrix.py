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
from solve_eq import solve_eq
from visualization_inte import *
import matplotlib.pyplot as plt

#torch.set_num_threads(4) #CHANGE THIS!

def make_mask(X):
    triu = np.triu(X)
    tril = np.tril(X)
    triuT = triu.T
    trilT = tril.T
    masku = abs(triu) > abs(trilT)
    maskl = abs(tril) > abs(triuT)
    main_mask = ~(masku | maskl)
    X[main_mask] = 0


sums_model = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/trained_model_epoch_100_sums.pt')
prods_model = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/trained_model_epoch_100_prods.pt')
alpha_comb_sums = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/trained_model_epoch_100_alpha_comb_sums.pt')
alpha_comb_prods = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/trained_model_epoch_100_alpha_comb_prods.pt')
gene_mult = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/trained_model_epoch_100_gene_multipliers.pt')

print(torch.sum(sums_model.linear_out.weight==0)/torch.numel(sums_model.linear_out.weight))
Wo_sums = np.transpose(sums_model.linear_out.weight.detach().numpy())
Bo_sums = np.transpose(sums_model.linear_out.bias.detach().numpy())
Wo_prods = np.transpose(prods_model.linear_out.weight.detach().numpy())
Bo_prods = np.transpose(prods_model.linear_out.bias.detach().numpy())
alpha_comb_sums = np.transpose(alpha_comb_sums.linear_out.weight.detach().numpy())
alpha_comb_prods = np.transpose(alpha_comb_prods.linear_out.weight.detach().numpy())
gene_mult = np.transpose(torch.relu(gene_mult.detach()).numpy())

effects_mat = np.matmul(Wo_sums,alpha_comb_sums) + np.matmul(Wo_prods,alpha_comb_prods)
effects_mat =   effects_mat #gene_mult.T*
make_mask(effects_mat)

np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/effects_mat.csv", effects_mat, delimiter=",")

