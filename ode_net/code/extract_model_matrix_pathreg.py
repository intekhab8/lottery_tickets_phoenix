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

def get_sparsity(model):
    
    # MODEL SPARSITY
    for i, layer in enumerate(model[1].odefunc.layers):
        if i ==0:
            all_weights = torch.abs(layer.sample_weights()).flatten()
        else:
            all_weights = torch.cat((all_weights,torch.abs(layer.sample_weights()).flatten()))
    model_spar = (torch.abs(all_weights)<1e-5).sum() / all_weights.shape[0]
    
    return model_spar


def get_sparsity_no_gates(model):
    
    # MODEL SPARSITY without any stochasticity
    for i, layer in enumerate(model[1].odefunc.layers):
        if i ==0:
            all_weights = torch.abs(layer.weights).flatten()
        else:
            all_weights = torch.cat((all_weights,torch.abs(layer.weights).flatten()))
    model_spar = (torch.abs(all_weights)<1e-5).sum() / all_weights.shape[0]
    return model_spar


#torch.set_num_threads(4) #CHANGE THIS!
pathreg_model = torch.load('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/_pretrained_best_model/best_val_model.pt')

'''
WM = torch.abs(pathreg_model['model_state_dict']['1.odefunc.output.0.weights'])
WM = torch.matmul(WM, torch.abs(pathreg_model['model_state_dict']['1.odefunc.output.2.weights']))
WM = torch.matmul(WM, torch.abs(pathreg_model['model_state_dict']['1.odefunc.output.3.weights']))
#WM = WM.T.cpu().detach().numpy()
WM = WM.cpu().detach().numpy()
WM[np.abs(WM)<1e-5] = 0.
'''

transpose = True
for i, layer in enumerate(pathreg_model[1].odefunc.layers):
    if i ==0:
        WM = torch.abs(layer.sample_weights_ones())
    else:
        WM = torch.matmul(WM,torch.abs(layer.sample_weights_ones()))
if transpose: 
    WM = WM.T.detach().numpy()
    print("Transposing W")
else:
    WM = WM.detach().numpy()
WM[np.abs(WM)<1e-5] = 0.

print("Sparsity : {:.2%}".format( get_sparsity(pathreg_model).item() ))
print("Sparsity (no stoch gates) : {:.2%}".format( get_sparsity_no_gates(pathreg_model).item() ))

np.savetxt("/home/ubuntu/lottery_tickets_phoenix/ode_net/code/model_inspect/effects_mat_pathreg.csv", WM, delimiter=",")

