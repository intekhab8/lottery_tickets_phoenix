import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Load data from the CSV file
my_noise = 0.025
data = np.genfromtxt('/home/ubuntu/lottery_tickets_phoenix/models_for_plots/sim350/simu_results_noise_{}_perfect_perfect_prior.csv'.format(my_noise), delimiter=',', dtype=str)

plt.figure(figsize=(9, 6.5)) 
# Extract relevant columns
subdir = data[1:, 0]
balanced_accuracy = data[1:, 2].astype(float)
sd_balanced_accuracy = data[1:, 3].astype(float)
prop_nonzero_parms = data[1:, 4].astype(float)
sd_prop_nonzero_parms = data[1:, 5].astype(float)

# Filter out rows labeled "phx_regularization (balanced accuracy)"
mask = np.array(['phx_regularization (balanced inaccuracy)' not in s for s in subdir], dtype=bool)
mask[-1] = True


filtered_subdir = subdir[mask]
filtered_balanced_accuracy = balanced_accuracy[mask]
filtered_sd_balanced_accuracy = sd_balanced_accuracy[mask]
filtered_prop_nonzero_parms = prop_nonzero_parms[mask]
filtered_sd_prop_nonzero_parms = sd_prop_nonzero_parms[mask]

unique_subdirs = [
    'phx_regularization (balanced inaccuracy)',
     "phx_regularization (post hoc)",
    'phx_L0',
    'blind_pruning',
    'phx_cnode',
     'causal_pruning_fully_bio',
   'phx_pathreg',
    'causal_pruning_PPI_T_STS_0.50_0.05'

]

# Define colors for each subdir
subdir_color = {
    'blind_pruning': 'burlywood',
    'causal_pruning_PPI_T_STS_0.50_0.05': 'lawngreen',
    'causal_pruning_fully_bio': 'forestgreen',
    'phx_L0': 'orange',
    'phx_cnode': 'dodgerblue',
    'phx_pathreg': 'orchid',
    'phx_regularization (balanced inaccuracy)': 'silver',
    "phx_regularization (post hoc)" : 'red'
}

subdir_labels = {
    'blind_pruning': 'IMP',
    'causal_pruning_PPI_T_STS_0.50_0.05': 'DASH',
    'causal_pruning_fully_bio': 'Bio Pruning',
    'phx_L0': '$L_0$',
    'phx_cnode': 'C-NODE',
    'phx_pathreg': 'PathReg',
    'phx_regularization (balanced inaccuracy)': 'Baseline (Unpruned)',
    "phx_regularization (post hoc)" : 'Post hoc (Finetuned)'
}

subdir_symbol = {
    'blind_pruning': 'v',
    'causal_pruning_PPI_T_STS_0.50_0.05': '*',
    'causal_pruning_fully_bio': 'v',
    'phx_L0': 'X',
    'phx_cnode': 'X',
    'phx_pathreg': 'X',
    'phx_regularization (balanced inaccuracy)': 'o',
     "phx_regularization (post hoc)" : 'P'
}

legend_labels = [subdir_labels[subdir_val] for subdir_val in unique_subdirs]

border_width = 1.5
tick_lab_size = 15
ax_lab_size = 17.5

plt.gca().spines['bottom'].set_linewidth(border_width)
plt.gca().spines['left'].set_linewidth(border_width)
plt.gca().spines['top'].set_linewidth(border_width)
plt.gca().spines['right'].set_linewidth(border_width)
plt.tick_params(axis='x', labelsize=tick_lab_size)
plt.tick_params(axis='y', labelsize=tick_lab_size)


for subdir_val in unique_subdirs:
    idx = np.where(filtered_subdir == subdir_val)
    color = subdir_color[subdir_val]
    symbol = subdir_symbol[subdir_val]

    if subdir_val == 'causal_pruning_PPI_T_STS_0.50_0.05':
        marker_size = 500
    else:
        marker_size = 280
    if not filtered_sd_prop_nonzero_parms[idx] < 0.015 and not filtered_sd_balanced_accuracy[idx] < 0.01:
        plt.errorbar(1 - filtered_prop_nonzero_parms[idx], filtered_balanced_accuracy[idx],
                    xerr=filtered_sd_prop_nonzero_parms[idx], yerr=filtered_sd_balanced_accuracy[idx],
                    fmt='None', capsize=5, ecolor='black',  alpha=0.8)  # Plot error bars
    elif filtered_sd_prop_nonzero_parms[idx] < 0.015 and not filtered_sd_balanced_accuracy[idx] < 0.01:
                plt.errorbar(1 - filtered_prop_nonzero_parms[idx], filtered_balanced_accuracy[idx],
                    yerr=filtered_sd_balanced_accuracy[idx],
                    fmt='None', capsize=5, ecolor='black',  alpha=0.8)  # Plot error bars
    elif not filtered_sd_prop_nonzero_parms[idx] < 0.015 and filtered_sd_balanced_accuracy[idx] < 0.01:
        plt.errorbar(1 - filtered_prop_nonzero_parms[idx], filtered_balanced_accuracy[idx],
            xerr=filtered_sd_prop_nonzero_parms[idx],
            fmt='None', capsize=5, ecolor='black',  alpha=0.8)  # Plot error bars           


    plt.scatter(1 - filtered_prop_nonzero_parms[idx], filtered_balanced_accuracy[idx],
                color=color, edgecolor='black',  # Set marker border color to black
                marker=symbol, s=marker_size, label=subdir_val,  alpha=1) 




arrow_text_size = 15.5
arrow_color = 'firebrick'
arrow_len_y = 0.04

plt.annotate('Sparser',  xy=(0.50, 0.98), xytext=(0.65, 0.98),
             arrowprops=dict(color='gray', arrowstyle='<-'), fontsize=arrow_text_size,
             horizontalalignment='left', verticalalignment='center', color=arrow_color)

plt.annotate('Denser', xy=(0.50, 0.98), xytext=(0.35, 0.98),
             arrowprops=dict(color='gray', arrowstyle='<-'), fontsize=arrow_text_size,
             horizontalalignment='right', verticalalignment='center', color=arrow_color)

plt.annotate('Meaningful biology', xy=(0.03, 0.75), xytext=(0.03, 0.75 + arrow_len_y),
             arrowprops=dict(color='gray', arrowstyle='<-'), fontsize=arrow_text_size,
             horizontalalignment='center', verticalalignment='bottom', color=arrow_color, rotation=90)

plt.annotate('Spurious biology', xy=(0.03, 0.75), xytext=(0.03, 0.75 - arrow_len_y),
             arrowprops=dict(color='gray', arrowstyle='<-'), fontsize=arrow_text_size,
             horizontalalignment='center', verticalalignment='top', color=arrow_color, rotation=90)


plt.ylim(0.5, 1)
plt.xlim(0, 1)

# Set labels and title
plt.xlabel('Model sparsity',  fontsize = ax_lab_size)
plt.ylabel('Balanced accuracy \n (non-zero features vs ground truth biology)',  fontsize = ax_lab_size)

# Format x and y axes as percentages
formatter = FuncFormatter(lambda x, _: f'{x:.0%}')
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
#plt.xscale('log')
# Add legend
# Create legend without error bars
handles, labels = plt.gca().get_legend_handles_labels()
# Filter out the error bars from handles
handles = [h for h in handles]  # Take only the data point without the error bars
plt.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.45, 1.18), fontsize=15, ncol=4, frameon=False)

# Add vertical line at x=0.5
plt.axvline(x=0.5, color='lightgray', linestyle='--', linewidth=2)

# Add horizontal line at y=0.75
plt.axhline(y=0.75, color='lightgray', linestyle='--', linewidth=2)

plt.savefig('/home/ubuntu/lottery_tickets_phoenix/ode_net/code/output/just_plots/manuscript_fig_BASparse_5.png'.format(my_noise), dpi=200, bbox_inches='tight')