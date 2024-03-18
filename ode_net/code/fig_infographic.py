import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

save_file_name = "just_plots"
output_root_dir = '{}/{}/'.format("output", save_file_name)
# Create a figure and axis object
plt.figure(figsize=(8, 6))



# Create a neural network graph
fig, ax = plt.subplots(figsize=(8, 6))
# Define network sizes
network_sizes = [8, 5, 3]  # Example: [input_size, hidden_size, output_size]

# Draw the first neural network (blue)
for i in range(len(network_sizes) - 1):
    x_start = i * 2
    x_end = (i + 1) * 2
    y_start = network_sizes[i]
    y_end = network_sizes[i + 1]

    # Draw circles representing nodes
    for j in range(y_start):
        ax.add_patch(plt.Circle((x_start, j), radius=0.3, color='skyblue'))
    for j in range(y_end):
        ax.add_patch(plt.Circle((x_end, j), radius=0.3, color='skyblue'))

    # Draw arrows representing connections
    for j in range(y_start):
        for k in range(y_end):
            ax.add_patch(FancyArrowPatch((x_start, j), (x_end, k),
                                          connectionstyle="arc3,rad=0.3", color='black'))

# Draw the pruning step (red with transparency)
pruning_x = len(network_sizes) * 2 + 1
for i in range(network_sizes[1]):
    ax.add_patch(plt.Circle((pruning_x, i), radius=0.3, color='red', alpha=0.3))  # Set transparency
for i in range(network_sizes[2]):
    ax.add_patch(plt.Circle((pruning_x + 2, i), radius=0.3, color='red', alpha=0.3))  # Set transparency

for i in range(network_sizes[1]):
    for j in range(network_sizes[2]):
        ax.add_patch(FancyArrowPatch((pruning_x, i), (pruning_x + 2, j),
                                      connectionstyle="arc3,rad=0.3", color='black', alpha=0.3))  # Set transparency

# Set axis limits and remove ticks
ax.set_xlim(-1, len(network_sizes) * 2 + 3)
ax.set_ylim(-1, max(network_sizes))
ax.axis('off')

# Add labels
ax.text(0, network_sizes[0] // 2, 'Neural Network 1', ha='center', va='center', fontsize=12)
ax.text(len(network_sizes) * 2 + 1, network_sizes[1] // 2, 'Pruning Step', ha='center', va='center', fontsize=12)
ax.text(len(network_sizes) * 2 + 3, network_sizes[2] // 2, 'Neural Network 2', ha='center', va='center', fontsize=12)

plt.title("Neural Network Transition with Pruning")

plt.savefig('{}/manuscript_fig_abstract.png'.format(output_root_dir),  bbox_inches='tight', dpi = 200) #
