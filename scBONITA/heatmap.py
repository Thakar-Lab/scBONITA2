import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def create_heatmap(path, title):
    data = []
    gene_names = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip().split(',')
            gene_name = line[0]
            time_data = [int(i) for i in line[1:]]
            data.append(time_data)
            gene_names.append(gene_name)
    
    num_genes = len(data)
    num_time_steps = len(data[0])

    # Adjusting the data to fit the provided shape
    data_array = np.array(data).reshape((num_genes, num_time_steps))

    # Create a custom colormap
    cmap = mcolors.ListedColormap(['grey', 'green'])
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create a heatmap
    plot = plt.figure(figsize=(12, 12))
    sns.heatmap(data_array, cmap='Greys', yticklabels=gene_names, xticklabels=True)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Genes')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    # plt.tight_layout()

    legend_elements = [
        Patch(facecolor='grey', edgecolor='grey', label='Gene Inactive'),
        Patch(facecolor='black', edgecolor='black', label='Gene Active')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Legend")


    plt.subplots_adjust(top=0.958, bottom=0.07, left=0.076, right=0.85, hspace=2, wspace=1)

    return plot

if __name__ == '__main__':

    atherosclerosis_main_path = 'scBONITA/attractor_analysis_output/atherosclerosis_attractors/hsa05166_attractors/'
    george_hiv_main_path = 'scBONITA/attractor_analysis_output/george_hiv_attractors/hsa04010_attractors/'


    create_heatmap(f'{george_hiv_main_path}/attractor_15/george_hiv_hsa04010_simulated_attractor_15.txt', 'George HIV hsa04010 attractor 15')
    create_heatmap(f'{george_hiv_main_path}/attractor_14/george_hiv_hsa04010_simulated_attractor_14.txt', 'George HIV hsa04010 attractor 14')
    create_heatmap(f'{george_hiv_main_path}/attractor_5/george_hiv_hsa04010_simulated_attractor_5.txt', 'George HIV hsa04010 attractor 5')




