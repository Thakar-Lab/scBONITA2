import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
import networkx as nx
import numpy as np
import pickle
import logging
import numexpr as ne
import random
from argparse import ArgumentParser

from user_input_prompts import *
from file_paths import file_paths


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    parser = ArgumentParser()

    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Name of the dataset to simulate'
    )

    parser.add_argument(
        '--network_name',
        type=str,
        required=True,
        help='Network name to simulate (e.g. hsa04670)'
    )

    parser.add_argument(
        '--num_cells',
        type=str,
        required=True,
        help='number of cells to simulate'
    )

    results = parser.parse_args()

    dataset_name = getattr(results, 'dataset_name')
    network_name = getattr(results, 'network_name')
    num_cells = getattr(results, 'num_cells')

    # Specifies the path to the correct network pickle file
    network_pickle_file = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/network_pickle_files/{dataset_name}_{network_name}.network.pickle'

    # Read in the network pickle file from the path
    network = pickle.load(open(network_pickle_file, 'rb'))

    # Convert the network's sparse dataset to a dense one
    dataset = network.dataset
    dense_dataset = np.array(dataset.todense())

    num_simulations = int(num_cells)
    with alive_bar(num_simulations) as bar:
        for i in range(num_simulations):
            # Select a random column from the network dataset
            cell_index = np.random.choice(dense_dataset.shape[1])

            # Reads in all of the rows for that columns
            # selected_column = np.array([random.choice([0,1]) for _ in dense_dataset[:, cell_index]])   #This code is picking random initial states. Comment it out if you are not doing this.
            selected_column = dense_dataset[:,cell_index] #This code initializes the states based on the expression.  
            
            # Transposes the list of gene expression into a column
            transposed_random_column = selected_column.reshape(-1,1)

            # Specify outfile path for the simulation results
            outfile_folder = f'{file_paths["trajectories"]}/{dataset_name}_{network_name}'
            png_folder = f'{outfile_folder}/png_files'
            text_folder = f'{outfile_folder}/text_files'

            os.makedirs(outfile_folder, exist_ok=True)
            os.makedirs(png_folder, exist_ok=True)
            os.makedirs(text_folder, exist_ok=True)
            
            # Simulate the network
            simulated_attractor = simulate_network(network.nodes, transposed_random_column)

            # Visualize the network simulation results
            fig = visualize_simulation(network.network, simulated_attractor, network, "False")

            # Save the attractor states to a csv file
            save_attractor_simulation(f'{text_folder}/cell_{cell_index}_trajectory.csv', network, simulated_attractor)
            plt.close(fig)

            # Create a heatmap of the expression for easier attractor visualization
            heatmap = create_heatmap(f'{text_folder}/cell_{cell_index}_trajectory.csv', f'Simulation for {dataset_name} {network_name} cell {cell_index} pathway ')
            # heatmap.show()

            # Saves a png of the results
            heatmap.savefig(f'{png_folder}/cell_{cell_index}_trajectory.png', format='png')
            plt.close(heatmap)
            bar()
