from importance_scores import CalculateImportanceScore
import logging
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.sparse import spmatrix, issparse
import matplotlib.pyplot as plt
import pickle
import os
import glob
from simulate_attractor import *
import argparse
import pickle

from setup.user_input_prompts import attractor_analysis_arguments
from file_paths import file_paths

def run_attractor_analysis(network, cells):
    """
    Runs the attractor analysis for the dataset and all of the networks.
    """
    # Set up the attractor analysis with the dataset and the parsed networks

    logging.info(f'\nNetwork: {network.name}')
    # Generate attractor
    logging.info(f'\tGenerating attractors...')

    network_attractors, simulated_dataset = generate_attractors(network.nodes, network.dataset)
    attractors_start = [attractor[:, 0] for cell, attractor in network_attractors.items()]

    # Convert the sparse dataset to a dense one
    if isinstance(simulated_dataset, spmatrix):
        dense_dataset = simulated_dataset.todense()
    else:
        dense_dataset = simulated_dataset
        
    # Generate Hamming distance matrix between cells and attractors
    logging.info(f'\tCalculating hamming distance between cells and attractors')
    logging.info(f'\tTransposed dataset shape: {dense_dataset.shape}')
    transposed_dataset = transpose_dataset(dense_dataset)
    num_nodes = len(network.nodes)
    num_cells = range(transposed_dataset.shape[0])
    num_attractors = range(len(attractors_start))

    logging.info(f'\t\tNodes: {num_nodes}')
    logging.info(f'\t\tCells: {transposed_dataset.shape[0]}')
    logging.info(f'\t\tAttractors: {len(attractors_start)}')
    cell_attractor_hamming_df = calculate_hamming_distance(num_attractors, num_cells, transposed_dataset, attractors_start)

    # Filter the attractors to only keep those that most closely match the expression of at least one cell
    filtered_attractor_indices = filter_attractors(cell_attractor_hamming_df)
    filtered_attractors = [attractors_start[i] for i in filtered_attractor_indices]
    num_filtered_attractors = range(len(filtered_attractors))

    # Create a distance matrix between each of the attractors
    logging.info(f'\tGenerating attractor distance matrix...')
    attractor_distance_matrix = calculate_hamming_distance(num_filtered_attractors, num_filtered_attractors, filtered_attractors, filtered_attractors)

    # Cluster the attractors using hierarchical agglomerative clustering
    logging.info(f'\tClustering the attractors...')
    clusters, cluster_fig = hierarchical_clustering(attractor_distance_matrix, len(network.nodes), show_plot=False)

    clustered_attractors = {}
    for i, cluster_num in enumerate(clusters):
        if cluster_num not in clustered_attractors:
            clustered_attractors[cluster_num] = {}
        clustered_attractors[cluster_num][filtered_attractor_indices[i]] = filtered_attractors[i]
    
    # Find the Hamming Distance between the cells and the attractors within the clusters
    logging.info(f'\tCalculating Hamming distance between cells and clustered attractors')

    # For each of the cells, find the hamming distance to the best cell
    logging.info(f'Number of cells in the full dataset: {network.dataset.shape[1]}')
    if issparse(network.dataset):
        full_dataset = network.dataset.toarray().T
    else:
        full_dataset = network.dataset.T

    cell_map = {}
    for cell_num, cell in enumerate(full_dataset):
        cell_map[cell_num] = {}

    for cluster, attractors in clustered_attractors.items():
        logging.debug(f'\nCluster {cluster}')

        num_filtered_attractors = range(len(attractors.values()))

        mapped_cluster_attractors = calculate_hamming_distance(num_filtered_attractors, num_cells, attractors.values(), transposed_dataset)

        # transposed_mapped_cluster_attractors = transpose_dataset(mapped_cluster_attractors)
        logging.debug(f'Mapped cluster attractors:\n {mapped_cluster_attractors}')

        # Map the cells to the attractors within the clusters
        filtered_attractor_indices = list(clustered_attractors[cluster].keys())
        logging.debug(f'Filtered attractor indices: {filtered_attractor_indices}')
        mapped_cluster_attractors.index = filtered_attractor_indices # type: ignore
        logging.debug(f'Cluster {cluster} mapped attractors')
        logging.debug(mapped_cluster_attractors)

        # Find the total Hamming distance between each attractor and all of the cells
        total_hamming_distance = mapped_cluster_attractors.sum(axis=1)
        min_sum_index = total_hamming_distance.idxmin()

        representative_attractor = attractors[min_sum_index]

        # Find the hamming distance for each cell for each cluster
        for cell_num, cell in enumerate(full_dataset):
            hamming_distance = 0
            for gene_num, gene in enumerate(cell):
                if cell[gene_num] != representative_attractor[gene_num]:
                   hamming_distance += 1
            
            cell_map[cell_num][cluster] = hamming_distance
    

        logging.debug(f'Representative attractor for cluster {cluster} (index {min_sum_index}):')
        logging.debug(f'\n{attractors[min_sum_index]}')

        network.representative_attractors[cluster] = representative_attractor
    
    # Calculate which cluster each cell should map to
    min_hamming_cluster = {}
    for cell_num, clusters in cell_map.items():

        # Find the cluster with the minimum Hamming distance
        min_cluster = min(clusters, key=clusters.get)
        min_hamming_cluster[cell_num] = min_cluster
        cell_object = cells[cell_num]

    network.cell_map = min_hamming_cluster

    for cell_object in cells:
        # Add the best attractor for each cell to that cell object
        cell_object.attractor_dict[network.name] = min_hamming_cluster[cell_object.index]

    return cluster_fig

def generate_attractors(nodes, dataset):
    """
    Uses the vectorized simulation function from the importance score calculations to find
    the attractors for the network.
    """

    calculate_importance_score = CalculateImportanceScore(nodes, dataset)

    simulated_dataset = calculate_importance_score.dataset        
    network_attractors = calculate_importance_score.vectorized_run_simulation()

    return network_attractors, simulated_dataset


def extract_node_rows(dataset, nodes):
    """
    Create a dense dataset only containing the nodes that are present in the network from a 
    sparse dataset
    """

    # Convert the sparse dataset to a dense one
    if isinstance(dataset, spmatrix):
        dense_dataset = dataset.todense()
    else:
        dense_dataset = dataset

    # Find the row indices in the dataset for the nodes in the network
    node_indices = []
    for node in nodes:
        if node.dataset_index not in node_indices:
            node_indices.append(node.dataset_index)
        
    return dense_dataset[node_indices]

def transpose_dataset(dataset):
    """
    Converts a dataset to a numpy array and transposes it. Used to order the nodes as columns and 
    the cells as rows.
    """

    try:
        numpy_dataset = np.array(dataset).squeeze() # Get rid of extra dimensions
        logging.info(f'\t\tExtra dimension in dataset, squeezing...')
    except:
        numpy_dataset = np.array(dataset)

    transposed_dataset = np.transpose(numpy_dataset)

    logging.info(f'\t\tTransposed dataset shape: {transposed_dataset.shape}')

    return transposed_dataset

def calculate_hamming_distance(df_index, df_columns, list_1, list_2):
    """
    Iterate through two lists and calculate the Hamming distance between the items, returns
    a matrix of the Hamming distances with list_1 as the columns and list_2 as the rows.
    """

    # Convert lists to numpy arrays for vectorized operations    
    array1 = np.array([list(item) for item in list_1])
    array2 = np.array([list(item) for item in list_2])
    
    # Calculate the hamming distances using broadcasting and vectorization
    # The distances array will be a 2D array where the element at [i, j] is the
    # Hamming distance between array1[i] and array2[j]
    distances = np.sum(array1[:, None, :] != array2[None, :, :], axis=2)
    
    # Create the DataFrame from the distances array
    passed = False

    reduced_dimension = -1
    try:
        df = pd.DataFrame(distances, index=df_index, columns=df_columns)

    # This fixed and issue where the index and columns were not the same dimension, I'm not sure how but I'm leaving it
    except ValueError:
        while passed == False:
            try:
                df_columns_adjusted = df_columns[:reduced_dimension]
                logging.info(f'df_columns_adjusted = {df_columns_adjusted}')
                df = pd.DataFrame(distances, index=df_index, columns=df_columns_adjusted)
                passed = True
            except:
                reduced_dimension -= 1

    return df

def filter_attractors(dataframe):
    """
    Filters a Hamming distance dataframe to extract only the attractors with a minimum
    value for at least one cell, excludes attractors that do not best explain any cells.
    """

    # Ensure that each fo the columns contain numeric values
    for column in dataframe.columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')

    # Exclude the first row and first column
    dataframe_sliced = dataframe.iloc[1:, 1:]

    # Find the row indices with the minimum Hammin distance for each column
    min_indices = dataframe_sliced.idxmin()

    # Keep only unique row indices
    unique_min_indices = min_indices.unique()

    # Filter out rows that do not contain a minimum Hamming distance for any cell
    filtered_df = dataframe.loc[unique_min_indices]

    # Find the attractor indices for these rows
    filtered_attractor_indices = filtered_df.index.tolist()

    return filtered_attractor_indices

def hierarchical_clustering(distance_matrix, num_nodes, show_plot):
    """
    Clusters the distance matrix of the attractors using Hierarchical Agglomerative Clustering.
    The cutoff for the clusters is set as <=25% the number of genes in the network. Returns the
    clusters and allows for a dendrogram to be displayed.
    """

    # Cluster the attractors using Hierarchical Agglomerative Clustering
    # Convert the distance matrix to a numpy array
    distance_matrix = distance_matrix.to_numpy()

    # Convert the square form distance matrix to a condensed form
    condensed_distance_matrix = ssd.squareform(distance_matrix)

    # Perform Hierarchical Agglomerative Clustering
    # 'ward' is one of the methods for calculating the distance between the newly formed cluster
    Z = sch.linkage(condensed_distance_matrix, method='ward')

    cutoff = 0.25 * num_nodes
    clusters = sch.fcluster(Z, t=cutoff, criterion='distance')

    logging.info(f'\t\tClustering cutoff value = {round(cutoff, 2)} (<=20% * number of genes {num_nodes})')
    
    # Plotting the dendrogram
    fig = plt.figure(figsize=(10, 7))
    plt.title("Attractors Hierarchical Clustering Dendrogram")
    
    dendrogram = sch.dendrogram(Z)
    plt.axhline(y=cutoff, color = 'r', linestyle='--')

    plt.ylabel('Distance')
    plt.xlabel('Attractors')

    if show_plot:
        plt.show()
    
    return clusters, fig



# If you want to run the attractor analysis by itself
if __name__ == '__main__':

    # Set the logging level for output
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Allow the user to either add in the dataset name and network name from the command line or as a prompt
    parser = argparse.ArgumentParser()

    dataset_name, show_simulation = attractor_analysis_arguments(parser)

    # Load the network pickle files
    all_networks = []

    # Load the cell objects for the dataset
    cell_population = pickle.load(open(f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/cells_pickle_file/{dataset_name}.cells.pickle', "rb"))
    cells = cell_population.cells

    logging.info(f'\nRunning attractor analysis for all networks...')
    pickle_file_path = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/network_pickle_files/'
    for pickle_file in glob.glob(pickle_file_path + str(dataset_name) + "_" + "*" + ".network.pickle"):
        if pickle_file:
            logging.info(f'\t\tLoading data file: {pickle_file}')
            network = pickle.load(open(pickle_file, "rb"))
            all_networks.append(network)
        else:
            assert FileNotFoundError("Network pickle file not found")
    
    try:
        dataset_found = all_networks[0].name
        logging.info(f'\nFound dataset!')

    except IndexError:
        error_message = "No networks loaded, check to make sure network pickle files exist in 'pickle_files/'"
        logging.error(error_message)
        raise Exception(error_message)
    
    for network in all_networks:
        dataset = network.dataset
        # Run the pathway analysis for each of the networks

        logging.info(f'\n----- ATTRACTOR ANALYSIS -----')
        cluster_fig = run_attractor_analysis(network, cells)
        cluster_path = f'{file_paths["attractor_analysis_output"]}/{dataset_name}_attractors'
        os.makedirs(cluster_path, exist_ok=True)

        cluster_filename = f'{cluster_path}/{network.name}_cluster.png'

        cluster_fig.savefig(cluster_filename, bbox_inches='tight')
        plt.close(cluster_fig)  # Close the figure after saving


        logging.info(f'\n-----PATHWAY ANALYSIS RESULTS -----')
        logging.info(f'\nNETWORK {network.name.upper()}')

        attractor_counts = {}
        total_cells = 0
        for cell in network.cell_map.keys():
            attractor = network.cell_map[cell]
            if attractor not in attractor_counts:
                attractor_counts[attractor] = 1
            else:
                attractor_counts[attractor] += 1
            total_cells += 1

        for attractor_num, num_cells in attractor_counts.items():
            logging.info(f'\tAttractor {attractor_num} contains {num_cells} cells ({round(num_cells / total_cells * 100, 3)}%)')

        for attractor_num, representative_attractor in network.representative_attractors.items():
            if attractor_num in attractor_counts:

                # Create a dataset- and network-specific directory for the output
                output_directory = f'{file_paths["attractor_analysis_output"]}/{dataset_name}_attractors/{network.name}_attractors/attractor_{str(attractor_num)}'
                os.makedirs(output_directory, exist_ok=True)

                filename = f'{dataset_name}_{network.name}_attractor_{attractor_num}.txt'
                np.savetxt(output_directory + "/" + filename, representative_attractor.T, fmt='%d')

                logging.info(f'\tSaved representative attractor {attractor_num}')
                logging.debug(f'\t{representative_attractor}')
                
                simulation_results = simulate_network(network.nodes, output_directory + "/" + filename)

                svg_output_path = f'{output_directory}/attractor_{attractor_num}_simulation_results.svg'
                png_output_path = f'{output_directory}/attractor_{attractor_num}_simulation_results.png'

                fig = visualize_simulation(network.network, simulation_results, network, show_simulation)

                plt.savefig(svg_output_path, format='svg')
                plt.savefig(png_output_path, format='png')
                plt.close(fig)  # Close the figure after saving

        logging.info(f'\n\tSaved attractor analysis results to "attractor_analysis_output/{dataset_name}_attractors/{network.name}_attractors')


        logging.info(f'\nAdding representative attractor map to network pickle files:')
        network_directory_path = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/network_pickle_files'
        os.makedirs(network_directory_path, exist_ok=True)

        network_pickle_file_path = f'{network_directory_path}/{dataset_name}_{network.name}.network.pickle'
        logging.info(f'\tFile: {dataset_name}_{network.name}.network.pickle')
        pickle.dump(network, open(network_pickle_file_path, "wb"))

        cell_pickle_file_path = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/cells_pickle_file'
        os.makedirs(cell_pickle_file_path, exist_ok=True)

        cells_pickle_file = f'{dataset_name}.cells.pickle'
        with open(f'{cell_pickle_file_path}/{cells_pickle_file}', "wb") as file:
            pickle.dump(cell_population, file)