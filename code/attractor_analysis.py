# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.sparse import spmatrix, issparse
import matplotlib.pyplot as plt
import pickle
import os
import glob
import argparse
import pickle
import numexpr as ne
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fastdtw import fastdtw
import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, inconsistent
from kneed import KneeLocator 
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from alive_progress import alive_bar
import statistics
import multiprocessing as mp
from itertools import combinations
import time
from sklearn.cluster import KMeans
from functools import lru_cache
from scipy.spatial.distance import euclidean

from user_input_prompts import attractor_analysis_arguments
from file_paths import file_paths

# Set your Euclidean distance threshold
EUCLIDEAN_THRESHOLD = 5  # Adjust this value based on desired precision

def vectorized_run_simulation(nodes: list, cell_column: list):
    steps = 20

    # Convert cell_column to NumPy array for faster processing
    starting_state = np.array(cell_column)

    # Preallocate simulation state matrix (steps, nodes)
    total_simulation_states = np.zeros((steps, len(nodes)), dtype=int)

    def evaluate_expression(data, expression):
        expression = expression.replace('and', '&').replace('or', '|').replace('not', '~')
        if any(op in expression for op in ['&', '|', '~']):
            local_vars = {key: np.array(value).astype(bool) for key, value in data.items()}
        else:
            local_vars = {key: np.array(value) for key, value in data.items()}
        return ne.evaluate(expression, local_dict=local_vars)

    # Run the simulation
    for step in range(steps):
        step_expression = []

        # Iterate through each node in the network
        for node_idx, node in enumerate(nodes):
            data = {}
            incoming_node_indices = [predecessor_index for predecessor_index in node.predecessors]

            # Get the rows in the dataset for the incoming nodes
            if step == 0:
                if len(incoming_node_indices) > 0:
                    data['A'] = starting_state[incoming_node_indices[0]]
                if len(incoming_node_indices) > 1:
                    data['B'] = starting_state[incoming_node_indices[1]]
                if len(incoming_node_indices) > 2:
                    data['C'] = starting_state[incoming_node_indices[2]]
                if len(incoming_node_indices) > 3:
                    data['D'] = starting_state[incoming_node_indices[3]]
            else:
                prev_state = total_simulation_states[step - 1]
                if len(incoming_node_indices) > 0:
                    data['A'] = prev_state[incoming_node_indices[0]]
                if len(incoming_node_indices) > 1:
                    data['B'] = prev_state[incoming_node_indices[1]]
                if len(incoming_node_indices) > 2:
                    data['C'] = prev_state[incoming_node_indices[2]]
                if len(incoming_node_indices) > 3:
                    data['D'] = prev_state[incoming_node_indices[3]]

            next_step_node_expression = evaluate_expression(data, node.calculation_function)

            # Save the expression for the node for this step
            step_expression.append(next_step_node_expression)

        # Convert step expression to integer and save the result
        total_simulation_states[step] = np.array(step_expression).astype(int)

    # Return the final matrix of simulation steps where rows are steps and columns are nodes
    return total_simulation_states.tolist()


def simulate_single_cell(cell_index: int, cell_name: str, dataset_array: np.ndarray, network: object, dataset_name: str, network_name: str, lock):
    """
    Simulates a single cell trajectory.

    Parameters
    ---------
    cell_index : int
        Index of the cell to simulate (column index in dataset_array)
    cell_name : str
        Barcode of the cell, used for output file naming
    dataset_array : np.ndarray
        Dense dataset for the cells
    network : object
        The network object containing the nodes
    dataset_name : str
        The name of the dataset
    network_name : str
        The name of the network
    """
    # Reads in the network gene expression for the chosen cell column
    cell_starting_state = np.array([int(gene_expr) for gene_expr in dataset_array[:, cell_index]])

    # Simulate the network using the expression in the selected column as the starting state
    trajectory = vectorized_run_simulation(network.nodes, cell_starting_state)

    # Save the attractor simulation to a csv file, named by barcode
    attractor_sim_path = f'{file_paths["trajectories"]}/{dataset_name}_{network_name}/text_files/cell_trajectories/cell_{cell_name}_trajectory.csv'
    png_dir = f'{file_paths["trajectories"]}/{dataset_name}_{network_name}/png_files/cell_trajectories'
    os.makedirs(png_dir, exist_ok=True)

    with lock:
        if not os.path.exists(attractor_sim_path):
            with open(attractor_sim_path, 'w') as file:
                trajectory = np.array(trajectory).T
                for gene_num, expression in enumerate(trajectory):
                    file.write(f'{network.nodes[gene_num].name},{",".join([str(i) for i in list(expression)])}\n')

    if len(os.listdir(png_dir)) <= 100:
        with lock:
            if os.path.exists(attractor_sim_path):
                heatmap = create_heatmap(
                    attractor_sim_path,
                    f'Simulation for {dataset_name} {network_name} cell {cell_name} pathway')
                heatmap_path = f'{png_dir}/cell_{cell_name}_trajectory.png'
                heatmap.savefig(heatmap_path, format='png')
                plt.close(heatmap)
            else:
                logging.warning(f'Trajectory file for cell {cell_name} not found. Heatmap not created.')


def simulate_single_cell_wrapper(args):
    """
    Wrapper to call simulate_single_cell and return 1 for progress tracking.
    This function needs to be in the global scope to be pickled by multiprocessing.
    """
    cell_index, cell_name, dataset_array, network, dataset_name, network_name, lock = args
    simulate_single_cell(cell_index, cell_name, dataset_array, network, dataset_name, network_name, lock)
    return 1


def simulate_cells(index_to_name: dict, dataset_array: np.ndarray, cells_to_simulate: list, num_simulations: int, network: object, dataset_name: str, network_name: str):
    """
    Simulates the cell trajectories using a network model with parallel processing.

    Parameters
    ---------
    index_to_name : dict
        Mapping from column index to cell barcode, built from network.cell_names
    dataset_array : np.ndarray
        The dense dataset for the cells
    cells_to_simulate : list
        The list of cell indices to simulate to ensure the same cells are simulated across networks
    num_simulations : int
        The number of simulations to run
    network : object
        The network object containing the nodes
    dataset_name : str
        The name of the dataset
    network_name : str
        The name of the network

    Returns
    -------
    cells_to_simulate : list
        The list of cell indices that were simulated for this network
    """
    logging.info(f'\tSimulating {num_simulations} cell trajectories')

    trajectory_dir = f'{file_paths["trajectories"]}/{dataset_name}_{network_name}/text_files/cell_trajectories'
    existing_files = os.listdir(trajectory_dir)

    # Find already-simulated barcodes from existing filenames
    existing_names = set()
    for filename in existing_files:
        if filename.startswith("cell_") and filename.endswith("_trajectory.csv"):
            existing_names.add(filename[len("cell_"):-len("_trajectory.csv")])

    # Find indices whose barcodes have not yet been simulated
    all_indices = set(range(dataset_array.shape[1]))
    indices_to_simulate = [i for i in all_indices if index_to_name.get(i, str(i)) not in existing_names]

    if num_simulations > len(indices_to_simulate):
        logging.warning(
            f"Requested {num_simulations} simulations, but only {len(indices_to_simulate)} unsimulated cells available.")
        num_simulations = len(indices_to_simulate)

    if len(cells_to_simulate) == 0:
        cells_to_simulate = np.random.choice(indices_to_simulate, num_simulations, replace=False).tolist()

    pool = mp.Pool(mp.cpu_count())
    manager = mp.Manager()
    lock = manager.Lock()

    arguments = [
        (cell_index, index_to_name.get(cell_index, str(cell_index)), dataset_array, network, dataset_name, network_name, lock)
        for cell_index in cells_to_simulate
    ]

    with alive_bar(num_simulations) as bar:
        results = pool.imap_unordered(simulate_single_cell_wrapper, arguments)
        for result in results:
            bar()
        pool.close()
        pool.join()

    return cells_to_simulate


def create_heatmap(trajectory_path: str, title: str):
    """
    Creates a trajectory figure using an sns heatmap
    """
    data = []
    gene_names = []

    with open(trajectory_path, 'r') as trajectory_file:
        for line in trajectory_file:
            line = line.strip().split(',')
            gene_name = line[0]
            time_data = [int(i) for i in line[1:]]
            data.append(time_data)
            gene_names.append(gene_name)

    num_genes = len(data)
    num_time_steps = len(data[0])

    data_array = np.array(data).reshape((num_genes, num_time_steps))

    plot = plt.figure(figsize=(12, 12))
    sns.heatmap(data_array, cmap='Greys', yticklabels=gene_names, xticklabels=True, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Genes')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    legend_elements = [
        Patch(facecolor='grey', edgecolor='grey', label='Gene Inactive'),
        Patch(facecolor='black', edgecolor='black', label='Gene Active')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Legend")
    plt.subplots_adjust(top=0.958, bottom=0.07, left=0.076, right=0.85, hspace=2, wspace=1)

    return plot


def downsample(ts, factor=2):
    """Reduces the length of the time series by the downsampling factor."""
    return np.array(ts[::factor]).flatten()

@lru_cache(maxsize=None)
def cached_dtw(ts1_tuple, ts2_tuple, radius=1):
    """Cached DTW calculation with a custom pointwise distance function."""
    ts1 = np.array(ts1_tuple).flatten()
    ts2 = np.array(ts2_tuple).flatten()
    distance, _ = fastdtw(ts1, ts2, radius=radius, dist=lambda x, y: (x - y) ** 2)
    return distance

def compute_dtw_distance_pair(cell1: str, cell2: str, cell_trajectory_dict: dict, radius=1, downsample_factor=2):
    distances = {}
    for gene, ts1 in cell_trajectory_dict[cell1].items():
        if gene in cell_trajectory_dict[cell2]:
            ts2 = cell_trajectory_dict[cell2][gene]
            ts1_downsampled = downsample(np.array(ts1), downsample_factor)
            ts2_downsampled = downsample(np.array(ts2), downsample_factor)
            squared_diff = np.sum((ts1_downsampled - ts2_downsampled) ** 2)
            if squared_diff < EUCLIDEAN_THRESHOLD:
                distances[gene] = squared_diff
            else:
                dtw_distance = cached_dtw(tuple(ts1_downsampled), tuple(ts2_downsampled), radius)
                distances[gene] = dtw_distance
    total_distance = sum(distances.values()) if distances else float('inf')
    return (cell1, cell2, total_distance)

def compute_dtw_distances(cell_trajectory_dict: dict):
    dtw_distances = {}
    cell_names = list(cell_trajectory_dict.keys())
    cell_pairs = list(combinations(cell_names, 2))
    num_cpus = min(8, mp.cpu_count())
    with mp.Pool(num_cpus) as pool:
        tasks = [(cell1, cell2, cell_trajectory_dict) for cell1, cell2 in cell_pairs]
        results = pool.starmap(compute_dtw_distance_pair, tasks)
        for cell1, cell2, total_distance in results:
            dtw_distances[(cell1, cell2)] = total_distance
    return dtw_distances


def create_distance_matrix(dtw_distances: dict, file_names: list):
    distance_matrix = np.zeros((len(file_names), len(file_names)))
    for (file1, file2), total_distance in dtw_distances.items():
        i = file_names.index(file1)
        j = file_names.index(file2)
        distance_matrix[i, j] = total_distance
        distance_matrix[j, i] = total_distance
    return distance_matrix


def hierarchical_clustering(dtw_distances: dict, num_clusters: int = 2):
    cells = set()
    for (cell1, cell2), _ in dtw_distances.items():
        cells.add(cell1.split('_trajectory')[0])
        cells.add(cell2.split('_trajectory')[0])
    cells = sorted(cells)

    distance_matrix = pd.DataFrame(np.inf, index=cells, columns=cells)
    for (cell1, cell2), distance in dtw_distances.items():
        distance_matrix.at[cell1.split('_trajectory')[0], cell2.split('_trajectory')[0]] = distance
        distance_matrix.at[cell2.split('_trajectory')[0], cell1.split('_trajectory')[0]] = distance

    distance_array = distance_matrix.values[np.triu_indices_from(distance_matrix, k=1)]
    # Replace inf/nan with the maximum finite value before sqrt to ensure linkage gets finite values
    finite_mask = np.isfinite(distance_array)
    if not finite_mask.all():
        finite_max = distance_array[finite_mask].max() if finite_mask.any() else 1.0
        logging.warning(f'hierarchical_clustering: {(~finite_mask).sum()} non-finite values in distance array, replacing with max finite value {finite_max:.4f}')
        distance_array = np.where(finite_mask, distance_array, finite_max)
    distance_array = np.sqrt(np.maximum(distance_array, 0.0))
    Z = linkage(distance_array, method='ward')

    plt.figure(figsize=(8, 10))
    dendrogram(Z, labels=distance_matrix.index, orientation='top')
    plt.title('Hierarchical Clustering Dendrogram', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=8, rotation=90)
    plt.xlabel('Chunk:Cluster', fontsize=8)
    plt.ylabel('Distance', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{file_paths["trajectories"]}/{dataset_name}_{network_name}/png_files/dendrogram.png')

    if num_clusters == 0:
        finite_max = distance_matrix[distance_matrix != np.inf].max().max()
        distance_matrix = distance_matrix.replace(np.inf, finite_max)
        mds = MDS(n_components=2, dissimilarity="precomputed", n_init=4, n_jobs=4, normalized_stress='auto')
        mds_coordinates = mds.fit_transform(distance_matrix)
        inertia = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(mds_coordinates)
            inertia.append(kmeans.inertia_)
        kneedle = KneeLocator(k_range, inertia, S=1.0, curve="convex", direction="decreasing")
        num_clusters = kneedle.elbow
        logging.info(f'\t\tUsing {num_clusters} clusters (determined by K-Means clustering)')

    clusters = fcluster(Z, num_clusters, criterion='maxclust')
    cluster_dict = {}
    for cell, cluster_id in zip(cells, clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(cell)

    plt.close()
    return cluster_dict, num_clusters


def summarize_clusters(directory: str, cell_names: list):
    gene_expr_dict = {}
    trajectory_files = []
    for filename in os.listdir(directory):
        if filename.endswith("_trajectory.csv"):
            trajectory_files.append(os.path.join(directory, filename))

    files_to_open = []
    for cell in cell_names:
        for file_name in trajectory_files:
            if cell.split("_")[1] == file_name.split("_")[-2]:
                files_to_open.append(file_name)

    for file_path in files_to_open:
        with open(file_path, 'r') as sim_file:
            for line in sim_file:
                line = line.strip().split(',')
                gene_name = line[0]
                gene_expression = [int(i) for i in line[1:]]
                if gene_name not in gene_expr_dict:
                    gene_expr_dict[gene_name] = []
                gene_expr_dict[gene_name].append(gene_expression)

    gene_avg_expr = {}
    for gene, simulation_results in gene_expr_dict.items():
        if gene not in gene_avg_expr:
            gene_avg_expr[gene] = []
        transposed_data = list(map(list, zip(*simulation_results)))
        for i in transposed_data:
            gene_avg_expr[gene].append(statistics.mean(i))

    avg_expr_df = pd.DataFrame(gene_avg_expr)
    avg_expr_df = avg_expr_df.transpose()
    return avg_expr_df


def plot_average_trajectory(df: pd.DataFrame, title: str, path: str):
    plt.figure(figsize=(8, 10))
    sns.heatmap(df, cmap='Greys', yticklabels=True, vmin=0, vmax=1)
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel='Simulation Time Steps', fontsize=12)
    plt.ylabel(ylabel='Gene', fontsize=12)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def create_trajectory_chunks(num_chunks: int, num_clusters: int, output_directory: str):
    trajectory_files_parsed = []
    cell_txt_traj_dir = f'{output_directory}/cell_trajectories'
    cluster_chunks = {}
    cells_in_chunks = {}
    logging.info(f'\t\tCreating chunks, this may take a while depending on how many cells and the chunk size')

    for chunk in range(num_chunks):
        start = time.time()
        cell_trajectory_dict = {}
        num_cells_parsed = 0

        for traj_filename in os.listdir(cell_txt_traj_dir):
            if num_cells_parsed <= num_cells_per_chunk-1 and traj_filename.endswith("_trajectory.csv") and traj_filename not in trajectory_files_parsed:
                trajectory_files_parsed.append(traj_filename)
                filepath = os.path.join(cell_txt_traj_dir, traj_filename)
                df = pd.read_csv(filepath, header=None)
                df.columns = ['Gene'] + [f'Time{i}' for i in range(1, df.shape[1])]
                df.set_index('Gene', inplace=True)
                cell_trajectory_dict[traj_filename] = {gene: df.loc[gene].values for gene in df.index}
                num_cells_parsed += 1

        dtw_distances = compute_dtw_distances(cell_trajectory_dict)

        if chunk == 0:
            with open(f'{output_directory}/distances.csv', 'w') as outfile:
                for (cell1, cell2), total_distance in dtw_distances.items():
                    outfile.write(f'{cell1},{cell2},{total_distance}\n')
        else:
            with open(f'{output_directory}/distances.csv', 'a') as outfile:
                for (cell1, cell2), total_distance in dtw_distances.items():
                    outfile.write(f'{cell1},{cell2},{total_distance}\n')

        cluster_dict, num_clusters = hierarchical_clustering(dtw_distances, num_clusters)

        for cluster, cell_list in cluster_dict.items():
            df = summarize_clusters(cell_txt_traj_dir, cell_list)
            df_binarized = pd.DataFrame(np.where(df < 0.5, 0, 1), index=df.index, columns=df.columns)
            cluster_chunks[f'{chunk}:{cluster}'] = df_binarized
            if chunk not in cells_in_chunks:
                cells_in_chunks[chunk] = {}
            cells_in_chunks[chunk][cluster] = cell_list

        end = time.time()
        length = end - start
        if chunk > 0:
            print(f'\t\tCreating chunk {chunk+1} / {num_chunks} (Est remaining: {round(length * (num_chunks - chunk))}s)')

    return cluster_chunks, cells_in_chunks, num_clusters


def cluster_cells(num_files: int, output_directory: str, num_cells_per_chunk: int):
    num_clusters: int = 0
    num_chunks: int = round(num_files / (num_cells_per_chunk))
    logging.info(f'\t\tCreating {num_chunks} chunks ({num_files} cells / {num_cells_per_chunk} cells per chunk)')
    if num_chunks == 0:
        num_chunks = 1
    group_dict = {}

    cluster_chunks, cells_in_chunks, num_clusters = create_trajectory_chunks(num_chunks, num_clusters, output_directory)

    logging.info(f'\t\tComparing chunks')
    chunk_dtw_distances = compute_dtw_distances(cluster_chunks)
    chunk_names: list = list(cluster_chunks.keys())
    distance_matrix: np.ndarray = create_distance_matrix(chunk_dtw_distances, chunk_names)
    group_cluster_dict, num_clusters = hierarchical_clustering(chunk_dtw_distances, num_clusters)

    condensed_distance_matrix = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distance_matrix, method='average')
    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    reordered_matrix = distance_matrix[np.ix_(order, order)]
    reordered_file_names = [i for i in order]

    plt.figure(figsize=(8, 9))
    sns.heatmap(data=reordered_matrix, xticklabels=reordered_file_names, yticklabels=reordered_file_names, cmap='Greys', annot=False)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.title("DTW Distance Heatmap")
    plt.tight_layout()
    plt.savefig(f'{file_paths["trajectories"]}/{dataset_name}_{network_name}/png_files/distance_heatmap')
    plt.close()

    cells_in_cluster: dict = {}
    for cluster, cell_list in group_cluster_dict.items():
        num_cells_in_cluster: int = 0
        for chunk_cluster in cell_list:
            chunk: str = chunk_cluster.split(':')[0]
            cluster: str = chunk_cluster.split(':')[1]
            num_cells_in_cluster += len(cells_in_chunks[int(chunk)][int(cluster)])
            if cluster not in cells_in_cluster:
                cells_in_cluster[cluster] = []
            cells_in_cluster[cluster].extend(cells_in_chunks[int(chunk)][int(cluster)])

    for cluster, cluster_group in group_cluster_dict.items():
        logging.info(f'\t\tSummarizing cluster {cluster}')
        df: pd.DataFrame = summarize_clusters(f'{output_directory}/cell_trajectories', cells_in_cluster[str(cluster)])

        os.makedirs(f'{file_paths["trajectories"]}/{dataset_name}_{network_name}/png_files/cluster_summaries', exist_ok=True)
        os.makedirs(f'{file_paths["trajectories"]}/{dataset_name}_{network_name}/text_files/cluster_summaries', exist_ok=True)

        title: str = f'Average Gene Expression Heatmap for Cluster {cluster}'
        path: str = f'{file_paths["trajectories"]}/{dataset_name}_{network_name}/png_files/cluster_summaries/cluster_{cluster}_summary'

        df.to_csv(f'{file_paths["trajectories"]}/{dataset_name}_{network_name}/text_files/cluster_summaries/cluster_{cluster}_summary.csv', header=False)
        plot_average_trajectory(df, title, path)

        groups = []
        pickle_file_path = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/network_pickle_files/{dataset_name}_*_pickle_files/'
        for path in glob.glob(pickle_file_path):
            for file in os.listdir(path):
                file_path = f'{path}{file}'
                with open(file_path, 'rb') as f:
                    group_network = pickle.load(f)
                    groups.append((group_network, file_path))

        cell_nums = [str(cell.split('_')[1]) for cell in cells_in_cluster[str(cluster)]]

        cells_seen = []
        for group_network, file_path in groups:
            for cell in group_network.cells:
                cell_str = str(cell).strip()
                if cell_str in cell_nums and cell_str not in cells_seen:
                    cells_seen.append(cell_str)
                    group_name = group_network.name.split('_')[1]
                    if network_name not in group_dict:
                        group_dict[network_name] = {}
                    if cluster not in group_dict[network_name]:
                        group_dict[network_name][cluster] = {}
                    if not group_name in group_dict[network_name][cluster]:
                        group_dict[network_name][cluster][group_name] = 0
                    group_dict[network_name][cluster][group_name] += 1
                    cell_group_dict[cell] = group_name
                    if cell not in cell_cluster_dict:
                        cell_cluster_dict[cell] = 0
                    cell_cluster_dict[cell] = cluster
            with open(file_path, 'wb') as f:
                pickle.dump(group_network, f)

    with open(f'{file_paths["trajectories"]}/{dataset_name}_{network_name}/text_files/group_data.csv', 'w') as file:
        file.write(f'cluster,group,num_cells\n')
        for network in group_dict:
            logging.info(f'\nNetwork: {network}')
            for cluster, groups in group_dict[network].items():
                logging.info(f'\tCluster {cluster}')
                for group, num_cells in groups.items():
                    logging.info(f'\t\t{group}: {num_cells} cells')
                    file.write(f'{cluster},{group},{num_cells}\n')


def create_cluster_combinations(cell_clusters: dict, cell_group_dict: dict):
    rows = []
    all_networks = set()
    for cell in cell_clusters:
        all_networks.update(cell_clusters[cell].keys())
    for cell in cell_clusters:
        row = {'Cell': cell, 'Group': cell_group_dict[cell]}
        for network in all_networks:
            row[network] = cell_clusters[cell].get(network, None)
        rows.append(row)
    df = pd.DataFrame(rows)
    network_columns = [col for col in df.columns if col.startswith('hsa')]
    grouped = df.groupby(network_columns + ['Group']).size().reset_index(name='Count')
    logging.info(f'\nCluster combinations for the different networks')
    logging.info(grouped)
    grouped.to_csv(f'{file_paths["trajectories"]}/{dataset_name}_cluster_by_group.csv')
    df.to_csv(f'{file_paths["trajectories"]}/{dataset_name}_cell_groups.csv')


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    dataset_name, num_cells_per_chunk, num_cells_to_analyze = attractor_analysis_arguments(parser)

    # Load only the network pickle files - no cell population pickle needed
    all_networks = []
    logging.info(f'\nRunning attractor analysis for all networks...')
    pickle_file_path = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/network_pickle_files/'
    for pickle_file in glob.glob(pickle_file_path + str(dataset_name) + "_" + "*" + ".network.pickle"):
        if pickle_file:
            logging.info(f'\tLoading data file: ...{pickle_file[-50:]}')
            network = pickle.load(open(pickle_file, "rb"))
            all_networks.append(network)
        else:
            assert FileNotFoundError("Network pickle file not found")

    try:
        successfully_loaded = all_networks[0].name
        logging.info(f'\nFound dataset!')
    except IndexError:
        error_message = "No networks loaded, check to make sure network pickle files exist in 'pickle_files/'"
        logging.error(error_message)
        raise Exception(error_message)

    cell_clusters = {}
    cells_to_simulate = []
    cell_group_dict = {}
    cell_cluster_dict = {}
    logging.info(f'\n----- ATTRACTOR ANALYSIS -----')

    for network in all_networks:
        logging.info(f'\tANALYZING NETWORK: {network.name}')

        # Convert the network's sparse dataset to a dense array
        dataset = network.dataset
        dense_dataset: np.ndarray = np.array(dataset.todense())
        num_cells_in_dataset = dense_dataset.shape[1]
        logging.info(f'\tThere are {num_cells_in_dataset} cells in dataset')
        network_name: str = network.name

        # Build index -> barcode mapping from network.cell_names, which is ordered
        # to match the columns of network.dataset
        cell_names = getattr(network, 'cell_names', None)
        if cell_names is None:
            logging.warning(f'\tnetwork.cell_names is MISSING (None) - the network pickle was likely saved before cell_names was added. Falling back to integer indices.')
            index_to_name = {i: str(i) for i in range(num_cells_in_dataset)}
        elif len(cell_names) != num_cells_in_dataset:
            logging.warning(
                f'\tnetwork.cell_names LENGTH MISMATCH - '
                f'network.cell_names has {len(cell_names)} entries but dataset has {num_cells_in_dataset} columns. '
                f'Falling back to integer indices.'
            )
            logging.warning(f'\tFirst 3 cell_names: {list(cell_names[:3])}')
            index_to_name = {i: str(i) for i in range(num_cells_in_dataset)}
        else:
            logging.info(f'\tnetwork.cell_names OK: {len(cell_names)} barcodes match {num_cells_in_dataset} dataset columns')
            logging.info(f'\tFirst 3 barcodes: {[(i, str(cell_names[i]).strip()) for i in range(min(3, len(cell_names)))]}')
            index_to_name = {i: str(cell_names[i]).strip() for i in range(num_cells_in_dataset)}

        outfile_dir = f'{file_paths["trajectories"]}/{dataset_name}_{network_name}'
        png_dir = f'{outfile_dir}/png_files'
        text_dir = f'{outfile_dir}/text_files/'
        txt_traj_dir = f'{text_dir}/cell_trajectories/'
        png_traj_dir = f'{png_dir}/cell_trajectories/'

        os.makedirs(txt_traj_dir, exist_ok=True)
        os.makedirs(png_traj_dir, exist_ok=True)

        logging.info(f'\n\tSIMULATING CELL TRAJECTORIES')
        num_existing_files: int = len([file for file in os.listdir(txt_traj_dir) if file.endswith('_trajectory.csv')])
        logging.info(f'\t\tFound {num_existing_files} trajectory files')

        num_simulations: int = min((num_cells_to_analyze - num_existing_files), num_cells_in_dataset - num_existing_files)
        if num_simulations < 0:
            num_simulations = 0

        cells_to_simulate = simulate_cells(index_to_name, dense_dataset, cells_to_simulate, num_simulations, network, dataset_name, network_name)

        num_files: int = len([file for file in os.listdir(txt_traj_dir) if file.endswith('_trajectory.csv')])
        logging.info(f'\t\t{num_files} trajectory files after simulation')

        num_files_to_process = min(num_cells_to_analyze, num_files)

        logging.info(f'\n\tCLUSTERING CELLS')
        logging.info(f'\t\tUsing {mp.cpu_count()} CPUs')
        logging.info(f'\t\tCalculating cell trajectory clusters and averaging the cluster trajectories')
        cluster_cells(num_files_to_process, text_dir, num_cells_per_chunk)

        for cell in cell_cluster_dict:
            if cell not in cell_clusters:
                cell_clusters[cell] = {}
            if network.name not in cell_clusters[cell]:
                cell_clusters[cell][network.name] = 0
            cell_clusters[cell][network.name] = cell_cluster_dict[cell]

    create_cluster_combinations(cell_clusters, cell_group_dict)