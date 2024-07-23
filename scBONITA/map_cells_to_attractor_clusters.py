import pickle
import logging
import argparse
import glob
import numpy as np
from scipy.stats import chi2_contingency

from file_paths import file_paths
from setup.user_input_prompts import cell_attractor_mapping_arguments

def load_all_network_pickles(dataset_name: str):
    """
    Loads in all network pickle files for a given dataset_name
    """
    # Load in the network pickle files
    all_networks = []

    pickle_file_path = f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/network_pickle_files/'
    for pickle_file in glob.glob(pickle_file_path + str(dataset_name) + "_" + "*" + ".network.pickle"):
        if pickle_file:
            logging.info(f'\tLoading data file: {pickle_file}')
            network: object = pickle.load(open(pickle_file, "rb"))
            all_networks.append(network)
        else:
            assert FileNotFoundError("Network pickle file not found")

    return all_networks

def create_attractor_map(cells):
    attractor_combos = {}
    attractor_barcode = []

    for cell_num, cell in enumerate(cells):
        cell.groups = cell_group_dict[cell.index]
        
        # Print out the cell information if you need to debug the code
        if cell_num < 5:
            logging.debug(f'\nCell Name {cell.name}, index {cell.index}')
            
            logging.debug(f'\tGroup: {cell.groups}')

            logging.debug(f'\tAttractor dict:')
            for attractor_num, (network_name, attractor) in enumerate(cell.attractor_dict.items()):
                if attractor_num < 5:
                    logging.info(f'\t\tNetwork: {network_name} Attractor: {attractor}')

            logging.debug(f'\tExpression:')
            for gene_num, (gene_name, expression) in enumerate(cell.expression.items()):
                if gene_num < 5:
                    logging.debug(f'\t\t{gene_name}: {expression}')
        
        # Create the cell barcode with the attractor states
        attractor_barcode.append([attractor for attractor in cell.attractor_dict.values()])
        cell.attractor_barcode = attractor_barcode[cell_num]
        barcode = ', '.join([str(i) for i in cell.attractor_barcode])

        # Set up the dictionary values to count the number of cells in each group with the attractor combo
        if not barcode in attractor_combos.keys():
            attractor_combos[barcode] = {}
        
        if not cell.groups in attractor_combos[barcode]:
            attractor_combos[barcode][cell.groups] = 0

        attractor_combos[barcode][cell.groups] += 1
    
    return attractor_combos

if __name__ == '__main__':
    # Set the logging level for output
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Allow the user to either add in the dataset name and network name from the command line or as a prompt
    parser = argparse.ArgumentParser()

    dataset_name = cell_attractor_mapping_arguments(parser)

    # Load in the network pickle files for this dataset
    logging.info(f'\nMapping cell states...')
    all_networks = load_all_network_pickles(dataset_name)

    # Find the number of cells in the dataset
    cell_group_dict = {}

    group_cell_counts = {}

    groups = []

    # Create a dictionary of cell indices to group
    for group, cell_indices in all_networks[0].group_cell_indices.items():
        logging.info(f'\tGroup: {group}, {len(cell_indices)} cells')
        groups.append(group)

        for cell_index in cell_indices:
            cell_group_dict[cell_index] = group[0]
            group_cell_counts[group] = len(cell_indices)

    cell_info = {}
    network_info = {}

    # Load the cell objects for the dataset
    cell_population = pickle.load(open(f'{file_paths["pickle_files"]}/{dataset_name}_pickle_files/cells_pickle_file/{dataset_name}.cells.pickle', "rb"))
    cells = cell_population.cells

    num_cells = len(cells)
    logging.info(f'\nTotal number of cells: {num_cells}')

    # Use the cell object info to get a count of how many cells from each group are in each attractor combo
    attractor_combos = create_attractor_map(cells)
 
    chi_square_table = []

    network_names = "\t".join([i.name for i in all_networks])
    group_names = "\t".join([i for i in groups])

    # Write the combination cell counts and groups to an output file
    logging.info(f'\nWriting out the attractor mapping results to {dataset_name}_cell_attractor_states.csv')
    path = f'{file_paths["attractor_analysis_output"]}/{dataset_name}_attractors/{dataset_name}_cell_attractor_states.csv'
    with open(path, 'w') as combination_file:
        combination_file.write(f'{network_names}\t{group_names}\ttotal\n')

        # Extract and write the information for each attractor combination to the combination file
        for attractor_combo, group_dict in attractor_combos.items():
            combination_counts = []
            total_cells = 0
            for group in groups:
                if group not in group_dict.keys():
                    group_dict[group] = 0
            for group, cell_count in group_dict.items():
                total_cells += cell_count
                combination_counts.append(cell_count + 5) # Add 5 to account for columns with 0 cells, so it works with chi-square test
            chi_square_table.append(combination_counts)
            attractors = "\t".join([i for i in attractor_combo.split(', ')])
            group_counts = "\t".join([str(value) for value in group_dict.values()])
            combination_file.write(f'{attractors}\t{group_counts}\t{total_cells}\n')

        # Performing the Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(np.array(chi_square_table))

        chi2, p_value

        combination_file.write(f'Chi Square Value: {chi2}, p_value = {p_value}')
        logging.info(f'\tChi Square Value: {chi2}, p_value = {p_value}')

        contributions = (chi_square_table - expected) ** 2 / expected
        logging.info("Contributions of each cell to the chi-square statistic:")
        logging.info(f'\t{contributions}')
