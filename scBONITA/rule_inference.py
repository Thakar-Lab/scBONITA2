import scipy.sparse as sparse
import numpy as np
from os import path
import matplotlib.pyplot as plt
import time
import csv
import networkx as nx
import copy

from cell_class import Cell
from sklearn import preprocessing

from network_setup import *
from kegg_parser import *
from deap_class import CustomDeap


class RuleInference(NetworkSetup):

    """Class for single-cell experiments"""

    def __init__(
        self,
        data_file,
        dataset_name,
        network_name,
        sep,
        node_indices,
        gene_list,
        max_nodes=15000,
        binarize_threshold=0.001,
        sample_cells=True,
    ):
        # Check is the data file exists
        if path.isfile(data_file):
            pass
        else:
            raise FileNotFoundError(f'File not found: {data_file}')
        
        self.data_file = data_file
        self.sample_cells = sample_cells
        self.node_indices = node_indices
        self.dataset_name = dataset_name
        self.network_name = network_name
        self.binarize_threshold = binarize_threshold
        self.max_samples = 15000
        self.cells = []

        logging.info(f'\n-----EXTRACTING AND FORMATTING DATA-----')
        # Extract the data from the data file based on the separator, sample the cells if over 15,000 cells
        logging.info(f'Extracting cell expression data from "{data_file}"')
        self.cell_names, self.gene_names, self.data = self._extract_data(data_file, sep, sample_cells, node_indices)


        logging.info(f'\tFirst 2 genes: {self.gene_names[:2]}')
        logging.info(f'\tFirst 2 cells: {self.cell_names[:2]}')
        
        logging.info(f'\tNumber of genes: {len(self.gene_names)}')
        logging.info(f'\tNumber of cells: {len(self.cell_names)}')

        self.sparse_matrix = sparse.csr_matrix(self.data)
        logging.info(f'\tCreated sparse matrix')
        logging.debug(f'\tShape: {self.sparse_matrix.shape}')
        
        self.gene_names = list(self.gene_names)
        self.cell_names = list(self.cell_names)
        self.sparse_matrix.eliminate_zeros()

        # Check if there are at least 1 sample selected
        if self.data.shape[0] > 0:
            # Binarize the values in the sparse matrix
            logging.info(f'\tBinarized sparse matrix')
            
            self.binarized_matrix = preprocessing.binarize(self.sparse_matrix, threshold=binarize_threshold, copy=True)
            logging.debug(f'{self.binarized_matrix[:5,:5]}')
            
        else:
            # Handle the case where no samples were selected
            raise ValueError("No samples selected for binarization")
            
        
        full_matrix = self.binarized_matrix.todense()

        # Create cell objects
        for cell_index, cell_name in enumerate(self.cell_names):
            cell = Cell(cell_index)
            cell.name = cell_name
            for row_num, row in enumerate(full_matrix):
                row_array = np.array(row).flatten()
                # print(f'Cell {cell_index}, Row {row_num}')
                try:
                    cell.expression[self.gene_names[row_num]] = row_array[cell_index]
                except IndexError as e:
                    logging.debug(f'Encountered error {e} at row {row_num}, col {cell_index}. If at the last gene position, ignore')

            self.cells.append(cell)


        self.max_nodes = max_nodes
        self.pathway_graphs = {}
        self.node_list = []
        self.node_positions = []
        self.gene_list = gene_list

    def _extract_data(self, data_file, sep, sample_cells, node_indices):
        """
        Extract the data from the data file
        Parameters
        ----------
        data_file
        sep
        sample_cells

        Returns
        -------
        cell_names, data
        """
        with open(data_file, 'r') as file:
            reader = csv.reader(file, delimiter=sep)

            # Extract the header (cell_names)
            cell_names = next(reader)[1:]

            cell_count = len(cell_names)

            # Randomly sample the cells in the dataset
            if cell_count >= self.max_samples or sample_cells:
                logging.info(f'\tRandomly sampling {self.max_samples} cells...')
                sampled_cell_indices = np.random.choice(
                    range(cell_count),
                    replace=False,
                    size=min(self.max_samples, cell_count),
                )
                logging.info(f'\t\tNumber of cells: {len(sampled_cell_indices)}')

            else:
                sampled_cell_indices = range(cell_count)
                logging.info(f'\tLoading all {len(sampled_cell_indices)} cells...')

            # Data extraction
            data_shape = (len(node_indices), len(sampled_cell_indices))
            data = np.empty(data_shape, dtype="float")
            gene_names = []
            data_row_index = 0  # Separate index for data array
            for i, row in enumerate(reader):
                if i in node_indices:  # Only keeps the nodes involved, skips the cell name row
                    gene_names.append(row[0])
                    
                    # Offset cell indices by 1 to skip the gene name column
                    selected_data = [float(row[cell_index+1]) for cell_index in sampled_cell_indices]
                    data[data_row_index, :] = selected_data
                    data_row_index += 1

            # Convert the filtered data to a NumPy array
            logging.info("\tConverting filtered data to numpy array...")
            # print(f'Raw data:\n\t{data}')

            return cell_names, gene_names, data

    def filterData(self, threshold):
        """Filters the data to include genes with high variability (genes with a std dev / mean ratio above the cv_cutoff threshold)"""
        self.cv_genes = []
        if threshold is not None:
            for i in range(0, self.sparse_matrix.get_shape()[0]):
                rowData = list(self.sparse_matrix.getrow(i).todense())
                if np.std(rowData) / np.mean(rowData) >= threshold:
                    self.cv_genes.append(self.gene_names[i])
        else:
            self.cv_genes = copy.deepcopy(self.gene_names)
    
    def genetic_algorithm(self, net):
        # Genetic algorithm
        custom_deap = CustomDeap(
            net,
            self.network_name,
            self.dataset_name,
            self.binarized_matrix,
            self.nodeList,
            self.nodes,
            self.deap_individual_length,
            self.nodeDict,
            self.successorNums
            )

        raw_fitnesses, population, logbook = custom_deap.genetic_algorithm()
        best_ruleset = custom_deap.find_best_individual(population, raw_fitnesses)

        return best_ruleset

    def rule_determination(self, graph):
        """Main function that performs rule determination and node scoring in preparation for pathway analysis"""

        # Load the processed graphml file as a NetworkX object
        start_time = time.time()
        if path.exists(graph):
            logging.info(f'\t\tLoading: {graph.split("/")[-1]}')
            self.network = nx.read_graphml(graph)
        else:
            msg = f'File "{graph} not found"'
            raise FileNotFoundError(msg)

        # Find the network gene names
        netGenes = [self.gene_names.index(gene) for gene in list(self.network) if gene in self.gene_names]

        logging.debug(f'Network Genes: {netGenes}')

        # Sets up the nodes and classes
        self.__inherit(self.network)
        
        # Runs the genetic algorithm and rule refinement
        self.best_ruleset = self.genetic_algorithm(self.network)

    def plot_graph_from_graphml(self, network):
        G = network

        # Extract values and ensure they are within [0, 1]
        values = [node.importance_score for node in self.nodes]
        logging.debug(f'\nNormalized Values: {values}')

        # Choose a colormap
        cmap = plt.cm.Greys

        def scale_numbers(numbers, new_min, new_max):
            # Calculate the min and max of the input numbers
            old_min = min(numbers)
            old_max = max(numbers)
            
            # Scale the numbers to the new range
            scaled_numbers = []
            for num in numbers:
                scaled_num = ((num - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
                scaled_numbers.append(scaled_num)
            
            return scaled_numbers

        new_min = 0.1
        new_max = 0.7
        scaled_numbers = scale_numbers(values, new_min, new_max)

        node_colors = [cmap(value) for value in scaled_numbers]

        # Map 'values' to colors
        logging.debug(f'\nNode Colors: {node_colors}')

        pos = nx.spring_layout(G, k=1)  # Layout for visualizing the graph

        # Draw the graph
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, font_color="black", font_size=10, ax=ax)

        ax.set_title("Importance Score for Each Node in the Network")
        ax.set_axis_off()  # Hide the axes
        # plt.show()

        return fig
    
    def __inherit(self, graph):
        super().__init__(graph)

