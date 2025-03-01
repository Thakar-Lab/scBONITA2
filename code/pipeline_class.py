from rule_inference import RuleInference
from kegg_parser import Pathways
from cell_class import CellPopulation
from file_paths import file_paths

# Setup
from scbonita_banner import make_banner
from user_input_prompts import rule_inference_arguments

# General packages
import os
import pickle
import logging

from argparse import ArgumentParser


class Pipeline():
    def __init__(self):
        # Set the logging level for output
        logging.basicConfig(format='%(message)s', level=logging.INFO)

        # Read in arguments from shell script using argparse
        parser = ArgumentParser()
        parser_argument = rule_inference_arguments(parser)
        
        # Store the results from argparse
        self.dataset_name = getattr(parser_argument, 'dataset_name', "")
        self.network_files = getattr(parser_argument, 'network_files', [])
        self.max_nodes = getattr(parser_argument, 'max_nodes', 20000)
        self.data_file = parser_argument.data_file
        self.datafile_sep = getattr(parser_argument, 'datafile_sep', ',')
        self.get_kegg_pathways = self._convert_string_to_boolean(getattr(parser_argument, 'get_kegg_pathways', True))
        self.list_of_kegg_pathways = getattr(parser_argument, 'list_of_kegg_pathways', [])
        self.organism = getattr(parser_argument, 'organism', 'hsa')
        self.cv_threshold = getattr(parser_argument, 'cv_threshold', 0.001)
        self.sample_cells = self._convert_string_to_boolean(getattr(parser_argument, 'sample_cells', True))
        self.binarize_threshold = float(getattr(parser_argument, 'binarize_threshold', 0.001))
        self.display_title = getattr(parser_argument, 'display_title', True)
        self.minOverlap = getattr(parser_argument, 'minimum_overlap', 25)
        
        # Other variables
        self.write_graphml = True
        self.node_indices = []

        # Make the paths for the output files
        for path in file_paths.values():
            os.makedirs(path, exist_ok=True)

        # Chooses whether to display the scBONITA banner in the terminal
        if self.display_title:
            logging.info(make_banner())
        
        # Runs the rule inference
        self.rule_inference()
    
    def rule_inference(self):
        """
        Wrapper script for running scBONITA rule inference
        """

        # Create a Pathways object to store the network information
        pathways = Pathways(
            self.dataset_name,
            self.cv_threshold,
            self.data_file,
            self.datafile_sep,
            self.write_graphml,
            self.organism)

        # Use a list of KEGG pathways specified by the user
        logging.info(f'-----PARSING NETWORKS-----')
        if self.list_of_kegg_pathways is None:
            logging.info(f'\tNo KEGG Pathways listed...')
            self.list_of_kegg_pathways = []
        else:
            logging.info(f'\tKEGG pathways = {self.list_of_kegg_pathways}')

            logging.info(f'\tFinding and formatting KEGG Pathways...')

            # Find and load the KEGG pathways
            pathways.pathway_graphs = pathways.find_kegg_pathways(
                kegg_pathway_list=self.list_of_kegg_pathways,
                write_graphml=self.write_graphml,
                organism=self.organism,
                minimumOverlap=self.minOverlap
            )

            # Add the pathways to the pathways object
            pathways.add_pathways(
                pathways.pathway_graphs,
                minOverlap=self.minOverlap,
                organism=self.organism
            )

        # Adds any custom graphml files
        if len(self.network_files) > 0:
            for i, file in enumerate(self.network_files):
                if '/' in file:
                    self.network_files.pop(i)
                    self.network_files.append(file.split('/')[-1])

            logging.info(f'\tCustom pathway: {self.network_files}')

            pathways.add_pathways(
                pathway_list=self.network_files,
                minOverlap=self.minOverlap,
                organism=self.organism
            )

        # Get the information from each pathway and pass the network information into a ruleset object
        for pathway_num in pathways.pathway_graphs:

            # Pull out the graph object for the current pathway
            graph = pathways.pathway_graphs[pathway_num]

            # Catches if the graph does not have enough overlapping nodes after processing
            if len(graph.nodes()) >= self.minOverlap:
                
                # Make a list of the position of each gene
                node_indices = []
                for node in graph.nodes():
                    node_indices.append(pathways.gene_list.index(node))

                # node_indices = set(node_indices)  # Only keep unique values
                self.node_indices = list(node_indices)  # Convert back to a list

                # Runs rule determination
                self.infer_rules(pathway_num, graph, node_indices)
            
            else:
                logging.info(f'\t\t\tNot enough overlapping nodes for {pathway_num} (min {self.minOverlap}, overlap {len(graph.nodes())})')

    def infer_rules(self, pathway_num, graph, node_indices):
        """
        Runs scBONITA rule determination via the rule_inference.py script.

        Saves output pickle files
        """

        logging.info(f'\n-----RULE INFERENCE-----')
        logging.info(f'Pathway: {pathway_num}')
        logging.info(f'Num nodes: {len(node_indices)}')

        # Create RuleInference object
        ruleset = RuleInference(
            self.data_file,
            graph,
            self.dataset_name,
            pathway_num,
            self.datafile_sep,
            node_indices,
            self.binarize_threshold,
            self.sample_cells)

        # Create the ruleset pickle files
        logging.info(f'\nRule inference complete, saving ruleset pickle file')

        # Specify the path to the ruleset pickle directory, ensures the directory exists
        data_pickle_folder = f'{file_paths["pickle_files"]}/{self.dataset_name}_pickle_files/ruleset_pickle_files'
        os.makedirs(data_pickle_folder, exist_ok=True)
        if self.organism not in pathway_num:
            data_pickle_file_path = f'{data_pickle_folder}/{self.dataset_name}_{self.organism}{pathway_num}.ruleset.pickle'
        else:
            data_pickle_file_path = f'{data_pickle_folder}/{self.dataset_name}_{pathway_num}.ruleset.pickle'
        logging.info(f'\tSaving to {data_pickle_file_path.split("/")[-1]}')

        
        # Save the ruleset object as a binary pickle file
        pickle.dump(ruleset, open(data_pickle_file_path, "wb"))

        # Write out the cells objects to a pickle file
        logging.info(f'Saving cell population pickle file')
        cell_population = CellPopulation(ruleset.cells)

        # Specify the path to the cell pickle directory, ensures sure the directory exists
        cell_pickle_dir = f'{file_paths["pickle_files"]}/{self.dataset_name}_pickle_files/cells_pickle_file'
        os.makedirs(cell_pickle_dir, exist_ok=True)
        cells_pickle_file = f'{self.dataset_name}.cells.pickle'

        # Save the cell population object as a pickle file (used to simulate individual cells)
        pickle.dump(cell_population, open(f'{cell_pickle_dir}/{cells_pickle_file}', "wb"))

    def _convert_string_to_boolean(self, variable):
        if variable == "True" or variable is True:
            return True
        else:
            return False

if __name__ == "__main__":
    Pipeline()

