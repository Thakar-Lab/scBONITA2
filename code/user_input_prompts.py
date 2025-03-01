import os
import logging
from argparse import ArgumentParser
from kegg_parser import *
import sys
import glob


def rule_inference_arguments(parser):
    kegg = KEGG()
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=False,
        default="",
        help='Name of the dataset for loading the correct ruleset pickle files'
    )

    parser.add_argument(
        "--data_file",
        type=str,
        help="Specify the name of the file containing processed scRNA-seq data",
        default=""
    )

    parser.add_argument(
        "--network_files",
        nargs="+",
        type=str,
        default="",
        help="File name of the processed network for which rules are to be inferred",
        required=False
    )

    parser.add_argument(
        "--max_nodes",
        type=int,
        help="Number of genes in the dataset", 
        default=20000
    )

    parser.add_argument(
        "--datafile_sep",
        type=str,
        help='Delimiting character in the data file. Must be one of , (comma), \\s (space) or \\t (tab)',
        default=",",
        choices=[",", r"\s", r"\t"]
    )

    parser.add_argument(
        "--get_kegg_pathways",
        type=str,
        help="Should scBonita automatically identify and download KEGG pathways with genes that are in your dataset? You can specify which pathways using the list_of_kegg_pathways option, or leave it blank to download all matching KEGG pathways",
        default="True",
        required=False
    )

    parser.add_argument(
        "--list_of_kegg_pathways",
        nargs="+",
        type=str,
        help="Which KEGG pathways should scBonita download? Specify the five letter pathway IDs.",
        required=False
    )

    parser.add_argument(
        "--pathway_list",
        nargs="+",
        type=str,
        help="Paths to GRAPHML files that should be used for scBONITA analysis. Usually networks from non-KEGG sources, saved in GRAPHML format",
        required=False
    )

    parser.add_argument(
        "--organism",
        type=str,
        help="Three-letter organism code. Which organism is the dataset derived from?",
        default="hsa",
        choices=kegg.organismIds,
        required=False,
        metavar='organism code'
    )

    parser.add_argument(
        "--cv_threshold",
        type=float,
        help="Minimum coefficient of variation to retain genes for scBONITA analysis",
        default=None,
        required=False
    )

    parser.add_argument(
        "--binarize_threshold",
        type=float,
        help="Threshold for binarization of the training dataset. Values above this are classed as 1 (or 'on') and values below this are classed as 0 (or 'off').",
        default=0.001,
        required=False
    )

    parser.add_argument(
        "--minimum_overlap",
        type=int,
        help="The minimum number of genes shared between the dataset and the patwhay. Default is 25.",
        default=25,
        required=False
    )

    parser.add_argument(
        "--sample_cells",
        type=str,
        help="If True, scBonita will use a representative set of samples to infer rules. This is automatically done if the number of cells in the training dataset exceeds 15000, in order to reduce memory usage.",
        default="False",
        required=False,
    )

    parser.add_argument(
        "--parallel_search",
        type=str,
        help="If True, scBonita will use a parallelized local search to speed up the rule inference for large networks (with ~ 100 nodes or more). We suggest setting this to False for smaller networks or for KEGG networks, to reduce memory overheads with minimal loss in speed.",
        default="False",
        required=False,
    )

    parser.add_argument(
        "--display_title",
        type=str,
        help="Display scBONITA title? [True/False]",
        default="True",
        required=False
    )
    
    # Store the results from the arguments
    try:
        results = parser.parse_args()
    except:
        print("Error: Unrecognized arguments or missing required arguments")
        sys.exit(1)

    return results

def add_dataset_name_arg(parser):
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=False,
        default="",
        help='Name of the dataset for loading the correct ruleset pickle files'
    )

def add_list_of_kegg_pathways(parser):
        parser.add_argument(
        "--list_of_kegg_pathways",
        nargs="+",
        type=str,
        help="Which KEGG pathways should scBonita download? Specify the five letter pathway IDs.",
        required=False,
        default=""
    )

def add_organism_code(parser):
        kegg = KEGG()
        parser.add_argument(
        "--organism",
        type=str,
        help="Three-letter organism code. Which organism is the dataset derived from?",
        default="hsa",
        choices=kegg.organismIds,
        required=False,
        metavar='organism code'
    )

def add_metadata_file_arg(parser):
    parser.add_argument(
        '--metadata_file',
        type=str,
        required=False,
        default="",
        help='Pass in the path for the metadata file'
    )


def add_metadata_sep_arg(parser):
    parser.add_argument(
        '--metadata_sep',
        type=str,
        required=False, 
        default="", 
        help='What is the separator character used between items in the metadata file?'
    )

def add_dataset_file_arg(parser):
    parser.add_argument(
        '--dataset_file',
        type=str,
        required=False,
        default="",
        help='Pass in the path for the dataset file'
    )

def add_dataset_sep_arg(parser):
     parser.add_argument(
        '--dataset_sep',
        type=str,
        required=False, 
        default="", 
        help='What is the separator character used between items in the dataset file?'
    )

def add_control_group_arg(parser):
    parser.add_argument(
        '--control_group',
        type=str,
        required=False,
        default="",
        help='Enter the group that you want to use as a control (uses its node importance scores, must be the same format as the file names, e.g. condition1_condition2)'
    )

def add_experimental_group_arg(parser):
    parser.add_argument(
        '--experimental_group',
        type=str,
        required=False,
        default="",
        help='Enter the experimental group (must be the same format as the file names, e.g. condition1_condition2)'
    ) 


def add_show_figure_arg(parser):
    parser.add_argument(
        '--show_figure',
        type=str,
        required=False,
        default="",
        help='Shows the figure of the representative attractors over time'
    )

def add_network_name(parser):
    parser.add_argument(
        '--network_name',
        type=str,
        required=False, 
        default="", 
        help='Specify PROCESSED network to anlayze or type "all". Requires a network pickle file for this dataset'
    )

def load_all_networks(dataset_name):
    list_of_networks = []

    for network_path in glob.glob(f'pickle_files/{dataset_name}_pickle_files/ruleset_pickle_files/{dataset_name}_*.ruleset.pickle'):
        network_name = network_path.split('/')[-1].split(str(dataset_name) + '_')[1].split('.ruleset.pickle')[0]
        list_of_networks.append(network_name)

    return list_of_networks

def add_cell_name_index(parser):
    parser.add_argument(
        '--cell_name_index',
        type=int,
        required=False, 
        default="", 
        help='Specify the column containing the cell names (first column = 0) '
    )

def add_group_indices(parser):
    parser.add_argument(
        "--group_indices",
        nargs="+",
        type=int,
        help="Specify the columns containing the group indices (first column = 0)",
        required=False
    )

def add_header(parser):
    parser.add_argument(
        '--header',
        type=str,
        required=False, 
        default="", 
        help='Does the line contain a header? (y or n)'
    )

def add_overwrite(parser):
    parser.add_argument(
        '--overwrite',
        type=str,
        required=False, 
        default="", 
        help='If the split group data files exist, do you want to overwrite? (y or n)'
    )

# ----- FILE ARGUMENTS -----

def metadata_arguments(parser):
    """
    Required arguments for splitting the dataset into groups based on the metadata file
    """
    add_metadata_file_arg(parser)
    add_metadata_sep_arg(parser)
    add_dataset_file_arg(parser)
    add_dataset_sep_arg(parser)

    args = parser.parse_args()

    metadata_file = check_metadata_file(args.metadata_file)
    metadata_sep = check_separator(args.metadata_sep)

    dataset_file = check_dataset_file(args.dataset_file)
    dataset_sep = check_separator(args.dataset_sep)

    return metadata_file, metadata_sep, dataset_file, dataset_sep

def split_dataset_arguments(parser):
    """
    Required arguments for the relative abundance calcuations
    """
    add_dataset_name_arg(parser)
    add_metadata_file_arg(parser)
    add_metadata_sep_arg(parser)
    add_dataset_file_arg(parser)
    add_dataset_sep_arg(parser)
    add_control_group_arg(parser)
    add_experimental_group_arg(parser)
    add_cell_name_index(parser)
    add_group_indices(parser)
    add_header(parser)
    add_overwrite(parser)

    args = parser.parse_args()

    dataset_name = check_dataset_name(args.dataset_name)
    dataset_file = check_dataset_file(args.dataset_file)
    dataset_sep = check_separator(args.dataset_sep)

    metadata_file = check_metadata_file(args.metadata_file)
    metadata_sep = check_separator(args.metadata_sep)

    control_group = args.control_group
    experimental_group = args.experimental_group

    cell_name_index = args.cell_name_index
    group_indices = args.group_indices
    header = args.header
    overwrite = args.overwrite

    list_of_networks = load_all_networks(dataset_name)
    
    return dataset_name, list_of_networks, metadata_file, metadata_sep, dataset_file, dataset_sep, control_group, experimental_group, cell_name_index, group_indices, header, overwrite

def attractor_analysis_arguments(parser):
    """
    Required arguments for attractor analysis
    """
    add_dataset_name_arg(parser)
    add_network_name(parser)

    parser.add_argument(
        '--num_cells_per_chunk',
        type=int,
        required=False,
        default=1,
        help='Number of cells per chunk used for attractor analysis'
    )

    parser.add_argument(
        '--num_cells_to_analyze',
        type=int,
        required=False,
        default=1,
        help='Number of cells to analyze for attractor analysis'
    )

    args = parser.parse_args()

    dataset_name = check_dataset_name(args.dataset_name)
    num_cells_per_chunk = args.num_cells_per_chunk
    num_cells_to_analyze = args.num_cells_to_analyze
    
    return dataset_name, num_cells_per_chunk, num_cells_to_analyze

def cell_attractor_mapping_arguments(parser):
    add_dataset_name_arg(parser)

    args = parser.parse_args()

    dataset_name = check_dataset_name(args.dataset_name)
    logging.info(dataset_name)
    
    return dataset_name

# ----- ARGUMENTS CHECKS -----
def check_dataset_name(dataset_name):
    if dataset_name == "":
        while True:
            dataset_name = input('What is the dataset name?: ')
            path = f'pickle_files/{dataset_name}_pickle_files/ruleset_pickle_files'

            if os.path.exists(path):
                logging.info(f'\tDataset pickle files found!')
                dataset_name = dataset_name
                return dataset_name
            
            else:
                logging.info(f'Error: Path does not exist, check spelling or make sure that rule inference has been run for this dataset')

    else:
        return dataset_name

def check_dataset_file(dataset_file):
    if dataset_file == "":
        while True:
            data_file = input("Enter the dataset file name: ")

            if os.path.exists(data_file):
                return dataset_file
            
            else:
                logging.info('Error: Path specifed does not exist, try again')
    else:
        return dataset_file

def check_metadata_file(metadata_file):
    if metadata_file == "":
        while True:
            metadata_file = input("Enter the metadata file name: ")

            if os.path.exists(metadata_file):
                metadata_file = metadata_file

                return metadata_file
                
            else:
                logging.info('Error: Path specifed does not exist, try again')
    else:
        return metadata_file

def check_separator(separator):
    if separator == "":
        choices=[",", " ", "    "]
        while True:
            separator = input("Enter the dataset file separactor character: ")

            if separator in choices:
                return separator
            
            else:
                logging.info(f'Separator must be a space, tab, or comma')
    else:
        return separator
    
def check_control_group(control_group):
    if control_group == "":
        control_group = input("Enter the name of the control group (same format as in file names): ")
    
    return control_group

def check_experimental_group(experimental_group):
    if experimental_group == "":
        experimental_group = input("Enter the name of the experimental group (same format as in file names): ")

    return experimental_group

def check_show_figure(show_figure):
    if show_figure == "":
        while True:
            show_figure = input("\nShow output graph? [y/n]: ")
            if show_figure.lower() == 'y':
                show_figure = True
                break
            elif show_figure.lower() == 'n':
                show_figure = False
                break            
            else:
                show_figure = False
                break
    
    elif show_figure == "True":
        show_figure = True
    
    elif show_figure == "False":
        show_figure = False
    
    return show_figure


# Metadata parser checks
def check_cell_name_index(cell_name_index):
    if cell_name_index == None:
        while True:
            cell_name_index = input(f'\tRow number containing the cell names (first row = 0): ')
            try:
                cell_name_index = int(cell_name_index)
                return cell_name_index
            except:
                logging.info(f'\t\tError: please type an integer')
    else:
        return cell_name_index

def check_group_indices(group_indices):
    if group_indices == None:
        group_indices = []
        while True:
            num_groups = input(f'\tHow many groups are in the dataset?: ')
            try:
                
                num_groups = int(num_groups)
                break
            except:
                logging.info(f'\t\tError: please type an integer')

        for group_num in range(num_groups):
            def number_to_ordinal(n):
                # Define the special cases for the endings of ordinal numbers
                if 10 <= n % 100 <= 20:
                    suffix = 'th'
                else:
                    # Define the suffixes for the last digit of n, default to 'th'
                    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
                return str(n) + suffix

            while True:
                group_index = input(f'\tRow number containing the {number_to_ordinal(group_num+1)} group: ')
                try:
                    logging.info(type(num_groups))
                    group_index = int(group_index)
                    group_indices.append(group_index)
                    break
                except:
                    logging.info(f'\t\tError: please type an integer')
        return group_indices
    else:
        return group_indices

def check_header(header):
    if header == "":
        while True:
            header = input(f'\tDoes the line contain a header? (y/n): ')
            if header.lower() == 'y':
                header = True
                return header
            elif header.lower() == 'n':
                header = False
                return header
            else:
                logging.info(f'\t\tPlease type y or n')
    else:
        return header

def overwrite_check(overwrite, output_path):
    if overwrite == 'y' or overwrite == True:
        overwrite = True
        return overwrite
    
    elif overwrite == 'n' or overwrite == False:
        overwrite = False
        return overwrite
    
    else:
        while True:
            overwrite_output_path = input(f'\n\t{output_path.split("/")[-1]} exists. Do you want to overwrite? (y/n): ')
            if overwrite_output_path.lower() == 'y':
                overwrite = True
                return overwrite

            elif overwrite_output_path.lower() == 'n':
                overwrite = False
                return overwrite

            else:
                logging.info(f'Error: please enter y or n')