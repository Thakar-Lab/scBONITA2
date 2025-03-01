from bioservices.kegg import KEGG
import networkx as nx
import requests
import re
from bs4 import BeautifulSoup
import itertools
from rule_inference import *
import logging
import os
from alive_progress import alive_bar
import sys

from file_paths import file_paths

class Pathways:
    """
    Reads in and processes the KEGG pathways as networkx graphs
    """
    def __init__(self, dataset_name, cv_threshold, data_file, sep, write_graphml, organism):
        self.cv_threshold = cv_threshold # The cutoff value threshold for binarizing 
        self.data_file = data_file
        self.gene_list = self._find_genes(sep)
        self.pathway_graphs = {}
        self.dataset_name = dataset_name
        self.output_path = f'{file_paths["graphml_files"]}/{dataset_name}/'
        self.gene_indices = []
        self.pathway_dict = {}
        self.organism = organism
        
        if self.cv_threshold:
            self.filter_data()

        os.makedirs(self.output_path, exist_ok=True)

    def _count_generator(self, reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)
    
    def _find_genes(self, separator):
        """
        Finds the names of the genes in the datafile
        """
        gene_list = []  # Initialize a list to store the first column data
        
        with open(self.data_file, "rb") as file:
            c_generator = self._count_generator(file.read)
            gene_count = sum(buffer.count(b'\n') for buffer in c_generator)

        with open(self.data_file, "r") as file:
            next(file)
            for line in file:
                # Split the line into columns (assuming columns are separated by spaces or tabs)
                line = line.replace('"', '')
                columns = line.strip().split(separator)
                
                if columns:
                    gene_list.append(columns[0])  # Append the first column to the list

        logging.info(f'Number of genes in the data file: {len(gene_list)}')
        logging.info(f'First 5 genes: {gene_list[0:5]}\n')

        return gene_list
    
    def filter_data(self):
        """
        Filters out genes with low variability. The threshold is low by default (0.001) to allow for most genes
        to be kept in. Increasing this value will lead to only including highly variable genes
        """
        logging.info(f'\tFiltering data based on cv threshold of {self.cv_threshold}')
        self.cv_genes = []
        with open(self.data_file, "r") as file:
            next(file)
            for line in file:
                column = line.split(',')
                gene_name = column[0]
                row_data = [float(cell_value) for cell_value in column[1:]]
                
                # Calculate the cutoff value 
                if np.std(row_data) / np.mean(row_data) >= self.cv_threshold:
                    if gene_name in self.gene_list:
                        self.cv_genes.append(gene_name)

    def parse_kegg_dict(self):
        """
        Makes a dictionary to convert ko numbers from KEGG into real gene names
        """
        logging.info(f'\t\tParsing KEGG dict...')
        gene_dict = {}

        # If the dictionary file exists, use that (much faster than streaming)
        if 'kegg_dict.csv' in os.listdir(f'{file_paths["pathway_xml_files"]}'):
            logging.info(f'\t\t\tReading in KEGG dictionary file...')
            with open(f'{file_paths["pathway_xml_files"]}/kegg_dict.csv', 'r') as kegg_dict_file:
                for line in kegg_dict_file:
                    line = line.strip().split('\t')
                    kegg_code = line[0]
                    gene_number = line[1]
                    gene_dict[kegg_code] = gene_number

        # If the dictionary file does not exist, write it and stream in the data for the dictionary
        else:
            logging.info(f'\t\t\tKEGG dictionary not found, downloading...')

            pathway_file = requests.get("http://rest.kegg.jp/get/br:ko00001", stream=True)
            with open(f'{file_paths["pathway_xml_files"]}/kegg_dict.csv', 'w') as kegg_dict_file:
                for line in pathway_file.iter_lines():
                    line = line.decode("utf-8")
                    if len(line) > 1 and line[0] == "D":  # lines which begin with D translate kegg codes to gene names
                        
                        # to split into kegg code, gene names
                        converter = re.split(r"\s+", re.split(r";", line)[0])
                        kegg_code = converter[1].upper()
                        gene_number = converter[2].upper().replace(',', '')
                        gene_dict[kegg_code] = gene_number
                        kegg_dict_file.write(f'{kegg_code}\t{gene_number}\n')
            pathway_file.close()
                
        return gene_dict

    def expand_groups(self, node_id, groups):
        """
        node_id: a node ID that may be a group
        groups: store group IDs and list of sub-ids
        return value: a list that contains all group IDs deconvoluted
        """
        node_list = []
        if node_id in groups.keys():
            for component_id in groups[node_id]:
                node_list.extend(self.expand_groups(component_id, groups))
        else:
            node_list.extend([node_id])
        return node_list
    
    def read_kegg(self, lines, graph, KEGGdict, hsaDict):
        # read all lines into a bs4 object using libXML parser
        logging.info('\t\tReading KEGG xml file')
        soup = BeautifulSoup("".join(lines), "xml")
        groups = {}  # store group IDs and list of sub-ids
        id_to_name = {}  # map id numbers to names
        subpaths = []
        
        # Look at each entry in the kgml file. Info: (https://www.kegg.jp/kegg/xml/)
        for entry in soup.find_all("entry"):

            # Name of each gene in the entry
            # If there are multiple genes in the entry, store them all with the same id
            entry_split = entry["name"].split(":")

            # If the entry is part of a group (in the network coded by a group containing lots of related genes)
            if len(entry_split) > 2:

                # Choose which dictionary to use based on whether the entries are hsa or kegg elements
                # Entries with hsa correspond to genes, entries with ko correspond to orthologs
                if entry_split[0] == "hsa" or entry_split[0] == "ko":
                    if entry_split[0] == "hsa":
                        useDict = hsaDict
                    elif entry_split[0] == "ko":
                        useDict = KEGGdict
                    nameList = []
                    
                    # Split off the first name
                    entry_name = ""
                    namer = entry_split.pop(0)
                    namer = entry_split.pop(0)
                    namer = namer.split()[0]

                    # Either use the dictionary name for the key or use the name directly if its not in the dictionary
                    entry_name = (
                        entry_name + useDict[namer]
                        if namer in useDict.keys()
                        else entry_name + namer
                    )

                    # Append each gene name to the list ([gene1, gene2])
                    for i in range(len(entry_split)):
                        nameList.append(entry_split[i].split()[0])

                    # For each of the gene names, separate them with a "-" (gene1-gene2)
                    for namer in nameList:
                        entry_name = (
                            entry_name + "-" + useDict[namer]
                            if namer in useDict.keys()
                            else entry_name + "-" + namer
                        )
                    entry_type = entry["type"]
                else:
                    entry_name = entry["name"]
                    entry_type = entry["type"]
            
            # If there is only one name
            else:
                # If the name is hsa
                if entry_split[0] == "hsa":
                    entry_name = entry_split[1] # Get the entry number
                    entry_type = entry["type"] # Get the entry type
                    entry_name = ( # Get the gene name from the entry number if its in the hsa gene name dict
                        hsaDict[entry_name] if entry_name in hsaDict.keys() else entry_name
                    )
                # If the name is ko, do the same as above but use the KEGGdict instead of the hsa gene name dict
                elif entry_split[0] == "ko":
                    entry_name = entry_split[1]
                    entry_type = entry["type"]
                    entry_name = (
                        KEGGdict[entry_name]
                        if entry_name in KEGGdict.keys()
                        else entry_name
                    )
                # If the entry is another KEGG pathway number, store the name of the signaling pathway
                elif entry_split[0] == "path":
                    entry_name = entry_split[1]
                    entry_type = "path"
                # If none of the above, just store the name and type
                else:
                    entry_name = entry["name"]
                    entry_type = entry["type"]
            
            # Get the unique entry ID for this pathway
            entry_id = entry["id"]

            # Some genes will have ',' at the end if there were more than one gene, remove that
            entry_name = re.sub(",", "", entry_name)

            # Map the id of the entry to the entry name
            id_to_name[entry_id] = entry_name
            # logging.info(f'Entry name: {entry_name} ID: {entry_id}')

            # If the entry is a pathway, store the pathway name
            if entry_type == "path":
                if entry_name not in subpaths:
                    subpaths.append(entry_name)
                

            # If the entry type is a gene group, find all component ids and add them to the id dictionary for the entry
            if entry_type == "group":
                group_ids = []
                for component in entry.find_all("component"):
                    group_ids.append(component["id"])
                groups[entry_id] = group_ids
            
            # If the entry is not a group, add its attributes to the graph
            else:
                graph.add_node(entry_name, name=entry_name, type=entry_type)

        # For each of the relationships
        for relation in soup.find_all("relation"):
            (color, signal) = ("black", "a")

            relation_entry1 = relation["entry1"] # Upstream node
            relation_entry2 = relation["entry2"] # Target node
            relation_type = relation["type"] # Type of relationship (PPel, GEcrel, etc.)
    
            subtypes = []

            # Relationship subtypes tell you about whats going on
            for subtype in relation.find_all("subtype"):
                subtypes.append(subtype["name"])
    
            if (
                ("activation" in subtypes)
                or ("expression" in subtypes)
                or ("glycosylation" in subtypes)
            ):
                color = "green"
                signal = "a"
            elif ("inhibition" in subtypes) or ("repression" in subtypes):
                color = "red"
                signal = "i"
            elif ("binding/association" in subtypes) or ("compound" in subtypes):
                color = "purple"
                signal = "a"
            elif "phosphorylation" in subtypes:
                color = "orange"
                signal = "a"
            elif "dephosphorylation" in subtypes:
                color = "pink"
                signal = "i"
            elif "indirect effect" in subtypes:
                color = "cyan"
                signal = "a"
            elif "dissociation" in subtypes:
                color = "yellow"
                signal = "i"
            elif "ubiquitination" in subtypes:
                color = "cyan"
                signal = "i"
            else:
                logging.debug("color not detected. Signal assigned to activation arbitrarily")
                logging.debug(subtypes)
                signal = "a"

            # For entries that are a group of genes, get a list of all of the sub-id's in that group
            entry1_list = self.expand_groups(relation_entry1, groups)
            entry2_list = self.expand_groups(relation_entry2, groups)

            # Find all connections between objects in the groups and add them to the grapgh
            for (entry1, entry2) in itertools.product(entry1_list, entry2_list):
                node1 = id_to_name[entry1]
                node2 = id_to_name[entry2]
                # if (node1.count('-') < 10 or node2.count('-') < 10):
                #     logging.info(f'{node1} --- {signal} ---> {node2}\n\t{"/".join(subtypes)}')
                graph.add_edge(
                    node1,
                    node2,
                    color=color,
                    subtype="/".join(subtypes),
                    type=relation_type,
                    signal=signal,
                )
        
        # ------------------ UNCOMMENTING THE FOLLOWING LOADS ALL REFERENCED SUBGRAPHS, WORK IN PROGRESS -------------------------------
        # logging.info(f'Subpaths:')
        # for path_name in subpaths:
        #     logging.info(f'\t{path_name}')

        # num_pathways = len(subpaths)
        # for pathway_num, pathway in enumerate(subpaths):
        #     for xml_file in os.listdir(f'{file_paths["pathway_xml_files"]}/{self.organism}'):
        #         xml_pathway_name = xml_file.split('.')[0]
        #         if pathway == xml_pathway_name:
        #             with open(f'{file_paths["pathway_xml_files"]}/{self.organism}/{xml_file}', 'r') as pathway_file:
        #                 text = [line for line in pathway_file]
        #             soup = BeautifulSoup("".join(text), "xml")
        #             for entry in soup.find_all("entry"):
        #                 # logging.info(f'\nEntry:')
        #                 # logging.info(f'\t{entry}')

        #                 # Name of each gene in the entry
        #                 # If there are multiple genes in the entry, store them all with the same id
        #                 entry_split = entry["name"].split(":")

        #                 # logging.info(f'\tentry_split: {entry_split}')
        #                 # logging.info(f'\tlen(entry_split) : {len(entry_split)}')
        #                 # If the entry is part of a group (in the network coded by a group containing lots of related genes)
        #                 if len(entry_split) > 2:

        #                     # Choose which dictionary to use based on whether the entries are hsa or kegg elements
        #                     # Entries with hsa correspond to genes, entries with ko correspond to orthologs
        #                     if entry_split[0] == "hsa" or entry_split[0] == "ko":
        #                         if entry_split[0] == "hsa":
        #                             useDict = hsaDict
        #                         elif entry_split[0] == "ko":
        #                             useDict = KEGGdict
        #                         nameList = []
                                
        #                         # Split off the first name
        #                         entry_name = ""
        #                         namer = entry_split.pop(0)
        #                         namer = entry_split.pop(0)
        #                         namer = namer.split()[0]

        #                         # Either use the dictionary name for the key or use the name directly if its not in the dictionary
        #                         entry_name = (
        #                             entry_name + useDict[namer]
        #                             if namer in useDict.keys()
        #                             else entry_name + namer
        #                         )

        #                         # Append each gene name to the list ([gene1, gene2])
        #                         for i in range(len(entry_split)):
        #                             nameList.append(entry_split[i].split()[0])

        #                         # For each of the gene names, separate them with a "-" (gene1-gene2)
        #                         for namer in nameList:
        #                             entry_name = (
        #                                 entry_name + "-" + useDict[namer]
        #                                 if namer in useDict.keys()
        #                                 else entry_name + "-" + namer
        #                             )
        #                         entry_type = entry["type"]
        #                     else:
        #                         entry_name = entry["name"]
        #                         entry_type = entry["type"]
                        
        #                 # If there is only one name
        #                 else:
        #                     # If the name is hsa
        #                     if entry_split[0] == "hsa":
        #                         entry_name = entry_split[1] # Get the entry number
        #                         entry_type = entry["type"] # Get the entry type
        #                         entry_name = ( # Get the gene name from the entry number if its in the hsa gene name dict
        #                             hsaDict[entry_name] if entry_name in hsaDict.keys() else entry_name
        #                         )
        #                     # If the name is ko, do the same as above but use the KEGGdict instead of the hsa gene name dict
        #                     elif entry_split[0] == "ko":
        #                         entry_name = entry_split[1]
        #                         entry_type = entry["type"]
        #                         entry_name = (
        #                             KEGGdict[entry_name]
        #                             if entry_name in KEGGdict.keys()
        #                             else entry_name
        #                         )
        #                     # If the entry is another KEGG pathway number, store the name of the signaling pathway
        #                     elif entry_split[0] == "path":
        #                         entry_name = entry_split[1]
        #                         entry_type = "path"
        #                     # If none of the above, just store the name and type
        #                     else:
        #                         entry_name = entry["name"]
        #                         entry_type = entry["type"]
                        
        #                 # Get the unique entry ID for this pathway
        #                 entry_id = entry["id"]

        #                 # Some genes will have ',' at the end if there were more than one gene, remove that
        #                 entry_name = re.sub(",", "", entry_name)

        #                 # Map the id of the entry to the entry name
        #                 id_to_name[entry_id] = entry_name
        #                 # logging.info(f'Entry name: {entry_name} ID: {entry_id}')

        #                 # # If the entry is a pathway, store the pathway name
        #                 # if entry_type == "path":
        #                 #     if entry_name not in subpaths:
        #                 #         subpaths.append(entry_name)
                            

        #                 # If the entry type is a gene group, find all component ids and add them to the id dictionary for the entry
        #                 if entry_type == "group":
        #                     group_ids = []
        #                     for component in entry.find_all("component"):
        #                         group_ids.append(component["id"])
        #                     groups[entry_id] = group_ids
                        
        #                 # If the entry is not a group, add its attributes to the graph
        #                 else:
        #                     graph.add_node(entry_name, name=entry_name, type=entry_type)

        #             # For each of the relationships
        #             for relation in soup.find_all("relation"):
        #                 # logging.info(f'Relation:')
        #                 # logging.info(f'\t{relation}')
        #                 (color, signal) = ("black", "a")

        #                 relation_entry1 = relation["entry1"] # Upstream node
        #                 relation_entry2 = relation["entry2"] # Target node
        #                 relation_type = relation["type"] # Type of relationship (PPel, GEcrel, etc.)
                
        #                 subtypes = []

        #                 # Relationship subtypes tell you about whats going on
        #                 for subtype in relation.find_all("subtype"):
        #                     subtypes.append(subtype["name"])
                
        #                 if (
        #                     ("activation" in subtypes)
        #                     or ("expression" in subtypes)
        #                     or ("glycosylation" in subtypes)
        #                 ):
        #                     color = "green"
        #                     signal = "a"
        #                 elif ("inhibition" in subtypes) or ("repression" in subtypes):
        #                     color = "red"
        #                     signal = "i"
        #                 elif ("binding/association" in subtypes) or ("compound" in subtypes):
        #                     color = "purple"
        #                     signal = "a"
        #                 elif "phosphorylation" in subtypes:
        #                     color = "orange"
        #                     signal = "a"
        #                 elif "dephosphorylation" in subtypes:
        #                     color = "pink"
        #                     signal = "i"
        #                 elif "indirect effect" in subtypes:
        #                     color = "cyan"
        #                     signal = "a"
        #                 elif "dissociation" in subtypes:
        #                     color = "yellow"
        #                     signal = "i"
        #                 elif "ubiquitination" in subtypes:
        #                     color = "cyan"
        #                     signal = "i"
        #                 else:
        #                     logging.debug("color not detected. Signal assigned to activation arbitrarily")
        #                     logging.debug(subtypes)
        #                     signal = "a"

        #                 # For entries that are a group of genes, get a list of all of the sub-id's in that group
        #                 entry1_list = self.expand_groups(relation_entry1, groups)
        #                 entry2_list = self.expand_groups(relation_entry2, groups)

        #                 # Find all connections between objects in the groups and add them to the grapgh
        #                 for (entry1, entry2) in itertools.product(entry1_list, entry2_list):
        #                     node1 = id_to_name[entry1]
        #                     node2 = id_to_name[entry2]
        #                     # if (node1.count('-') < 10 or node2.count('-') < 10):
        #                     # logging.info(f'{node1} --- {signal} ---> {node2}\n\t{"/".join(subtypes)}')
        #                     graph.add_edge(
        #                         node1,
        #                         node2,
        #                         color=color,
        #                         subtype="/".join(subtypes),
        #                         type=relation_type,
        #                         signal=signal,
        #                     )


        ### --------------------------------------------------------------------------------------------------


        # self.add_pathways(subpath_graphs, minOverlap=25, organism=self.organism)

        return graph
    

    def write_xml_files(self, organism: str, pathway_list: list):
        """
        Reads in all xml files for the organism, faster to do this once at the start and just use
        the cached files. They aren't that big, so I'd rather store them at the beginning.
        """
        
        # Function to silence KEGG initialization to the terminal
        def silent_kegg_initialization():
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = open('/dev/null', 'w')
                sys.stderr = open('/dev/null', 'w')
                kegg = KEGG(verbose=False) 
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = original_stdout
                sys.stderr = original_stderr
            return kegg
    
        k = silent_kegg_initialization()  # read KEGG from bioservices
        k.organism = organism
        
        logging.info(f'\t\tDownloading pathway files, this may take a while...')
        with alive_bar(len(pathway_list)) as bar:
            for pathway in pathway_list:
                pathway = pathway.replace("path:", "")
                code = str(pathway)
                code = re.sub(
                    "[a-zA-Z]+", "", code
                )  # eliminate org letters - retain only numbers from KEGG pathway codes
                origCode = code

                code = str("ko" + code)  # add ko
                os.makedirs(f'{file_paths["pathway_xml_files"]}/{organism}/', exist_ok=True)

                # If the pathway is not in the list of xml files, find it and create it
                if f'{code}.xml' not in os.listdir(f'{file_paths["pathway_xml_files"]}/{organism}/'):
                    logging.debug(f'\t\t\tFinding xml file for pathway ko{origCode} and {organism}{origCode}')

                    # Write out the ko pathway xml files
                    try:
                        with open(f'{file_paths["pathway_xml_files"]}/{organism}/{code}.xml', 'w') as pathway_file:
                            url = requests.get(
                                "http://rest.kegg.jp/get/" + code + "/kgml", stream=True
                            )
                            [pathway_file.write(line.decode("utf-8")) for line in url.iter_lines()]

                    except:
                        logging.debug("could not read code: " + code)
                        continue
                    
                    # Write out the organism pathway xml files
                    code = str(organism + origCode)  # set up with org letters

                    try:
                        with open(f'{file_paths["pathway_xml_files"]}/{organism}/{code}.xml', 'w') as pathway_file:
                            url = requests.get(
                                "http://rest.kegg.jp/get/" + code + "/kgml", stream=True
                            )
                            [pathway_file.write(line.decode("utf-8")) for line in url.iter_lines()]

                    except:
                        logging.debug("could not read code: " + code)
                        continue
                bar()
    
    def parse_kegg_pathway(self, graph, minimumOverlap, pathway, pathway_num, num_pathways):
        """
        Read in and format the KEGG pathway
        """

        pathway = pathway.replace("path:", "")
        code = str(pathway)
        code = re.sub(
            "[a-zA-Z]+", "", code
        )  # eliminate org letters - retain only numbers from KEGG pathway codes
        origCode = code

        coder = str("ko" + code)  # add ko

        # remove complexes and rewire components
        removeNodeList = [gene for gene in graph.nodes() if "-" in gene]
        for rm in removeNodeList:
            for start in graph.predecessors(rm):
                edge1 = graph.get_edge_data(start, rm)["signal"]
                if edge1 == "i":
                    for element in rm.split("-"):
                        graph.add_edge(start, element, signal="i")
                else:
                    for element in rm.split("-"):
                        graph.add_edge(start, element, signal="a")
            for finish in graph.successors(rm):
                edge2 = graph.get_edge_data(rm, finish)["signal"]
                if edge2 == "i":
                    for element in rm.split("-"):
                        graph.add_edge(element, finish, signal="i")
                else:
                    for element in rm.split("-"):
                        graph.add_edge(element, finish, signal="a")
            graph.remove_node(rm)

        # remove dependence of nodes on complexes that include that node
        for node in list(graph.nodes()):
            predlist = graph.predecessors(node)
            for pred in predlist:
                if "-" in pred:
                    genes = pred.split("-")
                    flag = True
                    for gene in genes:
                        if not gene in predlist:
                            flag = False
                    if flag:
                        graph.remove_edge(pred, node)

        # remove self edges
        for edge in list(graph.edges()):
            if edge[0] == edge[1]:
                graph.remove_edge(edge[0], edge[1])

        # check to see if there is a connected component, simplify graph and print if so
        pathway_nodes = set(graph.nodes())
        overlap = len(pathway_nodes.intersection(self.gene_list))
        
        # Keep the pathway if there are at least minimumOverlap genes shared between the network and the genes in the dataset
        if overlap > minimumOverlap and len(graph.edges()) > 0:  
            logging.info(f'\t\t\tPathway ({pathway_num}/{num_pathways}): {pathway} Overlap: {overlap} Edges: {len(graph.edges())}')
            nx.write_graphml(graph, self.output_path + coder + ".graphml")

            # Add the pathway graph to the dictionary with the pathway code as the key
            self.pathway_dict[code] = graph

        else:
            logging.debug(f'\t\t\tPathway ({pathway_num}/{num_pathways}): {pathway} not enough overlapping genes (min = {minimumOverlap}, found {overlap})')

    def find_kegg_pathways(self, kegg_pathway_list: list, write_graphml: bool, organism: str, minimumOverlap: int):
        """
        write_graphml = whether or not to write out a graphml (usually true)
        organism = organism code from kegg. Eg human = 'hsa', mouse = 'mus'

        Finds the KEGG pathways from the pathway dictionaries
        """
        logging.info("\t\tFinding KEGG pathways...")
        kegg_dict = self.parse_kegg_dict()  # parse the dictionary of ko codes
        logging.info("\t\t\tLoaded KEGG code dictionary")
        
        pathway_dict_path = f'{file_paths["pathway_xml_files"]}/{organism}_dict.csv'
        aliasDict = {}
        orgDict = {}

        # If the dictionary file exists, use that (much faster than streaming)
        if f'{organism}_dict.csv' in os.listdir(f'{file_paths["pathway_xml_files"]}'):
            logging.info(f'\t\t\tReading {organism} dictionary file...')
            with open(pathway_dict_path, 'r') as kegg_dict_file:
                for line in kegg_dict_file:
                    line = line.strip().split('\t')
                    k = line[0]
                    name = line[1]
                    orgDict[k] = name

        # If the dictionary file does not exist, write it and stream in the data for the dictionary
        else:
            logging.info(f'\t\t\tOrganism dictionary not present for {organism}, downloading...')
            try:  # try to retrieve and parse the dictionary containing organism gene names to codes conversion
                url = requests.get("http://rest.kegg.jp/list/" + organism, stream=True)
                # reads KEGG dictionary of identifiers between numbers and actual protein names and saves it to a python dictionary

                with open(pathway_dict_path, 'w') as kegg_dict_file:
                    for line in url.iter_lines():
                        line = line.decode("utf-8")
                        line_split = line.split("\t")
                        k = line_split[0].split(":")[1]
                        nameline = line_split[3].split(";")
                        name = nameline[0]
                        if "," in name:
                            nameline = name.split(",")
                            name = nameline[0]
                            for entry in range(1, len(nameline)):
                                aliasDict[nameline[entry].strip()] = name.upper()
                        orgDict[k] = name
                        kegg_dict_file.write(f'{k}\t{name}\n')
                url.close()
            except:
                logging.info("Could not get library: " + organism)

        # Writes xml files for the pathways in the pathway list
        self.write_xml_files(organism, kegg_pathway_list)

        xml_file_path = os.listdir(f'{file_paths["pathway_xml_files"]}/{organism}')

        def parse_xml_files(xml_file, pathway_name):
            """
            Reads in the pathway xml file and parses the connections. Creates a networkx directed graph of the pathway
            """
            with open(f'{file_paths["pathway_xml_files"]}/{organism}/{xml_file}', 'r') as pathway_file:
                text = [line for line in pathway_file]

                # Read the kegg xml file
                graph = self.read_kegg(text, nx.DiGraph(), kegg_dict, orgDict)

                # Parse the kegg pathway and determine if there is sufficient overlap for processing with scBONITA
                self.parse_kegg_pathway(graph, minimumOverlap, pathway_name, pathway_num, num_pathways)

        # If there aren't any kegg pathways specified, look for all overlapping pathways
        if len(kegg_pathway_list) == 0:
            # Read in the pre-downloaded xml files and read them into a DiGraph object
            num_pathways = len(xml_file_path)
            logging.info(f'\t\tNo KEGG pathways specified, searching all overlapping pathways')
            logging.info(f'\t\tFinding pathways with at least {minimumOverlap} genes that overlap with the dataset')
            with alive_bar(num_pathways) as bar:
                for pathway_num, xml_file in enumerate(xml_file_path):
                    pathway_name = xml_file.split('.')[0]
                    parse_xml_files(pathway_name)
                    bar()

        # If there are pathways specified by the user, load those in
        else:
            pathway_list = list(kegg_pathway_list)
            num_pathways = len(pathway_list)
            minimumOverlap = 1  # Minimum number of genes that need to be in both the dataset and pathway for the pathway to be considered
            logging.info(f'\t\tFinding pathways with at least {minimumOverlap} genes that overlap with the dataset')

            with alive_bar(num_pathways) as bar:
                for pathway_num, pathway in enumerate(pathway_list):
                    for xml_pathway_name in xml_file_path:
                        if organism + pathway + '.xml' == xml_pathway_name:
                            parse_xml_files(xml_pathway_name, pathway)

                        elif 'ko' + pathway + '.xml'== xml_pathway_name:
                            parse_xml_files(xml_pathway_name, pathway)
                        
                    bar()

        if len(self.pathway_dict.keys()) == 0:
            msg = f'WARNING: No pathways passed the minimum overlap of {minimumOverlap}'
            assert Exception(msg)
        
        return self.pathway_dict

    def add_pathways(self, pathway_list, minOverlap, write_graphml=True, removeSelfEdges=False, organism='hsa'):
        """
        Add a list of pathways in graphml format to the rule_inference object

        Writes out the "_processed.graphml" files
        """

        logging.info(f'\t\tAdding graphml pathways to rule_inference object...')

        # Get a list of the genes in the dataset
        if hasattr(self, "cv_genes"):
            pathway_genes = set(self.cv_genes)
        elif not hasattr(self, "cv_genes"):
            pathway_genes = set(self.gene_list)

        def create_processed_networkx_graphml(G, pathway):
            """
            Reads in the graph and the pathway and filters out self edges and isolates

            Creates the "_processed.graphml" files
            """
            nodes = set(G.nodes())

            # Compute the number of nodes that overlap with the pathway genes
            overlap = len(nodes.intersection(pathway_genes))

            # Check to see if there are enough genes in the dataset that overlap with the genes in the pathway
            if overlap >= minOverlap:

                logging.info(f'\t\tPathway: {pathway} Overlap: {overlap} Edges: {len(G.edges())}')
                nodes = list(G.nodes())

                if removeSelfEdges:
                    G.remove_edges_from(nx.selfloop_edges(G))  # remove self loops
                # remove genes not in dataset
                for pg in list(G.nodes()):
                    if pg not in pathway_genes:
                        G.remove_node(pg)


                # graph post-processing
                # remove singletons/isolates
                G.remove_nodes_from(list(nx.isolates(G)))

                self.pathway_graphs[pathway] = G
                logging.info(f'\t\t\tEdges after processing: {len(G.edges())} Overlap: {len(set(G.nodes()).intersection(pathway_genes))}')
                filtered_overlap = len(set(G.nodes()).intersection(pathway_genes))

                if write_graphml and filtered_overlap > minOverlap:
                    # Start the output file path with the otuput path
                    output_file_name = self.output_path
                    
                    # Add the organism to the pathway file if its not already there
                    if organism not in pathway:
                        output_file_name = output_file_name + organism
                    
                    # Add the pathway name to the output name
                    output_file_name = output_file_name + pathway
                    
                    # Only adds if not using _processed.graphml
                    if "_processed.graphml" not in output_file_name:
                        output_file_name = output_file_name + "_processed.graphml"

                    nx.write_graphml(
                        G, output_file_name, infer_numeric_types=True
                    )
                
                
            else:
                msg = f'Overlap {overlap} is below the minimum {minOverlap}'
                raise Exception(msg)

        # Create the "_processed.graphml" files

        # If pathway_list is a list
        if isinstance(pathway_list, list):
            for pathway in pathway_list:  
                if os.path.exists(pathway):
                    G = nx.read_graphml(pathway)
                else:
                    custom_graphml_path = f'{file_paths["custom_graphml"]}/{pathway}'
                    G = nx.read_graphml(custom_graphml_path)
                create_processed_networkx_graphml(G, pathway)
                

        # If pathway_list is a dictionary
        elif isinstance(pathway_list, dict):
                for pathway, G in pathway_list.items():
                    create_processed_networkx_graphml(G, pathway)
        

