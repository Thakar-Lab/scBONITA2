import numpy as np
import networkx as nx
from scipy.stats.stats import spearmanr
import ctypes as ctypes

from setup.parameters import Params
from parse_node import Node
import logging
from alive_progress import alive_bar

class NetworkSetup:
    def __init__(self, graph):
        """Initialize a NetworkSetup object for rule inference with scBONITA - RD"""

        # Initialize lists to store information about nodes and connections
        self.totalNodeList = []
        self.andNodeList = []
        self.andNodeInvertList = []
        self.andLenList = []
        self.totalLenList = []
        self.permList = []
        self.rvalues = []
        self.predecessors = []
        self.successorNums = []
        self.graph = graph
        self.params = Params()

        # Initialize node attributes
        self.nodeList = list(graph.nodes)  # List of nodes in the graph
        self.nodeDict = {self.nodeList[i]: i for i in range(len(self.nodeList))}  # Dictionary for node lookup
        
        # Initialize an empty directed graph
        self.ruleGraph = nx.empty_graph(0, create_using=nx.DiGraph)

        # Calculate the node information
        self.calculate_node_information()

        # Create Node objects containing the calculated information for each node in the network
        self.nodes, self.deap_individual_length = self.create_nodes()

        # Print information about the network (optional)
        self.print_network_information()

        self.print_node_information()

    def print_node_information(self):
        for node in self.nodes:
            logging.debug(f'\nNode {node.name}:')
            logging.debug(f'\tpredecessors {node.predecessors}')
            logging.debug(f'\tinversions {node.inversions}')
            logging.debug(f'\trule_length {len(node.possibilities)}')
            logging.debug(f'\trule_start_index {node.rule_start_index}')
            logging.debug(f'\trule_end_index {node.rule_end_index}')

    def print_network_information(self):
        """Prints information about the network."""
        logging.debug("\nNodelist: " + str(self.nodeList))
        logging.debug("\nNode positions: " + str(self.node_positions))
        logging.debug(f"\ntotalNodeList: {self.totalNodeList}")

    # -------- Parse Node Information for Rule Inference --------
    # 1. Calculates the node information and creates Node objects storing the information for each node
    def calculate_node_information(self):
        """
        Calculates the information for each node in the network and stores the information as object of class Node
        from parse_node.py
        """
        # Iterate over all nodes to find predecessors and calculate possible connections
        for i, node in enumerate(self.nodeList):
            predecessors_final = self.find_predecessors(self.ruleGraph, self.nodeList, self.graph, self.nodeDict, [], i)
            node_predecessors = [self.nodeList.index(corr_tuple[0]) for corr_tuple in predecessors_final]
            self.predecessors.append(node_predecessors)

    # 1.1 Removes self-loops from the nodes in the graph
    def remove_self_edges(self, removeSelfEdges, nodeList, graph):
        """
        Remove self loops from the graph
        """
        if removeSelfEdges:
            for node in nodeList:
                repeat = True
                while repeat:
                    repeat = False
                    if node in list(graph.successors(node)):
                        graph.remove_edge(node, node)
                        repeat = True
                    if node in list(graph.predecessors(node)):
                        graph.remove_edge(node, node)
                        repeat = True

    # 1.2 Finds the predecessors of each node and stores the top 3
    def find_predecessors(self, ruleGraph, nodeList, graph, nodeDict, possibilityLister, node):
        """
        Find the incoming nodes for each ndoe in the graph, store the top 3 connections as calculated by a spearman
        correlation
        Parameters
        ----------
        ruleGraph
        nodeList
        graph
        nodeDict
        possibilityLister
        node

        Returns
        -------

        """
        # --- Find the predecessors of each node ---
        predecessors_final, predCorr_temp = self.parse_connections(node, nodeList, graph, nodeDict, possibilityLister)

        # Store the correlations between incoming nodes in "rvalues"
        top_three_incoming_node_correlations = sorted(predCorr_temp, reverse=True)[:3]
        self.rvalues.append(top_three_incoming_node_correlations)

        # Append the permanent list with the top 3 predecessors for this node
        self.permList.append([pred[0] for pred in predecessors_final])

        # Add the incoming nodes and their properties to the newly created ruleGraph
        self.store_predecessors(graph, nodeList, predecessors_final, ruleGraph, node)
        return predecessors_final

    # 1.2.1 Finds the top 3 incoming nodes using Spearman correlation for the current node and stores that info
    def parse_connections(self, node_index, nodeList, graph, nodeDict, possibilityLister):
        """
        Find the top 3 incoming nodes for the node of interest and store the information (part of find_predecessors)
        """
        # Get NAMES of predecessors and successors of the node from original graph
        
        predecessors_temp = list(graph.predecessors(nodeList[node_index]))
        
        successors_temp = list(graph.successors(nodeList[node_index]))
        self.successorNums.append(len(successors_temp))

        possibilitytemp = [nodeDict[predder] for predder in predecessors_temp if nodeDict[predder] != node_index]
        possibilityLister.append(list(possibilitytemp))

        # Calculate the Spearman correlation for the incoming nodes
        predCorr_temp = self.calculate_spearman_correlation(node_index, predecessors_temp)

        # Select the top 3 predecessors of the node
        predecessors_final = sorted(
            zip(predecessors_temp, predCorr_temp),
            reverse=True,
            key=lambda corrs: corrs[1], )[:3]
        
        return predecessors_final, predCorr_temp
    
    def calculate_spearman_correlation(self, node, predecessors_temp):
        """
        Calculate the Spearman correlation between incoming nodes to find the top three with the
        highest correlation, used to reduce the dimensionality of the calculations.

        1. calculate_node_information
            1.2 find_predecessors
                1.2.1 parse_connections
                    1.2.1.1 calculate_spearman_correlation
        """
        # Find correlation between the predecessors and the node
        nodeData = (
            self.binarized_matrix[self.node_positions[node], :].todense().tolist()[0]
        )  # find binarized expression data for node "i"
        predCorr_temp = (
            []
        )  # temporarily store correlations between node "i" and all its predecessors

        for predecessor_gene in predecessors_temp:
            # find index of predecessor in the gene_list from the data
            predIndex = self.nodeList.index(predecessor_gene)

            # find binarized expression data for predecessor
            predData = (self.binarized_matrix[predIndex, :].todense().tolist()[0])
            mi, pvalue = spearmanr(nodeData, predData)

            if np.isnan(mi):
                predCorr_temp.append(0)
            else:
                predCorr_temp.append(mi)  # store the calculated correlation
        return predCorr_temp

    # 1.2.2 Stores the top 3 interactions in a new graph
    def store_predecessors(self, graph, nodeList, predecessors_final, ruleGraph, node):
        """
        Stores the information about each incoming node for each node in the graph to a new
        graph.
        """
        for parent in predecessors_final:
            if "interaction" in list(graph[parent[0]][nodeList[node]].keys()):
                ruleGraph.add_edge(
                    parent[0],
                    nodeList[node],
                    weight=parent[1],
                    activity=graph[parent[0]][nodeList[node]]["interaction"],
                )
            if "signal" in list(graph[parent[0]][nodeList[node]].keys()):
                ruleGraph.add_edge(
                    parent[0],
                    nodeList[node],
                    weight=parent[1],
                    activity=graph[parent[0]][nodeList[node]]["signal"],
                )

    # 1.5.1 Calculates the inversion rules for each combination
    def calculate_inversion_rules(self, node_predecessors, node_index):
        """
        Calculates the inversion rules for a node based on the graph interactions or signal for each incoming node
        Parameters
        ----------
        node

        Returns
        -------
        inversion_rules
        """

        inversion_rules = {}
        for incoming_node in list(node_predecessors):
            edge_attribute = list(self.graph[self.nodeList[incoming_node]][self.nodeList[node_index]].keys())

            # check the 'interaction' edge attribute
            if "interaction" in edge_attribute:
                if self.graph[self.nodeList[incoming_node]][self.nodeList[node_index]]["interaction"] == "i":
                    inversion_rules[incoming_node] = True
                else:
                    inversion_rules[incoming_node] = False

            # check the 'signal' edge attribute
            elif "signal" in edge_attribute:
                if self.graph[self.nodeList[incoming_node]][self.nodeList[node_index]]["signal"] == "i":
                    inversion_rules[incoming_node] = True
                else:
                    inversion_rules[incoming_node] = False
            
            # for some reason, when I used a modified processed graphml file as a custom graphml file I needed to use this method
            else:
                for _, value in self.graph[self.nodeList[incoming_node]][self.nodeList[node_index]].items():
                    for attribute, value in value.items():
                        if attribute == "signal" or "interaction":
                            if value == "i":
                                inversion_rules[incoming_node] = True
                            else:
                                inversion_rules[incoming_node] = False

        return inversion_rules

    # 1.6 Creates nodes containing the information calculated from the graph
    def create_nodes(self):
        """
        Creates Node class objects using the information calculated in the rest of the calculate_node_information
        function
        """
        inverted_nodeDict = {v: k for k, v in self.nodeDict.items()}
        gene_name_to_index = {gene_name: gene_index for gene_index, gene_name in enumerate(self.gene_names)}

        nodes = []
        rule_index = 0
        with alive_bar(len(self.nodeList)) as bar:
            for node_index, node_name in enumerate(self.nodeList):
                name = node_name
                # Safely retrieve predecessors and put them into a dictionary where key = node index, value = node name
                predecessor_indices = self.predecessors[node_index] if node_index < len(self.predecessors) else []
                predecessors = {}
                for index in predecessor_indices:
                    inverted_nodeDict = {v: k for k, v in self.nodeDict.items()}
                    predecessors[index] = inverted_nodeDict[index]
                
                node_inversions = self.calculate_inversion_rules(predecessors, node_index)
                # Create a new Node object
                node = Node(name, node_index, predecessors, node_inversions)

                # Find the dataset row index of the gene
                node.dataset_index = gene_name_to_index.get(node_name)

                node.rvalues = self.rvalues[node_index]

                # Find the start and end indices for where the rule combinations start and stop for this node in the
                rule_length = len(node.bitstring)
                if rule_length > 0:
                    node.rule_start_index = rule_index
                    rule_index += rule_length
                    node.rule_end_index = rule_index
                else:
                    node.rule_start_index = None
                    node.rule_end_index = None
                nodes.append(node)
                bar()
        return nodes, rule_index