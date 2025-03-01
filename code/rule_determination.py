from statistics import stdev
import logging
from itertools import product
import numpy as np
from alive_progress import alive_bar
import os
from file_paths import file_paths

class RuleDetermination:
    def __init__(self,network,network_name,dataset_name,binarized_matrix,nodes,node_dict):
        # Pass in the node objects
        self.nodes = nodes
        self.node_dict = node_dict

        # General parameters
        self.network = network
        self.network_name = network_name
        self.dataset_name = dataset_name
        self.binarized_matrix = binarized_matrix

        logging.basicConfig(format='%(message)s', level=logging.INFO)

        _, num_columns = np.shape(self.binarized_matrix)

        # Chunk to reduce noise if there are a lot of cells, otherwise just use the columns
        if num_columns > 2000:
            self.num_chunks = round(num_columns / 10)
            self.chunked_data_numpy = np.array(self.chunk_data(num_chunks=self.num_chunks))
            self.coarse_chunked_dataset = np.array(self.chunk_data(num_chunks=round(self.num_chunks / 2, 1)))
        else:
            self.num_chunks = num_columns
            self.chunked_data_numpy = np.array(self.chunk_data(num_chunks=num_columns))
            self.coarse_chunked_dataset = np.array(self.chunk_data(num_chunks=num_columns))

    def infer_ruleset(self):
        num_rows, num_columns = np.shape(self.binarized_matrix)
        chunked_rows, chunked_columns = np.shape(self.chunked_data_numpy)

        if chunked_columns < num_columns:
            logging.info(f'\n-----CHUNKING DATASET-----')
            logging.info(f"\tOriginal Data Shape: {num_rows} rows, {num_columns} columns")
            logging.info(f"\tChunked Data Shape: {chunked_rows} rows, {chunked_columns} columns")

        # Find all possible rule predictions for each node
        for node in self.nodes:
            node.find_all_rule_predictions()

        # Refine the rules
        best_ruleset, ruleset_errors, node_errors = self.refine_rules()

        # Write the ruleset out to a file
        logging.info(f'\n-----RULESET-----')
        logging.info(f'(Genes that signal to themselves not shown)')
        self.write_ruleset(best_ruleset, ruleset_errors[0], node_errors)
        
        # Set the best rules for the nodes based on which ruleset has the lowest error
        for node_index, node in enumerate(self.nodes):
            node_best_rule = best_ruleset[node_index][0]
            node_best_rule_index = best_ruleset[node_index][1]

            node.best_rule = node_best_rule
            node.best_rule_index = node_best_rule_index
            logging.debug(f'Node {node.name}')
            logging.debug(f'\tnode best rule {node.best_rule}')
            logging.debug(f'\tnode best rule index {node.best_rule_index}')

        logging.info(f'Note: rules with high error likely have incoming nodes from other pathways')
        
        return best_ruleset


    def write_ruleset(self, ruleset, error, node_errors):
        # Write out the rules to an output file
        rule_path = f'{file_paths["rules_output"]}/{self.dataset_name}_rules/{self.network_name}_{self.dataset_name}_rules.txt'
        os.makedirs(f'{file_paths["rules_output"]}/{self.dataset_name}_rules/', exist_ok=True)
        with open(rule_path, 'w') as rule_file:
            
            # Reverses the node_dict dictionary for easy node name lookup by index
            reversed_node_dict = {}
            for key, value in self.node_dict.items():   
                reversed_node_dict[value] = key         

            rule_index = 0
            for rule, _, _ in ruleset:
                rule_name = rule[0]
                incoming_nodes = rule[1]
                incoming_node_names = [reversed_node_dict[node] for node in incoming_nodes]
                logic = rule[2]

                # Replaces ABC with the correct gene names
                def replace_placeholders(rule, gene_names):
                    # Create a dictionary to map placeholders to gene names
                    placeholder_mapping = {}
                    if len(gene_names) > 0:
                        placeholder_mapping['A'] = gene_names[0]
                    if len(gene_names) > 1:
                        placeholder_mapping['B'] = gene_names[1]
                    if len(gene_names) > 2:
                        placeholder_mapping['C'] = gene_names[2]

                    # Replace placeholders with temporary placeholders to avoid conflicts
                    temp_rule = rule.replace('A', '{A_TEMP}').replace('B', '{B_TEMP}').replace('C', '{C_TEMP}')

                    # Perform the final replacement
                    for placeholder, gene_name in placeholder_mapping.items():
                        temp_rule = temp_rule.replace(f'{{{placeholder}_TEMP}}', gene_name)

                    return temp_rule

                rule_with_gene_names = replace_placeholders(logic, incoming_node_names)

                line = f'error: {round(node_errors[rule_index]*100)}%\t{rule_name} = {rule_with_gene_names}'

                # Don't write out self-loops
                if not rule_name == rule_with_gene_names:
                    logging.info(line)
                rule_file.write(line)
                rule_file.write('\n')

                rule_index += 1

            avg_error = error[0]
            stdev_error = error[1]
            max_error = error[2]
            min_error = error[3]

            logging.info(f'Refined Error:\n')
            logging.info(f'\tAverage = {avg_error}')
            logging.info(f'\tStdev = {stdev_error}')
            logging.info(f'\tMax = {max_error}')
            logging.info(f'\tMin = {min_error}')
            rule_file.write(f'Refined_error:\tavg={avg_error}|stdev={stdev_error}|max={max_error}|min={min_error}')


    def refine_rules(self):
        logging.info(f'\n-----RULE REFINEMENT-----')
        ruleset_error = []

        best_rules = []
        num_self_loops = 0

        with alive_bar(len(self.nodes)) as bar:
            for node in self.nodes:
                # If the node signals to itself, create that rule
                if node.node_rules[0][1][0] == node.index:
                    best_rule = self.handle_self_loop(node)
                    best_rules.append(best_rule) # Add the rule to the list of best rules
                    num_self_loops += 1

                # If there are no incoming nodes, set the node to signal to itself
                elif len(node.node_rules) == 0:
                    best_rule = self.handle_self_loop(node)
                    best_rules.append(best_rule) # Add the rule to the list of best rules
                    num_self_loops += 1

                # If the rule has multiple incoming nodes, calculate the minimum error rule
                else:
                    equivalent_best_rules = self.calculate_refined_errors(node, self.chunked_data_numpy)

                    # If the best rule has a high error, it likely responds to a gene not in the KEGG pathway
                    # It's better to use its expression for other rules, but not use upstream genes to change it
                    if equivalent_best_rules[0][2] > 0.20:
                        best_rule = self.handle_self_loop(node)
                    else:
                        best_rule = equivalent_best_rules[0]
                    best_rules.append(best_rule) # Add the first equivalent rule to the best_rules
                bar()

            # Calculates summary stats for the rules
            errors = [error for _, _, error in best_rules]
            average_error = sum(errors) / len(errors)
            stdev_error = stdev(errors)
            max_error = max(errors)
            min_error = min(errors)
            ruleset_error.append([average_error, stdev_error, max_error, min_error, num_self_loops])
            

        return (best_rules, ruleset_error, errors)


    def chunk_data(self, num_chunks):
        """
        Chunks the data by breaking the data into num_chunks number of chunks and taking the average value of all
        columns (cells) within the chunk for each row (genes). For each row, if the average value of all cells in
        the chunk is > 0.5, the chunk is set to 1 for that row. If the average is < 0.5, the chunk is set to 0.
        :param binarized_matrix:
        :param num_chunks:
        :return chunked_data:
        """

        num_chunks = int(num_chunks)

        # Shuffle the columns to randomize the order of cells in the chunks
        np.random.seed(42)  # Optional: for reproducibility. Will shuffle in the same way every time
        # chunked_data = self.binarized_matrix.todense()

        column_permutation = np.random.permutation(self.binarized_matrix.shape[1])
        shuffled_binarized_matrix = self.binarized_matrix[:, column_permutation]

        # Get the shape of the data
        num_rows, num_columns = np.shape(shuffled_binarized_matrix)

        # Define the chunk size by the number of cells and number of chunks
        # Ensure num_chunks is not greater than num_columns
        if num_chunks > num_columns:
            num_chunks = num_columns

        # Calculate chunk size
        chunk_size = max(1, num_columns // num_chunks)  # Ensure chunk_size is at least 1

        # Initiate a blank matrix filled with 0's of size: number of genes (rows) x number of chunks (columns)
        chunked_data = np.zeros((int(num_rows), int(num_chunks)))

        # Iterate through each of the chunks and take the average value of all columns within the chunks
        for chunk_index in range(int(num_chunks)):
            # Define the start and end column for this chunk
            start_column = chunk_index * chunk_size  # Current chunk * chunk size = start
            end_column = start_column + chunk_size  # End column defined by the start + chunk size

            # Find the columns between the start and end columns
            subset = shuffled_binarized_matrix[:, start_column:end_column]

            # Get the average row value for all columns in the chunk
            row_avg = np.mean(subset, axis=1)

            # Binarize the data
            row_avg[row_avg >= 0.5] = 1
            row_avg[row_avg < 0.5] = 0

            # Flattens the chunked array so that it is one column, add it to the chunked_data at the right index
            chunked_data[:, chunk_index] = row_avg.flatten()

        return chunked_data

    @staticmethod
    def handle_self_loop(node):
        """
        Handles instances when a node only connects to itself or has
        no incoming nodes

        return best_rule
        """
        # If the node has only an incoming connection from itself, skip the rule refinement
        # Add in self signaling for the node
        node.node_rules = [node.name, [node.index], 'A']
        node.predecessors[node.index] = node.name
        node.inversions[node.index] = False

        # Set the node's incoming node rules to itself
        best_rule = ([node.name, [node.index], 'A'], 0, 0)

        return best_rule

    @staticmethod
    def generate_not_combinations(rule):
        """
        Finds all possible combinations of rules with and without "not" (input rules in ABC format)

        return all_not_combinations
        """

        variables = []

        if 'A' in rule:
            variables.append('A')
        if 'B' in rule:
            variables.append('B')
        if 'C' in rule:
            variables.append('C')
        if 'D' in rule:
            variables.append('D')

        # Remove the existing 'not' statements
        rule = rule.replace('not ', '')

        # Generate all possible combinations of the variables and their negations
        combinations = list(product(*[(var, f'not {var}') for var in variables]))

        all_not_combinations = []
        for combo in combinations:
            # Create a temporary rule with the current combination
            temp_rule = rule
            for var, replacement in zip(variables, combo):
                # Replace variables with the current combination of variable or its negation
                temp_rule = temp_rule.replace(var, replacement)
            all_not_combinations.append(temp_rule)

        return all_not_combinations

    @staticmethod
    def prioritize_pkn_inversion_rules(node, rules, prediction_errors):
        """
        Prioritizes the rules that follow the inversion rules from the PKN (found in node.inversions). However, if the
        average error for all of the rules in the PKN is too high, then all possible rules are considered

        returns rules, prediction_errors
        """

        # Prioritize the rules that follow the node's inversion rules
        follows_inversion_rules = []
        does_not_follow_inversion_rules = []

        # Determines if the possibility follows the inversion rules from the PKN network
        for rule, error in zip(rules, prediction_errors):
            incoming_node_indices = rule[1]
            logic = rule[2]
            inversion_rules = node.inversions

            def not_in_inversion_rules():
                if (rule, error) not in does_not_follow_inversion_rules:
                    does_not_follow_inversion_rules.append((rule, error))

            possible_nodes = ['A', 'B', 'C']

            for i, _ in enumerate(incoming_node_indices):
                if f'not {possible_nodes[i]}' in logic and inversion_rules[
                    incoming_node_indices[i]] == False: not_in_inversion_rules()
                if f'{possible_nodes[i]}' in logic and f'not {possible_nodes[i]}' not in logic and inversion_rules[
                    incoming_node_indices[i]] == True: not_in_inversion_rules()

            if (rule, error) not in does_not_follow_inversion_rules:
                follows_inversion_rules.append((rule, error))

        # Finds the average error for the rules that follow the PKN inversion rules
        try:
            average_error = sum([i[1] + 1e-5 for i in follows_inversion_rules]) / len(follows_inversion_rules)
        except ZeroDivisionError:
            logging.warning(f'\n\n\t\t\tERROR: No rules follow the inversion rules for node {node.name}')

        # If the average error is less than 20% for the rules that follow the inversion rules, use those
        if average_error < 0.20:
            rules = [i[0] for i in follows_inversion_rules]
            prediction_errors = [i[1] for i in follows_inversion_rules]

        return rules, prediction_errors

    @staticmethod
    def find_min_error_indices(prediction_errors):
        """
        Finds the indices of the predicted rules with the minimum error

        return min_error_indices, min_error
        """
        min_error = min(prediction_errors)
        min_error_indices = [index for index, value in enumerate(prediction_errors) if
                             value == min_error]

        return min_error_indices, min_error

    @staticmethod
    def maximize_incoming_connections(min_error_indices, rules, prediction_errors):
        """
        For each of the rules with a minimum error, finds the rules with the greatest incoming node connections

        return max_incoming_node_rules, best_rule_indices, best_rule_errors
        """
        max_incoming_node_rules = []
        best_rule_errors = []
        best_rule_indices = []
        num_incoming_nodes_list = []

        # Create a list of the number of incoming nodes for each of the minimum rules
        for index in min_error_indices:
            num_incoming_nodes = 0
            if 'A' in rules[index][2]:
                num_incoming_nodes += 1
            if 'B' in rules[index][2]:
                num_incoming_nodes += 1
            if 'C' in rules[index][2]:
                num_incoming_nodes += 1
            num_incoming_nodes_list.append(num_incoming_nodes)

        # Find the maximum number of incoming nodes
        max_incoming_nodes = max(num_incoming_nodes_list)

        # Compare to see if the current node has the same number of nodes as the max
        for i, index in enumerate(min_error_indices):

            num_incoming_nodes = num_incoming_nodes_list[i]

            def append_best_rule(index):
                max_incoming_node_rules.append(rules[index])
                best_rule_indices.append(index)
                best_rule_errors.append(prediction_errors[index])

            # Find the rules with the same number of rules as the max
            if max_incoming_nodes == 1 and num_incoming_nodes == 1:
                append_best_rule(index)

            elif max_incoming_nodes == 2 and num_incoming_nodes == 2:
                append_best_rule(index)

            elif max_incoming_nodes == 3 and num_incoming_nodes == 3:
                append_best_rule(index)

        return max_incoming_node_rules, best_rule_indices, best_rule_errors

    @staticmethod
    def minimize_incoming_connections(min_error_indices, rules, prediction_errors):
        """
        For each of the rules with a minimum error, finds the rules with the greatest incoming node connections

        return max_incoming_node_rules, best_rule_indices, best_rule_errors
        """
        min_incoming_node_rules = []
        best_rule_errors = []
        best_rule_indices = []
        num_incoming_nodes_list = []

        # Create a list of the number of incoming nodes for each of the minimum rules
        for index in min_error_indices:
            num_incoming_nodes = 0
            if 'A' in rules[index][2]:
                num_incoming_nodes += 1
            if 'B' in rules[index][2]:
                num_incoming_nodes += 1
            if 'C' in rules[index][2]:
                num_incoming_nodes += 1
            num_incoming_nodes_list.append(num_incoming_nodes)

        # Find the maximum number of incoming nodes
        min_incoming_nodes = min(num_incoming_nodes_list)

        # Compare to see if the current node has the same number of nodes as the max
        for i, index in enumerate(min_error_indices):

            num_incoming_nodes = num_incoming_nodes_list[i]

            def append_best_rule(index):
                min_incoming_node_rules.append(rules[index])
                best_rule_indices.append(index)
                best_rule_errors.append(prediction_errors[index])

            # Find the rules with the same number of rules as the max
            if min_incoming_nodes == 1 and num_incoming_nodes == 1:
                append_best_rule(index)

            elif min_incoming_nodes == 2 and num_incoming_nodes == 2:
                append_best_rule(index)

            elif min_incoming_nodes == 3 and num_incoming_nodes == 3:
                append_best_rule(index)

        return min_incoming_node_rules, best_rule_indices, best_rule_errors

    def calculate_refined_errors(self, node, chunked_dataset):
        """
        1) Finds all possible rule combinations for the node
        2) Calculates the error for each rule based on chunked_dataset
        3) Finds the rules with the minimum error
        4) Finds the simplest rules with the minimum error (fewest incoming nodes)
        5) Finds the simplest rules with the greatest number of 'or' connections

        return best_rules
        """

        # Calculate the error for each prediction based on the entire dataset (not chunked)
        prediction_errors = []
        best_rules = []

        # Calculates the error for each possible rule for the node
        rules = node.node_rules

        # Generate all 'not' combinations for each rule
        not_combinations = [self.generate_not_combinations(rule[2]) for rule in rules]

        
        for combination in not_combinations:
            for rule in combination:
                # Find the incoming nodes for the current rule
                incoming_node_indices = [predecessor_index for predecessor_index in node.predecessors]
                # Format the rule
                formatted_rule = [node.name, incoming_node_indices, rule]
                if formatted_rule not in rules:
                    rules.append([node.name, incoming_node_indices, rule])

        logging.debug(f'{node.name}')
        for rule in rules:
            # Calculate the error of each rule and append it to a list
            difference, count = self.calculate_error(node, rule[2], chunked_dataset)
            prediction_error = difference / count
            prediction_errors.append(prediction_error)
            logging.debug(f'\terror: {round(prediction_error,2)}\t{rule[2]}')
        logging.debug(f'\tmin_error: {round(min(prediction_errors),2)}')

        # Find all rules that have the minimum error
        if len(prediction_errors) > 0:  # Excludes nodes with no predictions

            # Prioritizes rules that follow the inversion rules for the gene
            rules, prediction_errors = self.prioritize_pkn_inversion_rules(node, rules, prediction_errors)

            # Finds the rule with the minimum error
            min_error_indices, min_error = self.find_min_error_indices(prediction_errors)

            # Now that we have found the rules with the minimum error, we want to maximize the number of connections
            max_incoming_node_rules, best_rule_indices, best_rule_errors = self.maximize_incoming_connections(
                min_error_indices, rules, prediction_errors)

            # Find the rules with the fewest 'and' statements
            min_rules = min([rule.count("and") for rule in max_incoming_node_rules])
            for i, rule in enumerate(max_incoming_node_rules):
                if rule.count("and") == min_rules:
                    best_rules.append((max_incoming_node_rules[i], best_rule_indices[i], best_rule_errors[i]))

        logging.debug(f'\tbest_rule: {round(best_rules[0][2],2)}')

        return best_rules

    @staticmethod
    def calculate_error(node, predicted_rule, dataset):
        """
        Calculates the error for the node across the dataset using the predicted rule.

        Requires the dataset to be network-specific, dont use the whole dataset. `node.index` is based
        on the dataset that was sliced to only contain gene rows for this network (made during rule_inference: _extract_data())

        1) Maps `node` and its incoming nodes to their rows in the network dataset
        2) Uses a vectorized eval() function with `predicted_rule` to predict what the `node`'s expression should be
            based on the expression of the incoming nodes for each cell (column) in `dataset`
        3) Calculates the number of cells where the expected expression of the node based on the result from (2) and the actual
            expression of `node` are different

        returns the number of differences between the expected and actual expression for `node` and the total number
        of cells

        return difference, count
        """

        # logging.info(f'CustomDeap calculate_error:')
        # Get the row in the dataset for the node being evaluated
        node_evaluated = dataset[node.index]
        # logging.info(f'\tNode {node.name}, Dataset index {node.index}')

        # Get the dataset row indices for the incoming nodes included in this rule
        # logging.info(f'\tPredicted rule: {predicted_rule}')
        incoming_node_indices = [predecessor_index for predecessor_index in node.predecessors]
        # logging.info(f'\tIncoming node indices: {incoming_node_indices}')

        # Initialize A, B, C to False by default (adjust according to what makes sense in context)
        A, B, C = (False,) * 3

        data = {}

        # Map dataset values to A, B, C based on their indices
        if len(incoming_node_indices) > 0:
            data['A'] = dataset[incoming_node_indices[0]]
        if len(incoming_node_indices) > 1:
            data['B'] = dataset[incoming_node_indices[1]]
        if len(incoming_node_indices) > 2:
            data['C'] = dataset[incoming_node_indices[2]]
        if len(incoming_node_indices) > 3:
            data['D'] = dataset[incoming_node_indices[3]]

        def evaluate_expression(var_data, expression):
            # logging.info(f'\tEvaluate_expression function:')
            # logging.info(f'\t\tvar_data = {var_data}')
            # logging.info(f'\t\texpression = {expression}')

            def eval_func(*args):
                local_vars = {name: arg for name, arg in zip(var_data.keys(), args)}
                # logging.info(f'\t\texpression: {expression}, local_vars = {local_vars}')
                return eval(expression, {}, local_vars)

            vectorized_eval = np.vectorize(eval_func)

            # Prepare the argument list in the correct order, corresponding to the keys in var_data
            arg_list = [var_data[key] for key in var_data.keys()]
            return vectorized_eval(*arg_list)

        # logging.info(f'\tdata = {data}')
        predicted = evaluate_expression(data, predicted_rule)

        # # Evaluate the rule using numpy's vectorize function
        # predicted = np.vectorize(eval(predicted_rule))

        # logging.info(f'Predicted: {predicted}')
        difference = np.sum(predicted != node_evaluated)
        # logging.info(f'\tDifferences: {difference}')
        # logging.info(f'\tError = {difference / len(predicted) * 100}%')

        count = len(predicted)

        return difference, count