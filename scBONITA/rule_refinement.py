import logging
from itertools import product
import numpy as np

def find_min_error_individuals(population, fitnesses):
    """
    Finds the individuals with the minimum error from the population
    
    return unique_min_error_individuals
    """
    # logging.info(f'Finding the minimum error individuals')
    min_error_individuals = [ind for ind in population if ind.fitness.values == min(fitnesses)]

    # Set to keep track of unique value combinations
    seen_values = set()

    # List comprehension to filter out duplicates
    unique_min_error_individuals = []
    for ind in min_error_individuals:
        # Convert the individual's values to a tuple (or another immutable type if necessary)
        values_tuple = tuple(ind[1])

        # Check if we've already seen these values
        if values_tuple not in seen_values:
            unique_min_error_individuals.append(ind)
            seen_values.add(values_tuple)
    
    return unique_min_error_individuals

def get_third_element(item):
    """
    Returns the third item in a list, gets around the restriction of 
    using lambda when pickling objects

    return item[2]
    """
    return item[2]

def handle_self_loop(node):
    """
    Handles instances when a node only connects to itself or has
    no incoming nodes

    return best_rule
    """
    # If the node has only an incoming connection from itself, skip the rule refinement
    # Add in self signaling for the node
    # logging.info(f'\tNode {node.name} signals only to itself')
    # logging.info(f'\t[{node.name}, [{node.index}]')
    node.node_rules = [node.name, [node.index], 'A']
    node.predecessors[node.index] = node.name
    node.inversions[node.index] = False

    # Set the node's incoming node rules to itself
    best_rule = ([node.name, [node.index], 'A'], 0, 0)

    return best_rule

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
    # logging.info(f'\t\t\tNode combinations: {combinations}')
    
    all_not_combinations = []
    for combo in combinations:
        # Create a temporary rule with the current combination
        temp_rule = rule
        for var, replacement in zip(variables, combo):
            # Replace variables with the current combination of variable or its negation
            temp_rule = temp_rule.replace(var, replacement)
        all_not_combinations.append(temp_rule)
    
    return all_not_combinations

def prioritize_pkn_inversion_rules(node, rules, prediction_errors):
    """
    Prioritizes the rules that follow the inversion rules from the PKN (found in node.inversions). However, if the 
    average error for all of the rules in the PKN is too high, then all possible rules are considered

    returns rules, prediction_errors
    """

    # Prioritize the rules that follow the node's inversion rules
    # logging.info(f'\n\t\tHANDLING INVERSION RULES')
    # logging.info(f'\t\t\tNode {node.name} inversion rules:')
    # for key, value in node.inversions.items():
    #     logging.info(f'\t\t\t\t{node.predecessors[key]}: {value}')
    follows_inversion_rules = []
    does_not_follow_inversion_rules = []

    # Determines if the possibility follows the inversion rules from the PKN network
    for rule, error in zip(rules, prediction_errors):
        incoming_node_indices = rule[1]
        logic = rule[2]
        inversion_rules = node.inversions
        # logging.info(f'\t\t\tRule = {logic}; Error = {error}')

        # logging.info(f'\t\t\tIncoming node indices: {incoming_node_indices}')
        # logging.info(f'\t\t\tInversion Rules: {inversion_rules}')

        def not_in_inversion_rules():
            if (rule, error) not in does_not_follow_inversion_rules:
                # logging.info(f'\t\t\t\t\t{[i for i in inversion_rules.values()]} Logic does not follow inversion')
                does_not_follow_inversion_rules.append((rule, error))

        possible_nodes = ['A', 'B', 'C']

        for i, _ in enumerate(incoming_node_indices):
            if f'not {possible_nodes[i]}' in logic and inversion_rules[incoming_node_indices[i]] == False: not_in_inversion_rules() 
            if f'{possible_nodes[i]}' in logic and f'not {possible_nodes[i]}' not in logic and inversion_rules[incoming_node_indices[i]] == True: not_in_inversion_rules()
        
        if (rule, error) not in does_not_follow_inversion_rules:
            # logging.info(f'\t\t\t\t\t{[i for i in inversion_rules.values()]} Logic rule DOES follow inversion')
            follows_inversion_rules.append((rule, error))
    
    # Finds the average error for the rules that follow the PKN inversion rules
    try:
        average_error = sum([i[1] + 1e-5 for i in follows_inversion_rules]) / len(follows_inversion_rules)
        # logging.info(f'\n\t\t\tAvg error following PKN inversion rules = {average_error}')
    except ZeroDivisionError:
        logging.warning(f'\n\n\t\t\tERROR: No rules follow the inversion rules for node {node.name}')

    # If the average error is less than 85% for the rules that follow the inversion rules, use those
    if average_error < 0.85:
        # logging.info(f'\t\t\t\tError is less than 85% for rules that follow the PKN inversion rules, only considering those {len(follows_inversion_rules)}')
        rules = [i[0] for i in follows_inversion_rules]
        prediction_errors = [i[1] for i in follows_inversion_rules]
    # else:
        # logging.info(f'\t\t\t\tError is greater than 85% for rules that follow the PKN inversion rules, considering all rules {len(follows_inversion_rules) + len(does_not_follow_inversion_rules)}')
    
    return rules, prediction_errors

def find_min_error_indices(prediction_errors):
    """
    Finds the indices of the predicted rules with the minimum error

    return min_error_indices, min_error
    """
    # logging.info(f'\t\t\tPrediction errors: {prediction_errors}')
    min_error = min(prediction_errors)
    min_error_indices = [index for index, value in enumerate(prediction_errors) if
                                value == min_error]

    return min_error_indices, min_error

def maximize_incoming_connections(min_error_indices, rules, prediction_errors):
    """
    For each of the rules with a minimum error, finds the rules with the greatest incoming node connections

    return max_incoming_node_rules, best_rule_indices, best_rule_errors
    """
    # logging.info(f'\n\t\tMAXIMIZING INCOMING CONNECTIONS')
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
    # logging.info(f'\t\t\tNum incoming nodes list: {num_incoming_nodes_list}')

    # Find the maximum number of incoming nodes
    max_incoming_nodes = max(num_incoming_nodes_list)

    # Compare to see if the current node has the same number of nodes as the max
    for i, index in enumerate(min_error_indices): 

        num_incoming_nodes = num_incoming_nodes_list[i]

        # logging.info(f'\n\t\t\tRule: {rules[index]}')
        # logging.info(f'\t\t\tminimum error index: {index}')
        # logging.info(f'\t\t\t\tmax_incoming_nodes = {max_incoming_nodes}')
        # logging.info(f'\t\t\t\tnum_incoming_nodes = {num_incoming_nodes}')

        def append_best_rule(index):
            # logging.info(f'\t\t\t\tbest_rule: {rules[index]}, Index: {index}, error: {prediction_errors[index]}')
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

def calculate_refined_errors(node, chunked_dataset):
    """
    1) Finds all possible rule combinations for the node
    2) Calculates the error for each rule based on chunked_dataset
    3) Finds the rules with the minimum error
    4) Finds the simplest rules with the minimum error (fewest incoming nodes)
    5) Finds the simplest rules with the greatest number of 'or' connections

    return best_rules
    """

    # logging.info(f'\tCALCULATING REFINED ERROR FOR {node.name}')

    # Calculate the error for each prediction based on the entire dataset (not chunked)
    # logging.info(f'\tCalculate_refined_errors function:')
    prediction_errors = []
    best_rules = []

    # logging.info(f'\t\tPredicted rule errors:')

    # # Calculates the error for each possible rule for the node
    # logging.info(f'\t\t\tNode rules: {node.node_rules}')
    # logging.info(f'\t\tFinding all possible rules for {node.name} based on {len(node.predecessors)} incoming nodes')
    rules = node.find_all_rule_predictions()
    
    # Generate all 'not' combinations for each rule
    not_combinations = [generate_not_combinations(rule[2]) for rule in rules]

    # logging.info(f'\t\t\tGenerated {len(not_combinations)} possible rules')

    # print('NOT COMBINATIONS')
    for combination in not_combinations:
        for rule in combination:
            # print(rule)
            incoming_node_indices = [predecessor_index for predecessor_index in node.predecessors]
            formatted_rule = [node.name, incoming_node_indices, rule]
            if formatted_rule not in rules:
                rules.append([node.name, incoming_node_indices, rule])

    # logging.info(f'\n\n\n RULES')
    # for rule in rules:
    #     print(rule)

    # logging.info(f'\t\tCalculating the error for each rule')
    for rule in rules:
        # logging.info(f'\t\t\tPredicted rule: {rule}')
        difference, count = calculate_error(node, rule[2], chunked_dataset)
        prediction_error = difference / count
        # logging.info(f'\t\t\t\tError: {prediction_error}')
        prediction_errors.append(prediction_error)
    
    # Find all of the rules that have the minimum error
    if len(prediction_errors) > 0: # Excludes nodes with no predictions
        
        # logging.info(f'\t\tPrioritizing using inversion rules from the PKN (unless average rule error > 85%)')
        rules, prediction_errors = prioritize_pkn_inversion_rules(node, rules, prediction_errors)
        # logging.info(f'\t\t\tFound {len(rules)} rule(s)')
        
        # logging.info(f'\t\tFinding the indices for rules with the minimum error')
        min_error_indices, min_error = find_min_error_indices(prediction_errors)
        # logging.info(f'\t\t\tFound {len(min_error_indices)} minimum error rule(s) (min error {min_error})')

        # logging.info(f'\n\t\t\tMinimum error indices: {min_error_indices}, minimum error: {min_error}\n')

        # Now that we have found the rules with the minimum error, we want to maximize the number of connections
        # logging.info(f'\t\tMaximizing the number of connections')
        max_incoming_node_rules, best_rule_indices, best_rule_errors = maximize_incoming_connections(min_error_indices, rules, prediction_errors)
        # logging.info(f'\t\t\tFound {len(max_incoming_node_rules)} rule(s)')

        # Find the rules with the fewest 'and' statements
        # logging.info(f'\t\tFinding the rules with the fewest "and" statements')
        min_rules = min([rule.count("and") for rule in max_incoming_node_rules])
        # logging.info(f'\t\tmax_incoming_node_rules: {max_incoming_node_rules}')
        for i, rule in enumerate(max_incoming_node_rules):
            if rule.count("and") == min_rules:
                best_rules.append((max_incoming_node_rules[i], best_rule_indices[i], best_rule_errors[i]))
        
        # logging.info(f'\n\tBEST RULES FOR {node.name}:')
        # for i in best_rules:
            # logging.info(f'\t\tRule = {i[0][2]} | Error = {i[2]}')
        # logging.info(f'\n')

    return best_rules

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




