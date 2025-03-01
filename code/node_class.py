from itertools import combinations, product
import numpy as np
import random

class Node:
    def __init__(self,
                 name,
                 index,
                 predecessors,
                 inversions):

        self.name = name # Gene name
        self.index = index # Index in the network
        self.dataset_index = None # Row index in the dataset

        # Inversion rules for each of the incoming node combinations
        self.inversions = inversions

        # Upstream nodes signaling to this node (dictionary where keys = node index, value = node name)
        # If there are no upstream nodes, set the predecessor to itself
        if len(predecessors) == 0:
            self.predecessors = {self.index : self.name}
        else:
            self.predecessors = predecessors

        # Finds the possible rules based on the number of predecessors and chooses one using one-hot encoding
        self.possibilities = self.enumerate_possibilities()
        self.node_rules = None

        # Finds the best rule
        self.best_rule_index = None
        self.best_rule = None
        self.best_rule_error = None

        # The calculation function for the nodes best_rule
        self.calculation_function = None

        # Importance scores
        self.importance_score = 0
        self.importance_score_stdev = 0.0

        # Relative abundance
        self.relative_abundance = None
    
    def enumerate_possibilities(self):
        # This creates all possible AND and OR possibilities between all input strings
        # See: https://stackoverflow.com/questions/76611116/explore-all-possible-boolean-combinations-of-variables
        cache = dict()
        or0,or1,and0,and1 = "  or  "," or ","  and  "," and "  
        
        def cacheResult(keys,result=None):
            if not result:
                return [ r.format(*keys) for r in cache.get(len(keys),[]) ]   
            cache[len(keys)] = resFormat = []
            result = sorted(result,key=lambda x:x.replace("  "," "))
            for r in result:
                r = r.replace("and","&").replace("or","|")
                for i,k in enumerate(keys):
                    r = r.replace(k,"{"+str(i)+"}")
                r = r.replace("&","and").replace("|","or")
                resFormat.append(r)
            return result

        def boolCombo(keys):
            if len(keys)==1: 
                return list(keys)
            
            result = cacheResult(keys) or set()
            if result: 
                return result
            
            def addResult(left,right):
                OR = or0.join(sorted(left.split(or0)+right.split(or0)))
                result.add(OR.replace(and0,and1))
                if or0 in left:  
                    left  = f"({left})"
                if or0 in right: 
                    right = f"({right})"
                AND = and0.join(sorted(left.split(and0)+right.split(and0)))
                result.add(AND.replace(or0,or1))
                    
            seenLeft  = set()
            for leftSize in range(1,len(keys)//2+1):
                for leftKeys in combinations(keys,leftSize):
                    rightKeys = [k for k in keys if k not in leftKeys]
                    if len(leftKeys)==len(rightKeys):
                        if tuple(rightKeys) in seenLeft: continue
                        seenLeft.add(tuple(leftKeys))
                    for left,right in product(*map(boolCombo,(leftKeys,rightKeys))):
                        addResult(left,right)
            return cacheResult(keys,result)
        
        num_incoming_nodes = len(self.predecessors)
        
        # Find the possible combinations based on the number of incoming nodes
        if num_incoming_nodes == 4:
            possibilities = np.array(boolCombo("ABCD") + boolCombo("ABC") + boolCombo("AB") + boolCombo("AC") + boolCombo("BC") + ["A"] + ["B"] + ["C"])
        if num_incoming_nodes == 3:
            possibilities = np.array(boolCombo("ABC") + boolCombo("AB") + boolCombo("AC") + boolCombo("BC") + ["A"] + ["B"] + ["C"])
        elif num_incoming_nodes == 2:
            possibilities = np.array(boolCombo("AB") + ["A"] + ["B"])
        elif num_incoming_nodes == 1 or num_incoming_nodes == 0:
            possibilities = np.array(["A"])
        else:
            assert IndexError('Num incoming nodes out of range')

        return possibilities
    
    def find_all_rule_predictions(self):
        rule_predictions = []
        
        rules = [rule.replace("  ", " ") for rule in self.possibilities]

        # Add in the inversion rules
        for rule in rules:
            for i, invert_node in enumerate(self.inversions.values()):
                if invert_node == True:
                    if i == 0:
                        rule = rule.replace('A', 'not A')
                    if i == 1:
                        rule = rule.replace('B', 'not B')
                    elif i == 2:
                        rule = rule.replace('C', 'not C')
            rule_predictions.append(rule)
        
        # Update the predicted rules for this node
        node_rules = []
        for rule in rule_predictions:
            node_rules.append([self.name, [i for i in self.predecessors], rule])

        self.node_rules = node_rules
        
        return node_rules