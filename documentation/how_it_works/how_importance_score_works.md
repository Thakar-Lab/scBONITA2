# Importance Score Calculations
1) Once the ruleset is inferred, the flow of a signal through the network is simulated. We sample the cells in the dataset and set the state of the nodes in the network to the expression of the cell (if the cell state for a gene is 1, we set the initial state of the node to 1). We then apply the rules synchronously across the network and keep track of the state of each node in the network. The network is simulated until a cycle of states called an attractor is found. 

2) Each of the nodes is iteratively knocked out (expression is forced to be 0 throughout the simulation) or knocked in (expression is forced to be 1 throughout the simulation). The network is simulated again for the same set of cell states and the attractors recorded. 

3) Once the knock out and knock in simulations are conducted for each node in the network, the importance of each node is scored by recording the number of nodes that are altered in the attractor cycle when the nodes are knocked out and knocked in compared to normal. 
    - The greater the number of differences, the more important the node is. 
    - If a node changes the signaling pattern of the network greatly when its expression is altered, then under- or over-expression of that gene will have a greater impact on the pathway compared to a node that does not change the signaling pattern greatly.

Here is what the output from a successful importance score calculation step should look like:

```
 --------------------------------------------------
|     RUNNING IMPORTANCE SCORES FOR GEORGE_HIV     |
 --------------------------------------------------

Loading: george_hiv_hsa04370.ruleset.pickle
Calculating importance score for network hsa04370

-----RUNNING NETWORK SIMULATION-----
|████████████████████████████████████████| 59/59 [100%] in 22.8s (2.59/s)

-----CALCULATING IMPORTANCE SCORES-----
|████████████████████████████████████████| 59/59 [100%] in 0.9s (68.14/s)
Saving importance scores to file: 04370_george_hiv_importance_scores.txt
Saving importance score figures
Saving network object as a pickle file
```