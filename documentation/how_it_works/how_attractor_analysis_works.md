# Attractor Analysis

The attractor analysis portion of scBONITA2 consists of two main steps: **simulating individual cells** and **clustering
the cells** based on the similarity of their signaling trajectories for the network.

## Simulating cell trajectories

First, the network pickle file containing the information about the pathway model is loaded. The network object stores 
the data for each gene in the network, the information about each node in the network, and the name of the network.

The cell trajectories are simulated by randomly choosing `num_simulations` columns (cells) in the dataset. For each 
chosen column, a `vectorized_run_simulation` function similar to the one used for calculating importance scores is used
to simulate the Boolean network model where the starting state of the nodes in the network are set using the
chosen cell's gene expression for the genes in the network. This process is multithreaded to increase the computational
efficiency when simulating a large number of cells. Once a cell's trajectory has been simulated, the resulting trajectory
is stored as a csv file.

## Clustering the cells
The goal of this process is to group cells by how similar their simulation trajectories are, which shows different cell
signaling states. Now that the cells have been simulated and the resulting trajectories recorded, we can compare how
similar each cell's trajectory is to each other cell's trajectory. 

To measure the similarity between two gene trajectories across different cells, we use a process called **dynamic time
warping**, which measures the similarity between two time series sequences that may vary in speed or be offset in time.
For a good introduction to dynamic time warping, I encourage you to read [Dynamic Time Warping Algorithm in Time 
Series](https://www.theaidream.com/post/dynamic-time-warping-dtw-algorithm-in-time-series). When comparing two cells,
the dynamic time warping distance between each gene trajectory is summed. Each cell is compared to each other cell, 
creating a distance matrix that is used to group the cells via **hierarchical clustering** to identify clusters of cells
with similar signaling to one another. 

Once the cells are clustered, we find the average expression of each gene at each time point across all cells in the
cluster. This results in an average trajectory graph for the cluster, which helps to identify what makes each cluster
unique. 


### Chunking
Pairwise distance calculations scale exponentially with the number of cells, so to keep the processing time and memory
requirements reasonable, the cells to be analyzed are first divided into random chunks. The entire clustering process 
is used to group the cells in each chunk into different clusters, and the signaling trajectory of each cluster is used
as a trajectory that represents the overall signaling of each cluster for that chunk of cells. 

These representative trajectories are then put through the clustering process again instead of single cell trajectories
to create overall clusters that incorporate all cells. During the whole process, the cells in each cluster are tracked
and the final average trajectory graph is created using their expression.

### Comparing experimental conditions
At the end of the process, the cells in each cluster are grouped further into their different experimental groups used
in the relative abundance portion of scBONITA2. This shows whether each cluster corresponds to a different 
experimental condition, or if the different clusters correspond simply to different possible cell states that are not
altered due to disease. If cells from a certain condition are highly enriched in a single cluster, the average 
trajectory graphs can show how the conditions alter cell signaling.




