
# How scBONITA2 works:
### 1. Running the BASH script
scBONITA runs using a [bash script](setup_instructions/bash_script_instructions.md) as a wrapper to pass user arguments
into the Python script.

### 2. Processing the KEGG networks
The [networks are downloaded and processed](how_it_works/how_network_processing_works.md) to create graphml files
containing the genes that are also in the scRNA-seq dataset, their connections to other genes, and whether that
connection is activating or inhibiting.

### 3. Loading the dataset
The [scRNA-seq data is loaded and processed](how_it_works/how_loading_data_works.md) to binarize the data and store gene
and cell names.

### 4. Building the Boolean network model
Now that the networks and data are processed, a Boolean model of the network is built using
[scBONITA2 Rule Determination](how_it_works/how_rule_determination_works.md). This generates a Boolean ruleset that
adds logic to the connections between genes. 

### 5. Calculating the importance score of each gene
After a Boolean model is created for the network, the importance of each gene to the signaling pattern of the pathway
is determined using [scBONITA2 Importance Score Calculation](how_it_works/how_importance_score_works.md). The importance
score calculation simulates knocking-in and knocking-out each gene to determine how much the signaling pattern changes
when each gene is perturbed. The more the signaling pattern changes, the higher the importance score of the gene.

### 6. Determining the relative gene expression between conditions
Once we have the importance of each gene in the network, we have more information to determine how much a disease alters
the signaling pathway for a network. [scBONITA2 Relative Abundance](how_it_works/how_relative_abundance_works.md) 
calculates an overall pathway modulation score between conditions by scaling the relative gene expression of each gene
between groups by its importance score.

### 7. Grouping cells by common attractors
Individual cells are simulated and clustered by signaling similarity to identify signaling differences between
experimental conditions.