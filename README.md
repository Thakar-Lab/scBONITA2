# scBONITA2
Infers Boolean molecular signaling networks using scRNAseq data and prior knowledge networks, performs attractor analysis, and calculates the importance of each node in the network

## Setup:

> NOTE: scBONITA only runs on Linux environments, if you use Windows please download and install Windows Subsystem for Linux (WSL) [here](https://learn.microsoft.com/en-us/windows/wsl/install)

**Cloning repository**
Go the the directory where you want to install this project and enter `git clone https://github.com/Thakar-Lab/scBONITA2.git` and enter your username and password.

**Setting up your conda environment**
1) Install Anaconda from https://www.anaconda.com/download
2) Run the Anaconda installer in the terminal on Linux or WSL (Windows Subsystem for Linux)
   - `bash Anaconda3-20204.09-0-Linux-x86_64.sh` (if the file you downloaded is different, use that file name)
   - Follow the prompts to install
3) Once Anaconda is installed, close and re-open your terminal. You should see `(base)` before your username
4) To create the correct conda environment, navigate to the `scBONITA2` directory that you cloned from before in the terminal. Once at the correct directory, enter `conda env create --file spec-file.txt --name scBonita`
5) Once conda has finished working, you can confirm that the environment was created by entering `conda activate scBonita`. This will switch you into the correct environment to work with scBONITA.
   - A conda environment is basically a pre-packaged python installation that has a specific python version and package list that works with the code. This makes it so that you don't have to install each required package one-by-one, and you can have different package versions by having different conda environments

**Testing that scBONITA is working**
1) Ensure that the `scBonita` conda environment is active (or enter `conda activate scBonita` to activate)
2) Navigate to `scBONITA2/scBONITA` in your terminal
3) Run the test data for the scBONITA pipeline by entering `bash bash_scripts/local_george_hiv.sh`
   - The `local_george_hiv.sh` file can be copied and modified to run different datasets / conditions
   - When running scBONITA, place bash scripts into this folder to run. scBONITA will automatically create the necessary output files in a directory called `scBONITA_output`.
   - Make sure that you have the `george_data` directory downloaded 
   - If a package is missing, download it using the command `conda install <PACKAGE_NAME>` or `conda install -c conda-forge <PACKAGE_NAME>`

## Running scBONITA
**BASH Script** 

scBONITA runs using a BASH script that allows you to specify all of your parameters in one place. These BASH scripts are found in the `scBONITA2/scBONITA/bash_scripts` directory. To run using your own data, simply copy the bash file and modify the values.  

Each of the steps for running scBONITA is modular, meaning that you can run the rule determination step first. This means that you dont have to run it all at the same time (although you do need to run it in order).

You have to run Rule Determination $\rightarrow$ Importance Score $\rightarrow$ Relative Abundance $\rightarrow$ Attractor Analysis, but not at the same time. I recommend running Rule Determination first to make sure your environment and paths are correct.

> IMPORTANT: Make sure that the CONDA_ENVIRONMENT_PYTHON path is correct. You can find the path using `which python` on macOS and Linux


```bash
# IMPORTANT!!! MAKE SURE THAT THERE ARE NO SPACES IN FILE NAMES

# Which parts do you want to run? Set True to run or False to skip
    # Rule determination must be run prior to importance score, importance score must be run prior to relative abundance
RUN_RULE_DETERMINATION=True
RUN_IMPORTANCE_SCORE=False
RUN_RELATIVE_ABUNDANCE=False
RUN_ATTRACTOR_ANALYSIS=False

# General Arguments (Required for all steps)
DATA_FILE="../../george_data/hiv_dataset/HIV_dataset_normalized_integrated_counts.csv"
CONDA_ENVIRONMENT_PYTHON="/home/emoeller/anaconda3/envs/scBonita_test2/bin/python" # Path to the installation of Python for the scBonita conda environment
DATASET_NAME="george_hiv"
DATAFILE_SEP=","
KEGG_PATHWAYS=("04370") # Enter KEGG pathway codes or leave blank to find all pathways with overlapping genes. Separate like: ("hsa04670" "hsa05171")
CUSTOM_PATHWAYS=() #("modified_network.graphml") #Put custom networks in the input folder
BINARIZE_THRESHOLD=0.01 # Data points with values above this number will be set to 1, lower set to 0
MINIMUM_OVERLAP=1 # Specifies how many genes you want to ensure overlap with the genes in the KEGG pathways. Default is 25
ORGANISM_CODE="hsa" # Organism code in front of KEGG pathway numbers

# Relative Abundance arguments
METADATA_FILE="../../george_data/hiv_dataset/hiv_meta.txt"
METADATA_SEP=" "
HEADER="n" # Does the metadata file contain a header before the entries start?
OVERWRITE="n" # Do you want to overwrite the files generated for each of your different experimental groups?
CELL_NAME_COL=1 # What column contains the cell names (first column = 0)
GROUP_INDICES=(2)

# Specify the control groups and experimental groups that you want to compare
    # 1st entry in control is compared to 1st entry in experimental, 2nd entry compared to 2nd entry, etc.

CONTROL_GROUPS=("Healthy")
EXPERIMENTAL_GROUPS=("HIV")
```

For the relative abundance section, scBONITA expects a metadata file with the following format:

```
"0" "C1_Healthy_AAACGGGAGTCGTTTG.1" "Healthy"
"1" "C1_Healthy_AAACGGGTCAGGCGAA.1" "Healthy"
"2" "C1_Healthy_AAAGCAATCTCAACTT.1" "Healthy"
"3" "C1_Healthy_AACACGTAGAGCAATT.1" "Healthy"
"4" "C1_Healthy_AACCGCGCAATCCGAT.1" "Healthy"
"5" "C1_Healthy_AACCGCGTCCTATTCA.1" "Healthy"
```

You can specify the separator, cell name column, which columns have the name of the groups (the first column is 0), and whether or not there is a header on the first line. Specify the control groups and the experimental groups to compare for the relative abundance calculations.

## Understanding the Process:
### Network Processing:
1) scBONITA starts by finding either all KEGG pathways (if the `KEGG_PATHWAYS` variable in the bash file is blank) or the KEGG pathways specified in the `KEGG_PATHWAYS` variable in the bash file. It will download all pathway xml files into the `pathway_xml_files` directory specific to the organism to make parsing the pathways faster than streaming the data directly from the KEGG API.
2) Next, scBONITA will check that the dataset you provided has enough overlapping genes with the pathways specified by the user, or all KEGG pathways (default is >25 shared genes, you can modify this value in the BASH file if you need to).
3) Once the pathways are identified, the pathways are processed to create "_processed.graphml" files.

The output should look similar to this:

```
-----PARSING NETWORKS-----
        KEGG pathways = ['04370']
        Finding and formatting KEGG Pathways...
                Finding KEGG pathways...
                Parsing KEGG dict...
                        Reading in KEGG dictionary file...
                        Loaded KEGG code dictionary
                        Reading hsa dictionary file...
                Downloading any missing pathway xml files, this may take a while...
|████████████████████████████████████████| 359/359 [100%] in 0.2s (1892.88/s)
                Finding pathways with at least 1 genes that overlap with the dataset
|               Reading KEGG xml file    | ▁▃▅ 0/1 [0%] in 0s (0.0/s, eta: ?)
                        Pathway (0/1): 04370 Overlap: 30 Edges: 65
                Reading KEGG xml file
                        Pathway (0/1): 04370 Overlap: 59 Edges: 166
|████████████████████████████████████████| 1/1 [100%] in 0.1s (9.92/s)
                Adding graphml pathways to rule_inference object...
                Pathway: 04370 Overlap: 59 Edges: 166
                        Edges after processing: 158 Overlap: 59
```

### Loading the Data:
1) scBONITA first extracts the data, randomly sampling the cells if there are more than 15000 samples to limit the size of the dataset.
2) The matrix is then converted to a sparse matrix to condense the information, and the information is binarized to 1 or 0 based on the value of the `BINARIZE_THRESHOLD` variable in the bash file. Expression values above this threshold are set to 1 and values below this threshold are set to 0, indicating that the gene is either ON or OFF
3) The genes in the dataset are filtered to include variable genes with a coefficient of variation above the cv_threshold (default 0.001, this can be set as a pipeline.py argument in the bash file).
4) Once the dataset is loaded and configured, the processed network and data are used for determining the Boolean rules linking genes together.

Here is what a successful run looks like:

```
-----RULE INFERENCE-----
Pathway: 04370
Num nodes: 59

-----EXTRACTING AND FORMATTING DATA-----
Extracting cell expression data from "../../george_data/hiv_dataset/HIV_dataset_normalized_integrated_counts.csv"
        Loading all 3621 cells...
        Converting filtered data to numpy array...
        First 2 genes: ['PIK3CD', 'CASP9']
        First 2 cells: ['C1_Healthy_AAACGGGAGTCGTTTG.1', 'C1_Healthy_AAACGGGTCAGGCGAA.1']
        Number of genes: 59
        Number of cells: 3621
        Created sparse matrix
        Binarized sparse matrix
        Setting ruleset parameters
        Running rule inference for 04370
                Loading: hsa04370_processed.graphml
/home/emoeller/anaconda3/envs/scBonita/lib/python3.6/site-packages/scipy/stats/stats.py:4196: SpearmanRConstantInputWarning: An input array is constant; the correlation coefficent is not defined.
  warnings.warn(SpearmanRConstantInputWarning())
|████████████████████████████████████████| 59/59 [100%] in 0.0s (1838.83/s)
```

### Rule Inference
1) The network is loaded as a NetworkX object from the processed graphml file.
2) The cells are clustered by randomly shuffling the columns of the dataset and binning the expression data for each gene based on whether the majority of the cells within the bin are 1 or 0.
3) The genetic algorithm runs to find a set of rules that has a low error. Each of the nodes in the network has other nodes signaling to it. These incoming nodes can be connected by Boolean rules, for example if node 1 and node 2 signal to node 3, the possibilities are that node 1 AND node 2 activate node 3 or that node 1 OR node 2 activates node 3.

5) We repeat this for each cell cluster in the dataset, adding up the total error as the fitness of the individual. The higher the error, the lower the fitness. The population for the genetic algorithm is made of many individual random rulesets, and over time the individuals with the lowest error are selected.

6) Once the best individuals from the genetic algorithm are selected, a rule refinement method is performed on for each node in the best individuals, where the other rule possibilities are considered for nodes with a high error. This is done to optimize rules with high error that were not optimized during the genetic algorithm. The rulesets are formatted and output to the rules_output directory for the project.

Here is what the output from a successful rule output should look like:

```
-----CHUNKING DATASET-----
        Original Data Shape: 59 rows, 3621 columns
        Chunked Data Shape: 59 rows, 100 columns
        Coarse Chunked Data Shape: 59 rows, 50 columns

-----GENETIC ALGORITHM-----
ngen    nevals  avg     std     min     max
0       20      0.233   0.011   0.213   0.256
1       20      0.234   0.009   0.218   0.252
2       20      0.232   0.012   0.218   0.252
3       20      0.231   0.011   0.221   0.252
4       20      0.233   0.013   0.211   0.252

-----RULE REFINEMENT-----
Equivalent Ruleset 1 / 1
|████████████████████████████████████████| 59/59 [100%] in 2.6s (22.41/s)

Equivalent Ruleset 1
CDC42 = KDR
KDR = VEGFA
PTGS2 = not NFATC2
HSPB1 = MAPKAPK2 and MAPKAPK3
NFATC2 = (PPP3CA or PPP3CC) and PPP3CB
PTK2 = KDR
PXN = KDR
SH2D2A = (PLCG1 or KDR) and PLCG2
SRC = KDR
SHC2 = KDR
VEGFA = VEGFA
RAF1 = NRAS or HRAS and PRKCA
NOS3 = (PLCG2 or AKT1) and PLCG1
CASP9 = (AKT1 or AKT3) and AKT2
.
.
.
MAP2K1 = RAF1
MAP2K2 = RAF1
HRAS = SPHK2 and SPHK1
KRAS = SPHK1 and SPHK2
NRAS = SPHK2 and SPHK1
RAC1 = PIK3R2 or PIK3CB and PIK3R1
RAC2 = (PIK3R2 or PIK3CD) and PIK3R1
RAC3 = (PIK3R2 or PIK3CD) and PIK3R1
Refined Error:

        Average = 0.08474576271186439
        Stdev = 0.14517950758075673
        Max = 0.57
        Min = 0.0
        Percent_self_loops = 2.0%

Rule inference complete, saving to george_hiv_hsa04370.ruleset.pickle
``` 

### Importance Score Calculations
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

### Relative Abundance

**User input in the BASH file**
```bash
# Relative Abundance arguments
METADATA_FILE="../input/george_data/hiv_dataset/hiv_meta.txt"
METADATA_SEP=" "
HEADER="n" # Does the metadata file contain a header before the entries start?
OVERWRITE="n" # Do you want to overwrite the files generated for each of your different experimental groups?
CELL_NAME_COL=1 # What column contains the cell names (first column = 0)
GROUP_INDICES=(2)

# Specify the control groups and experimental groups that you want to compare
    # 1st entry in control is compared to 1st entry in experimental, 2nd entry compared to 2nd entry, etc.
CONTROL_GROUPS=("Healthy")
EXPERIMENTAL_GROUPS=("HIV")
```

**Metadata File**

```
"0" "C1_Healthy_AAACGGGAGTCGTTTG.1" "Healthy"
"1" "C1_Healthy_AAACGGGTCAGGCGAA.1" "Healthy"
"2" "C1_Healthy_AAAGCAATCTCAACTT.1" "Healthy"
"3" "C1_Healthy_AACACGTAGAGCAATT.1" "Healthy"
"4" "C1_Healthy_AACCGCGCAATCCGAT.1" "Healthy"
```


**Argument Explanation:** 

Here, the dataset is being split based on the metadata file `hiv_meta.txt`. Columns in the file are separated with a space (`" "`), there is no header, and the names of the cells are in column 1 (indexing starts at 0). The group each cell belongs to is found in column 2, and the two groups of interest are "Healthy" and "HIV".

1) First, the columns in the dataset are split into the two groups "Healthy" and "HIV" and saved to separate csv files in the same location as the original data file. If you have already created the split group data files in the past and want to overwrite them, specify `OVERWRITE="y"` in the BASH file.

```
 ----------------------------------------------------------
|     RELATIVE ABUNDANCE FOR GEORGE_HIV HEALTHY VS HIV     |
 ----------------------------------------------------------

----- Splitting Data Files -----
        Group 1: Healthy
        Group 2: HIV

----- Saving Group Data Files -----
                Writing data to the group file...
                Writing data to the group file...
                Using existing group network hsa04370 file for Healthy
                Using existing group network hsa04370 file for HIV
                Control Group: Healthy
                Experimental Group: HIV
```

2) Next, the networks of interest are loaded for both groups

```
----- Loading george_hiv networks -----
        Loading CONTROL group networks
                Loaded network hsa04370_Healthy

        Loading EXPERIMENTAL group networks
                Loaded network hsa04370_HIV
```

3) Then, the relative abundance of each group is calculated along with a pathway modulation score. The pathway modulation score weighs the importance score and relative abundance of each node to determine if the pathway is significantly altered. It uses bootstrapping to determine if the pathway modulation score is significantly different compared to if the importance scores and relative abundances were randomly distributed among the genes.

> The distribution generated by bootstrapping can be viewed in the relative abundnace output folder

```
----- Calculating Relative Abundance between groups HIV and Healthy -----
        Network: hsa04370
                Pathway Modulation Score: 0.06087929272259514
                Calculating p-value with bootstrapping:
                        P-value: 0.5676
                        -log10(P-value): 0.2459576131380494
```

### Attractor Analysis

> NOTE: This is a current work in progress, I am switching to a different method to map cells to attractors.

This process attempts to find attractors that are similar to one another and create a representative attractor that best represents the different signaling states of each cell. Each cell is compared to each attractor to find the attractor that best represents that cell. Attractors that do not match most closely to at least one cell are removed. Hamming distance is then used to compare each attractor to generate a distance matrix. The attractors are clustered using Hierarchical clustering. Each cell is then assigned to a cluster based on which cluster contains the attractor it is most similar to. The attractor that best matches the greatest number of cells is used as the representative attractor for that cluster.

Here is what a successful run should look like:

```
----- ATTRACTOR ANALYSIS -----

Network: hsa04370
        Generating attractors...
        Calculating hamming distance between cells and attractors
        Transposed dataset shape: (59, 500)
                Extra dimension in dataset, squeezing...
                Transposed dataset shape: (500, 59)
                Nodes: 59
                Cells: 500
                Attractors: 500
        Generating attractor distance matrix...
        Clustering the attractors...
                Clustering cutoff value = 14.75 (<=20% * number of genes 59)
        Calculating Hamming distance between cells and clustered attractors
Number of cells in the full dataset: 3621

-----PATHWAY ANALYSIS RESULTS -----

NETWORK HSA04370
        Attractor 1 contains 3619 cells (99.945%)
        Attractor 2 contains 2 cells (0.055%)
        Saved representative attractor 1
        Saved representative attractor 2

        Saved attractor analysis results to "attractor_analysis_output/george_hiv_attractors/hsa04370_attractors

Adding representative attractor map to network pickle files:
        File: george_hiv_hsa04370.network.pickle
```






