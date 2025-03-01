# Understanding the BASH Script

scBONITA runs using a **BASH script** that allows you to specify all of your parameters in one place. 

- These BASH scripts are Found in the `scBONITA2/bash_scripts` directory. 

To run using your own data, simply copy the bash file and modify the values.  

Each of the steps for running scBONITA is **modular**, meaning that you can run the rule determination step first. This means that you dont have to run it all at the same time (although you do need to run it in order).

## Determining which parts of scBONITA you want to run:

```bash
RUN_RULE_DETERMINATION=True
RUN_IMPORTANCE_SCORE=True
RUN_RELATIVE_ABUNDANCE=True
RUN_ATTRACTOR_ANALYSIS=True
```

1. `RUN_RULE_DETERMINATION` Runs the **rule determination** step of scBONITA2. 
    - This step uses the scRNA-seq data and KEGG pathways to generate Boolean models of the molecular signaling pathways.

2. `RUN_IMPORTANCE_SCORE`: Runs the **importance score** calculation step. 
    - This step simulates knocking-in and knocking-out each gene sequentially and compares how much the signaling changes to assign each gene in the network an importance score. The greater the change in signaling when the gene is knocked out, the higher the importance score.

3. `RUN_RELATIVE_ABUNDANCE`: Runs the **relative abundance** calculation step.
    - This step calculates a pathway modulation score between two conditions (e.g. healthy vs disease) by taking into account the relative gene expression scaled by the importance score of the gene.

4. `RUN_ATTRACTOR_ANALYSIS`: Runs the **attractor analysis** step. 
    - This identifies common signaling patterns and groups each cell in the dataset by which of the common signaling patterns it is most similar to. This helps to identify different signaling states between healthy and diseased cells (still a work in progress).

You can select which of these steps you would like to run. You can run them individually or all at once, but there are **dependencies**. You have to (initially) run the code in this order:
- Rule determination $\rightarrow$ importance score $\rightarrow$ relative abundance $\rightarrow$ attractor analysis

Once you run the required previous step, you dont have to run it again to run a later step. The results are saved as pickle files that can be loaded if needed later.

## Rule determination arguments

```bash
DATA_FILE="input/tutorial_data/tutorial_04370_data.csv"
DATASET_NAME="tutorial_dataset"
DATAFILE_SEP=","
KEGG_PATHWAYS=("04370")
CUSTOM_PATHWAYS=()
BINARIZE_THRESHOLD=0.01
MINIMUM_OVERLAP=1
ORGANISM_CODE="hsa"
```
1. `DATA_FILE`: The path to your scRNA-seq data
    - Should be a matrix with rows as the genes and columns as the cells. The first column should contain the gene names, and the first row should contain the cell barcode. Our datasets are log2 normalized count matrices.

2. `DATASET_NAME`: This can be anything, it's used to delineate different projects in the output file names and for finding the right data.

3. `DATAFILE_SEP`: The separator used in your datafile.

4. `KEGG_PATHWAYS`: A list of KEGG pathway numbers you are interested in. 
    - Specify multiple pathways like so: `("04370" "04680" "04151")`
    - Leave blank to look at all pathways that have a `MINIMUM_OVERLAP` number of overlapping genes with the genes in your dataset.

5. `CUSTOM_PATHWAYS`: A list of any custom .graphml network files you would like to run scBONITA2 on. 
    - Put any custom networks in the `input/custom_graphml_files` directory
    - They can be referenced like so: `CUSTOM_PATHWAYS=("<modified_network>.graphml")`

6. `BINARIZE_THRESHOLD`: Data will be binarized (set to 0 or 1) based on this threshold. The default is 0.1, but look at how your data is distributed to see what the best threshold would be (ideally, look for the point between peaks in a binomial distribution)

7. `MINIMUM_OVERLAP`: This compares how many genes are in both your dataset and in the pathway. A lower overlap will use pathways where there are fewer matching genes.

8. `ORGANISM_CODE`: The three-letter KEGG organism identifier found in front of the pathway codes.
    - Default is `hsa` (human)

## Relative abundance arguments
```bash
METADATA_FILE="input/tutorial_data/tutorial_metadata.txt"
METADATA_SEP=" "
HEADER="n"
OVERWRITE="y"
CELL_NAME_COL=1
GROUP_INDICES=(2)
CONTROL_GROUPS=("Healthy")
EXPERIMENTAL_GROUPS=("HIV")
```

These parameters are used to split the dataset into different groups so they can be compared.

1. `METADATA_FILE`: The path to the metadata for the dataset.

2. `METADATA_SEP`: The separator between columns in the metadata file.

3. `HEADER`: Is there a header at the top of the file? `y` or `n`.

4. `OVERWRITE`: Do you want to overwrite the group data files? 
    - Splitting the dataset into different groups can take a while. If this variable is set to `n`, it will use the pre-existing group datasets. If you need to write over the group datasets, set the variable to `y`.

5. `CELL_NAME_COL`: Which column contains the cell identifiers? (indexing starts at 0)

6. `GROUP_INDICES`: Which column(s) contain the group names?

7. `CONTROL_GROUPS`: What is the identifier for the control groups in the metadata file?

8. `EXPERIMENTAL_GROUPS`: What is the identifier for the experimental group(s) in the metadata file?

**Metadata file example:**

For the relative abundance section, scBONITA expects a metadata file with the following format:

```
"0" "C1_Healthy_AAACGGGAGTCGTTTG.1" "Healthy"
"1" "C1_Healthy_AAACGGGTCAGGCGAA.1" "Healthy"
"2" "C1_Healthy_AAAGCAATCTCAACTT.1" "Healthy"
"3" "C1_Healthy_AACACGTAGAGCAATT.1" "Healthy"
"4" "C1_Healthy_AACCGCGCAATCCGAT.1" "Healthy"
"5" "C1_Healthy_AACCGCGTCCTATTCA.1" "Healthy"
```

Here, `CELL_NAME_COL` would be 1 and `GROUP_INDICES` would be 2.