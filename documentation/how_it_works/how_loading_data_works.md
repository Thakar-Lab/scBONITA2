# Loading the scRNA-seq data:
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