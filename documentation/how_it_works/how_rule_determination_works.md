# Rule Inference
1) The network is loaded as a NetworkX object from the processed graphml file.
2) The cells are chunked by randomly shuffling the columns of the dataset and binning the expression data for each gene 
based on whether the majority of the cells within the bin are 1 or 0.

3) A rule refinement method identifies the rule that best fits the expression data.

Here is what the output from a successful rule output should look like:

```
-----CHUNKING DATASET-----
        Original Data Shape: 59 rows, 3621 columns
        Chunked Data Shape: 59 rows, 100 columns
        Coarse Chunked Data Shape: 59 rows, 50 columns

-----RULE REFINEMENT-----
|████████████████████████████████████████| 59/59 [100%] in 2.6s (22.41/s)

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