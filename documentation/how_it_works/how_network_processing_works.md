# Network Processing:

1) scBONITA starts by finding either all KEGG pathways (if the `KEGG_PATHWAYS` variable in the bash file is blank) or the KEGG pathways specified in the `KEGG_PATHWAYS` variable in the bash file. It will download all pathway xml files into the `pathway_xml_files` directory specific to the organism to make parsing the pathways faster than streaming the data directly from the KEGG API.
2) Next, scBONITA will check that the dataset you provided has enough overlapping genes with the pathways specified by the user, or all KEGG pathways (default is >25 shared genes, you can modify this value in the BASH file if you need to).
3) Once the pathways are identified, the pathways are processed to create "_processed.graphml" files.

### Normal Output:

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