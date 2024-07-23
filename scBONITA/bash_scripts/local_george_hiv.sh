#!/bin/bash

# -------------- User Input --------------

# IMPORTANT!!! MAKE SURE THAT THERE ARE NO SPACES IN FILE NAMES

# Which parts do you want to run? Set True to run or False to skip
    # Rule determination must be run prior to importance score, importance score must be run prior to relative abundance
RUN_RULE_DETERMINATION=True
RUN_IMPORTANCE_SCORE=True
RUN_RELATIVE_ABUNDANCE=True
RUN_ATTRACTOR_ANALYSIS=True

# General Arguments (Required for all steps)
DATA_FILE="../input/george_data/hiv_dataset/HIV_dataset_normalized_integrated_counts.csv"
CONDA_ENVIRONMENT_PYTHON="/home/emoeller/anaconda3/envs/scBonita/bin/python" # Path to the installation of Python for the scBonita conda environment
DATASET_NAME="george_hiv" # Enter the name of your dataset
DATAFILE_SEP="," # Enter the character that the values in your dataset are split by
KEGG_PATHWAYS=("04370") # Enter KEGG pathway codes or leave blank to find all pathways with overlapping genes. Separate like: ("hsa04670" "hsa05171")
CUSTOM_PATHWAYS=() #("modified_network.graphml") #Put custom networks in the input folder
BINARIZE_THRESHOLD=0.01 # Data points with values above this number will be set to 1, lower set to 0
MINIMUM_OVERLAP=1 # Specifies how many genes you want to ensure overlap with the genes in the KEGG pathways. Default is 25
ORGANISM_CODE="hsa" # Organism code in front of KEGG pathway numbers

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

# -------------- End of user input, shouldn't have to change anything below here --------------

#  ----------------------------
# |     RULE DETERMINATION     |
#  ----------------------------

if [ "$RUN_RULE_DETERMINATION" = "True" ]; then
    echo "Running Rule Determination..."


    if [ ${#KEGG_PATHWAYS[@]} -gt 0 ]; then
        echo "Running with KEGG pathways"

        # Using a list of KEGG pathways:
        KEGG_PATHWAYS_ARGS="${KEGG_PATHWAYS[@]}"

        $CONDA_ENVIRONMENT_PYTHON pipeline_class.py \
            --data_file "$DATA_FILE" \
            --dataset_name "$DATASET_NAME" \
            --datafile_sep "$DATAFILE_SEP" \
            --list_of_kegg_pathways $KEGG_PATHWAYS_ARGS \
            --binarize_threshold $BINARIZE_THRESHOLD \
            --organism $ORGANISM_CODE \
            --minimum_overlap $MINIMUM_OVERLAP
    else
        echo "No KEGG pathways specified, finding kegg pathways with overlapping genes..."
        $CONDA_ENVIRONMENT_PYTHON pipeline_class.py \
        --data_file "$DATA_FILE" \
        --dataset_name "$DATASET_NAME" \
        --datafile_sep "$DATAFILE_SEP" \
        --get_kegg_pathways True \
        --binarize_threshold $BINARIZE_THRESHOLD \
        --organism $ORGANISM_CODE \
        --minimum_overlap $MINIMUM_OVERLAP
    fi

    # Using a custom network saved to the scBONITA directory:

    # Check and execute for Custom Pathways if the array is not empty
    if [ ${#CUSTOM_PATHWAYS[@]} -gt 0 ]; then
        echo "Running with Custom Pathways..."
        
        CUSTOM_PATHWAYS_ARGS=""
        for pathway in "${CUSTOM_PATHWAYS[@]}"; do
            CUSTOM_PATHWAYS_ARGS+="--network_files $pathway "
        done

        $CONDA_ENVIRONMENT_PYTHON pipeline_class.py \
        --data_file "$DATA_FILE" \
        --dataset_name "$DATASET_NAME" \
        --datafile_sep "$DATAFILE_SEP" \
        $CUSTOM_PATHWAYS_ARGS \
        --binarize_threshold $BINARIZE_THRESHOLD \
        --get_kegg_pathways "False" \
        --minimum_overlap $MINIMUM_OVERLAP
    else
        echo "No Custom Pathways specified, skipping this part..."
    fi
fi

#  --------------------------------------
# |     IMPORTANCE SCORE CALCULATION     |
#  --------------------------------------

if [ "$RUN_IMPORTANCE_SCORE" = "True" ]; then
    echo "Running Importance Score Calculation..."

    if [ ${#KEGG_PATHWAYS[@]} -gt 0 ]; then
        echo "Running with KEGG pathways"
        # Using a list of KEGG pathways:
        KEGG_PATHWAYS_ARGS="${KEGG_PATHWAYS[@]}"

        $CONDA_ENVIRONMENT_PYTHON importance_scores.py \
            --dataset_name "$DATASET_NAME" \
            --list_of_kegg_pathways $KEGG_PATHWAYS_ARGS
    else
        echo "No KEGG Pathways specified"
    fi
fi

#  -----------------------------------------
# |     RELATIVE ABUNDANCE CALCULATIONS     |
#  -----------------------------------------

if [ "$RUN_RELATIVE_ABUNDANCE" = "True" ]; then
    echo "Running Relative Abundance Calculations..."

    GROUP_INDICES_ARGS="${GROUP_INDICES[@]}"

    # Check that both arrays have the same length
    if [ ${#CONTROL_GROUPS[@]} -ne ${#EXPERIMENTAL_GROUPS[@]} ]; then
        echo "Control and Experimental groups arrays do not match in length!"
        exit 1
    fi

    if [ ${#KEGG_PATHWAYS[@]} -gt 0 ]; then
        echo "Running with KEGG pathways"
        # Using a list of KEGG pathways:
        KEGG_PATHWAYS_ARGS="${KEGG_PATHWAYS[@]}"
    else
        echo "No KEGG Pathways specified"
    fi

    # Loop through the control and experimental groups
    for (( i=0; i<${#CONTROL_GROUPS[@]}; i++ )); do

        # Extract the current pair of control and experimental group
        CONTROL_GROUP=${CONTROL_GROUPS[$i]}
        EXPERIMENTAL_GROUP=${EXPERIMENTAL_GROUPS[$i]}

        # Execute the command with the current pair of control and experimental group
        $CONDA_ENVIRONMENT_PYTHON relative_abundance.py \
            --dataset_name "$DATASET_NAME" \
            --dataset_file "$DATA_FILE" \
            --metadata_file "$METADATA_FILE" \
            --metadata_sep "$METADATA_SEP" \
            --dataset_sep "$DATAFILE_SEP" \
            --control_group "$CONTROL_GROUP" \
            --experimental_group "$EXPERIMENTAL_GROUP" \
            --cell_name_index $CELL_NAME_COL \
            --group_indices $GROUP_INDICES_ARGS \
            --header "$HEADER" \
            --overwrite "$OVERWRITE" \
            --organism "$ORGANISM_CODE" \
            --list_of_kegg_pathways $KEGG_PATHWAYS_ARGS
        done
fi

#  --------------------------------------
# |          ATTRACTOR ANALYSIS          |
#  --------------------------------------

# Runs the attractor analysis, requires importance score calculations
if [ "$RUN_ATTRACTOR_ANALYSIS" = "True" ]; then
    echo "Running Attractor Analysis..."

    $CONDA_ENVIRONMENT_PYTHON attractor_analysis.py \
        --dataset_name "$DATASET_NAME"
fi