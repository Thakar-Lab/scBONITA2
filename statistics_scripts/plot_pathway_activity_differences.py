import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

matplotlib.use("Agg")

rel_abund_path = "/home/emoeller/github/scBONITA2/scBONITA_output/relative_abundance_output/george_hiv/HIV_vs_Healthy/text_files"

# Find the relative abundance differences between each pathway
relative_abundance_diff_dict = {
    "hsa04210": 0.49938571912541685,
    "hsa04010": 1.041349779769243,
    "hsa04620": 0.25331731911958005,
    "hsa04630": 0.735211917670854,
    "hsa04064": 0.19394701593669822,
    "hsa04666": 0.32410096519796655,
    "hsa04150": 0.22785747178729307,
    "hsa04060": 0.818081096281327,
    "hsa04621": 0.4713028140521658,
    
}
    
# Convert the dictionary to a DataFrame
relative_abundance_df = pd.DataFrame.from_dict(relative_abundance_diff_dict, orient='index', columns=["Relative_Abundance"])
print(relative_abundance_df.head())
relative_abundance_df.sort_values(by="Relative_Abundance", ascending=False, inplace=True)

# Normalize the relative abundance differences
max_value = relative_abundance_df["Relative_Abundance"].max()
relative_abundance_df["Normalized_Abundance"] = relative_abundance_df["Relative_Abundance"]

# Sort the DataFrame by Relative Abundance in ascending order
relative_abundance_df = relative_abundance_df.sort_values(by="Relative_Abundance", ascending=False)

# Create a bubble plot with updated adjustments
plt.figure(figsize=(6, 10))
plt.scatter(
    relative_abundance_df["Relative_Abundance"],  # X-axis: relative abundance
    relative_abundance_df.index,  # Y-axis: pathway names
    s=100,  # Fixed bubble size
    color='steelblue'
)

# Adjust x-axis limits
plt.xlim(0, 2)

# Add labels and title
plt.title("Bubble Plot of Relative Abundance by Pathway", fontsize=14)
plt.xlabel("Relative Abundance", fontsize=12)
plt.ylabel("Pathway", fontsize=12)
plt.tight_layout()

# Save the plot
plt.savefig("statistics_scripts/normalized_relative_abundance_by_pathway.png", dpi=200)
plt.close()

print("Plot saved as 'normalized_relative_abundance_by_pathway.png'.")   
    
cell_traj_path = "/home/emoeller/github/scBONITA2/scBONITA_output/trajectories"

cell_signaling = {}

cell_groups = pd.read_csv(f"{cell_traj_path}/george_hiv_cell_groups.csv", header=0, index_col=0)
print(cell_groups.head())

cols_of_interest = ["Cell", "Group"]
cell_groups = cell_groups[cols_of_interest]
cell_groups["Cell"] = cell_groups["Cell"].astype(str)

# Iterate through each pathway
pathway_names = []
for file in os.listdir(cell_traj_path):
    for pathway_name in relative_abundance_diff_dict.keys():
        
        if pathway_name in file:
            print(f'Analyzing {pathway_name}')
            pathway_names.append(pathway_name)
            cell_traj_dir = f'{cell_traj_path}/{file}/text_files/cell_trajectories'
            cell_groups[pathway_name] = np.nan
            
            # Iterate through each cell trajectory file and sum the number of active genes in the pathway 
            # along the trajectory
            for traj_file in tqdm(os.listdir(cell_traj_dir)):
                cell_num = traj_file.split("_")[1]
                df = pd.read_csv(f'{cell_traj_dir}/{traj_file}', index_col=0, header=None)
                cell_traj_sum = df.values.sum()
                
                cell_groups.loc[cell_groups["Cell"] == cell_num, [pathway_name]] = cell_traj_sum

cell_groups = cell_groups.dropna()            
print(cell_groups.head())

hiv_group = cell_groups[cell_groups["Group"] == "HIV"]
healthy_group = cell_groups[cell_groups["Group"] == "Healthy"]

signaling_by_group = pd.DataFrame({"HIV": hiv_group.describe().loc['mean'], "Healthy": healthy_group.describe().loc['mean']})
signaling_by_group["Difference"] = (signaling_by_group["HIV"] / signaling_by_group["Healthy"]) - 1

signaling_by_group = signaling_by_group.sort_values(by="Difference", ascending=False)

print(signaling_by_group)       

# Plot with X and Y axes switched
plt.figure(figsize=(6, 8))
plt.scatter(
    signaling_by_group["Difference"],
    signaling_by_group.index,
    s=100,
    color='steelblue'
    )

plt.title('Normalized Difference Between HIV and Healthy', fontsize=14)
plt.xlabel('Normalized Difference (Scaled -1 to 1)', fontsize=12)
plt.ylabel('Pathway', fontsize=12)
plt.xlim([-1,1])
plt.tight_layout()
plt.savefig("statistics_scripts/hiv_vs_healthy_total_cell_trajectory_signaling_differences.png", dpi=200)
plt.close()
