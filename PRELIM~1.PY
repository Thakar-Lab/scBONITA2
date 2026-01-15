import os
import csv
from igraph import Graph, plot
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
import sklearn
from sklearn.manifold import TSNE
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# Directory containing the trajectory files
trajectory_dir = os.path.dirname(os.path.realpath(__file__))

# Dictionary to store columns from all files
columns_dict = {}
# Dictionary to store the trajectory of each file
trajectory_dict = {}

# Read each trajectory file
for filename in os.listdir(os.path.join(trajectory_dir, "..")):
    if filename.startswith("cell_") and filename.endswith("_trajectory.csv"):
        filepath = os.path.join(trajectory_dir, "..", filename)
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            # Read the trajectory from the file
            columns = []
            for row in reader:
                for i in range(1, len(row)):
                    if len(columns) < i:
                        columns.append([])
                    try:
                        columns[i-1].append(int(row[i]))  # Read columns excluding the first one
                    except ValueError:
                        print(f"Skipping non-numeric value: {row[i]}")

            # Once each file is read, store the columns in a dictionary
            # Create a dictionary to store the columns if they are unique
            state_counter = len(columns_dict)  # Start counting from the current number of states
            single_trajectory = []
            for i, column in enumerate(columns):
                column_tuple = tuple(column)
                if column_tuple not in columns_dict.values():
                    state_counter += 1
                    columns_dict[f"State {state_counter}"] = column_tuple
                    # store the state number as a part of the trajectory
                    single_trajectory.append(state_counter)
                else:
                    # Find the state number for the trajectory
                    for key, value in columns_dict.items():
                        if value == column_tuple:
                            single_trajectory.append(int(key.split()[1]))
                            break
            # save the single_trajectory to a dictionary
            trajectory_dict[filename[:-4]] = single_trajectory
# Print out all the unique trajectory into a combined txt file
with open("Combined_Trajectory.txt", 'w') as f:
    # Sort the keys to ensure they are in order
    for key in sorted(trajectory_dict.keys(), key=lambda x: int(x.split('_')[1])):
        value = trajectory_dict[key]
        f.write(f"{key}: {value}\n")

# Check the value in the trajectory_dict as occurance
# Check the occurrence of each state  
state_occurrence = {}
for key, value in trajectory_dict.items():
    for state in value:
        if state not in state_occurrence:
            state_occurrence[state] = 1
        else:
            state_occurrence[state] += 1
            # write the state occurrence to the file
with open("Combined_State_Occurrence.txt", 'w') as f:    
    for key, value in state_occurrence.items(): 
        f.write(f"State {key}: {value}\n")            
    
# Calculate the probability of each state
state_probability = {}
total_states = sum(state_occurrence.values())
for key, value in state_occurrence.items():
    state_probability[key] = value / total_states
    #Write the state probability to the existing file
with open("Combined_State_Probability.txt", 'a') as f:
    for key, value in state_probability.items():
        f.write(f"State {key} Probability: {value}\n")

# Print out all the unique states into a combined txt file
with open("Combined_Unique_States.txt", 'w') as f:
    for key, value in columns_dict.items():
        f.write(f"{key}: {value}\n")
# Make t-SNE plot of the unique states

# Use Combined_Unique_States.txt to create a t-SNE plot
# Transpose the columns_dict to get the states as rows
# Prepare data for t-SNE
unique_states = list(columns_dict.values())
state_labels = list(columns_dict.keys())

# Convert to numpy array
data = np.array(unique_states)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(data)

# Get the sizes for each state based on their probabilities
sizes = [state_probability[int(key.split()[1])] * 1000 for key in state_labels]

# Plot the t-SNE results
fig1 = plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=sizes, marker='s')
plt.title("t-SNE Plot of Unique States")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
# Save the plot
fig1.savefig("t-SNE_Plot.png")

# Create a 3D plot of the unique states with z-axis as the probability of occurrence of the state 
# Prepare data for 3D plot
x = tsne_results[:, 0]
y = tsne_results[:, 1]
z = [state_probability[int(key.split()[1])] for key in state_labels]

# Create a 3D plot

# Define a custom colormap from light blue to dark blue
blues_cmap = LinearSegmentedColormap.from_list("custom_blues", ["#add8e6", "#00008b"])

fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, s=sizes, c=z, cmap=blues_cmap, marker='s')

# Add labels and title
ax.set_title("3D Plot of Unique States with State Probabilities")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_zlabel("State Probability")

# Add color bar
cbar = fig2.colorbar(sc)
cbar.set_label('State Probability')
plt.show()
# Save the plot
fig2.savefig("3D_Plot.png")
