import os
import re
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.image as mpimg
import matplotlib.cm as cm

# -------------------- Utility Functions --------------------

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def read_trajectories(file_path):
    with open(file_path, 'r') as file:
        return {
            cell.strip(): list(map(int, traj.strip().strip('[]').split(',')))
            for cell, traj in (line.strip().split(':') for line in file)
        }

def clean_trajectory(trajectory):
    cleaned, seen = [], set()
    for state in trajectory:
        if state in seen:
            break
        cleaned.append(state)
        seen.add(state)
    return cleaned

def find_attractors(trajectory):
    visited = {}
    for i, state in enumerate(trajectory):
        if state in visited:
            cycle_start = visited[state]
            cycle_states = trajectory[cycle_start:i]
            attractor_state = cycle_states[-1]
            return attractor_state, cycle_states, attractor_state, trajectory[0]
        visited[state] = i
    return trajectory[-1], [], trajectory[-1], trajectory[0]

def plot_stg(trajectories):
    stg = nx.DiGraph()
    for cell, traj in trajectories.items():
        if len(set(traj)) == 1:
            continue
        for i in range(len(traj) - 1):
            stg.add_edge(traj[i], traj[i + 1])
            label = stg[traj[i]][traj[i + 1]].get('label', '')
            stg[traj[i]][traj[i + 1]]['label'] = f"{label}, {cell.split('_')[1]}".strip(', ')
    return stg

def plot_tsne(data, title, filename, marker):
    tsne_results = TSNE(n_components=2, random_state=24, perplexity=150).fit_transform(data)
    plt.figure()
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=1, marker=marker)
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
    return tsne_results

def plot_tsne_3d(data, title, filename, marker):
    tsne_results = TSNE(n_components=3, random_state=24, perplexity=150).fit_transform(data)
    plt.figure()
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=1, marker=marker)
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
    return tsne_results

def plot_3d_surface(tsne_results, z, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    grid_x, grid_y = np.mgrid[
        min(tsne_results[:, 0]):max(tsne_results[:, 0]):100j,
        min(tsne_results[:, 1]):max(tsne_results[:, 1]):100j
    ]
    grid_z = griddata((tsne_results[:, 0], tsne_results[:, 1]), z, (grid_x, grid_y), method='cubic')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("State Probability")
    plt.show()

def calculate_similarity_matching(*arrays):
    max_length = max(len(arr) for arr in arrays)
    padded_arrays = []
    for arr in arrays:
        arr = list(arr)
        if len(arr) < max_length:
            arr = [arr[0]] * (max_length - len(arr)) + arr if arr else [0] * max_length
        elif len(arr) > max_length:
            arr = arr[-max_length:]
        padded_arrays.append(np.array(arr))
    stacked = np.vstack(padded_arrays)
    matching = np.all(stacked == stacked[0], axis=0).sum()
    return matching / max_length

def find_most_traveled_paths(basins, trajectories, output_file="most_traveled_paths.txt"):
    most_traveled_paths = {}
    for attractor, cells in basins.items():
        transition_counts, path_counts = {}, {}
        for cell in cells:
            traj = trajectories[cell]
            for i in range(len(traj) - 1):
                transition = (traj[i], traj[i + 1])
                transition_counts[transition] = transition_counts.get(transition, 0) + 1
            traj_tuple = tuple(traj)
            path_counts[traj_tuple] = path_counts.get(traj_tuple, 0) + 1
        if path_counts:
            most_common_path = max(path_counts, key=path_counts.get)
            most_common_count = path_counts[most_common_path]
            most_traveled_paths[attractor] = (most_common_path, most_common_count)
        else:
            most_traveled_paths[attractor] = None
    with open(output_file, "w") as f:
        for attractor, result in most_traveled_paths.items():
            if result is not None:
                path, count = result
                f.write(f"Attractor {attractor}: most traveled path is {path} with {count} occurrences\n")
            else:
                f.write(f"Attractor {attractor}: no paths recorded\n")
    return most_traveled_paths

def compute_state_and_cell_transition_matrices(cleaned_trajectories, state_dict, trajectories):
    unique_states = [state_dict[key] for key in sorted(state_dict.keys(), key=lambda x: int(x.split()[1]))]
    num_states = len(unique_states)
    num_cells = len(cleaned_trajectories)
    state_transition_matrix = np.zeros((num_states, num_states))
    cell_transition_matrix = np.zeros((num_cells, num_cells))
    state_occupancy = np.zeros((num_cells, num_states))
    weighted_transition_matrix = np.zeros((num_states, num_states))
    for cell_id, traj in enumerate(cleaned_trajectories.values()):
        if not traj:
            continue
        for state in traj:
            if 1 <= state <= num_states:
                state_occupancy[cell_id, state - 1] = 1
        for i in range(len(traj) - 1):
            state_i, state_j = traj[i], traj[i + 1]
            row_idx, col_idx = state_i - 1, state_j - 1
            state_transition_matrix[row_idx, col_idx] += 1
            weighted_transition_matrix[row_idx, col_idx] += 1
    row_sums = state_transition_matrix.sum(axis=1, keepdims=True)
    non_zero_rows = row_sums.flatten() != 0
    state_transition_matrix[non_zero_rows] /= row_sums[non_zero_rows]
    for i in range(num_cells):
        for j in range(num_cells):
            if i != j:
                cell_transition_matrix[i, j] = np.sum(state_occupancy[i, :] * state_transition_matrix @ state_occupancy[j, :])
    row_sums_cell = cell_transition_matrix.sum(axis=1, keepdims=True)
    non_zero_rows_cell = row_sums_cell.flatten() != 0
    if non_zero_rows_cell.any():
        cell_transition_matrix[non_zero_rows_cell] /= row_sums_cell[non_zero_rows_cell]
    return state_transition_matrix, state_occupancy, cell_transition_matrix, weighted_transition_matrix

def generalized_jaccard_similarity(arr1, arr2):
    arr1, arr2 = np.array(arr1), np.array(arr2)
    matching = np.sum(arr1 == arr2)
    return matching / len(arr1)

def compute_weighted_pseudotime(trajectories, attractor_value_dict):
    pseudotime_dict, max_observed_steps_dict, observed_steps_dict, shortest_path_length_dict = {}, {}, {}, {}
    trajectory_keys = list(trajectories.keys())
    dissimilarity_matrix = np.zeros((len(trajectory_keys), len(trajectory_keys)))
    unique_attractors = set()
    for cell, trajectory in trajectories.items():
        cleaned_trajectory = clean_trajectory(trajectory)
        single_attractor, cyclic_attractor, attractor_state, initial_state = find_attractors(cleaned_trajectory)
        attractor = single_attractor if single_attractor is not None else cyclic_attractor[0]
        observed_steps = cleaned_trajectory.index(attractor) + 1
        observed_steps_dict[cell] = observed_steps
        unique_attractors.add(attractor)
    max_observed_steps = max(observed_steps_dict.values())
    min_observed_steps = min(observed_steps_dict.values())
    for cell, observed_steps in observed_steps_dict.items():
        max_observed_steps_dict[cell] = max_observed_steps
        shortest_path_length_dict[cell] = min_observed_steps
        if max_observed_steps != min_observed_steps:
            pseudotime = 1 - (np.abs(observed_steps - min_observed_steps) / (max_observed_steps - min_observed_steps))
        else:
            pseudotime = 0
        pseudotime_dict[cell] = pseudotime
    sorted_attractors = sorted(unique_attractors)
    for i, cell in enumerate(trajectory_keys):
        cleaned_trajectory = clean_trajectory(trajectories[cell])
        attractor = find_attractors(cleaned_trajectory)[0] or find_attractors(cleaned_trajectory)[1][0]
        for j, cell2 in enumerate(trajectory_keys):
            if i == j:
                dissimilarity_matrix[i, j] = 0
                continue
            cleaned_trajectory2 = clean_trajectory(trajectories[cell2])
            basins = {att: [c for c, traj in trajectories.items() if att in traj] for att in unique_attractors}
            most_traveled_paths = find_most_traveled_paths(basins, trajectories)
            most_traveled_path = most_traveled_paths.get(attractor, ([], 0))[0]
            if most_traveled_path:
                most_traveled_path = clean_trajectory(most_traveled_path)
            max_len = max(len(cleaned_trajectory), len(cleaned_trajectory2), len(most_traveled_path))
            padded_cleaned_trajectory = [cleaned_trajectory[0]] * (max_len - len(cleaned_trajectory)) + cleaned_trajectory
            padded_cleaned_trajectory2 = [cleaned_trajectory2[0]] * (max_len - len(cleaned_trajectory2)) + cleaned_trajectory2
            padded_most_traveled_path = [most_traveled_path[0]] * (max_len - len(most_traveled_path)) + most_traveled_path if most_traveled_path else []
            similarity_weight1_ij = calculate_similarity_matching(padded_cleaned_trajectory, padded_cleaned_trajectory2)
            if most_traveled_path:
                similarity_weight2_ijl = calculate_similarity_matching(padded_cleaned_trajectory, padded_most_traveled_path) if cleaned_trajectory in most_traveled_path else calculate_similarity_matching(padded_cleaned_trajectory2, padded_most_traveled_path)
            else:
                similarity_weight2_ijl = similarity_weight1_ij
            attractor1 = attractor_value_dict.get(int(cell.split('_')[1]), [])
            attractor2 = attractor_value_dict.get(int(cell2.split('_')[1]), [])
            similarity_weight3_ij = calculate_similarity_matching(attractor1, attractor2)
            attractor3 = attractor_value_dict.get(sorted_attractors[0], [])
            similarity_weight4_ijk = 1 if attractor1 == attractor3 or attractor2 == attractor3 else calculate_similarity_matching(attractor1, attractor2, attractor3)
            similarity_score = (similarity_weight1_ij + similarity_weight2_ijl + similarity_weight3_ij + similarity_weight4_ijk) / 4
            tolerance = 1e-2
            pseudotime_diff = np.abs(pseudotime_dict[cell] - pseudotime_dict[cell2])
            if similarity_score > 0.999 and pseudotime_diff < tolerance:
                distance_score = 0
            else:
                distance_score = 1 - (similarity_score * (1 - pseudotime_diff))
            if pseudotime_dict[cell] == 1 or pseudotime_dict[cell2] == 1:
                distance_score = 1 - similarity_score
            if similarity_weight1_ij == 1:
                distance_score = 0
            dissimilarity_matrix[i, j] = distance_score
    return pseudotime_dict, dissimilarity_matrix

def attractor_ASplus_minus_analysis(attractor_value_to_check, cell_attractor_file, metadata_file):
    as_plus_count = 0
    as_minus_count = 0
    total_cells = 0
    metadata = pd.read_csv(metadata_file, sep=" ", header=None).reset_index(drop=True)
    with open(cell_attractor_file, 'r') as f:
        lines = f.readlines()
    as_plus_indices, as_minus_indices = [], []
    for line in lines:
        match = re.search(r"Attractor:\s*(\d+),", line)
        if match:
            attractor_value = int(match.group(1))
            if attractor_value == attractor_value_to_check:
                total_cells += 1
                cell_num_trajectory = line.split(':')[0].split('_')[1]
                cell_state = metadata.iloc[int(cell_num_trajectory), 2]
                if cell_state == "AS+":
                    as_plus_count += 1
                    as_plus_indices.append(cell_num_trajectory)
                elif cell_state == "AS-":
                    as_minus_count += 1
                    as_minus_indices.append(cell_num_trajectory)
    if total_cells > 0:
        as_plus_percentage = (as_plus_count / total_cells) * 100
        as_minus_percentage = (as_minus_count / total_cells) * 100
    else:
        as_plus_percentage = 0
        as_minus_percentage = 0
    return as_plus_count, as_minus_count, total_cells, as_plus_percentage, as_minus_percentage, as_plus_indices, as_minus_indices

def jitter_positions(pos, jitter=0.01, min_dist=0.05, max_iter=100):
    """
    Jitter node positions to reduce overlap.
    Args:
        pos: dict of node positions {node: (x, y)}
        jitter: float, base jitter to apply
        min_dist: float, minimum allowed distance between nodes
        max_iter: int, maximum number of jittering iterations
    Returns:
        dict of adjusted positions
    """
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes])
    for _ in range(max_iter):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < min_dist:
                    # Move nodes apart
                    direction = coords[i] - coords[j]
                    if np.all(direction == 0):
                        direction = np.random.randn(2)
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    coords[i] += direction * jitter
                    coords[j] -= direction * jitter
                    moved = True
        if not moved:
            break
    return {n: tuple(coords[i]) for i, n in enumerate(nodes)}


# -------------------- Main Function -------------------------------------------------------------------------------------------
# -------------------- Input and Output Files ----------------------------------------------------------------------------------
INPUT_FILES = {
    "trajectories": "Combined_Trajectory.txt",
    "unique_states": "Combined_Unique_States.txt",
    "cell_trajectory": "cell_1_trajectory.csv",
    "metadata": r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\input\tutorial_data\combined_metadata.txt",
    "importance_scores": r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\scBONITA_output\importance_score_output\tutorial_dataset\text_files\hsa05417_importance_score.txt",
    "knockout_results": r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\scBONITA_output\importance_score_output\tutorial_dataset\intermediate_files\hsa05417\knockout_results_{gene}.pkl",
    "knockin_results": r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\scBONITA_output\importance_score_output\tutorial_dataset\intermediate_files\hsa05417\knockin_results_{gene}.pkl",
    "network": r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\scBONITA_output\graphml_files\tutorial_dataset\hsa05417_processed.graphml",
    "foldchange": r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\scBONITA_output\relative_abundance_output\tutorial_dataset\AS+_vs_AS-\text_files\hsa05417_tutorial_dataset_AS+_vs_AS-_relative_abundance.txt"
}

OUTPUT_FILES = {
    "cleaned_trajectory": "cleaned_trajectory.txt",
    "cell_and_attractor": "cell_and_attractor.txt",
    "attractor_value_dict": "attractor_value_dict.txt",
    "most_traveled_paths": "most_traveled_paths.txt",
    "report_steps_to_attractor": "report_steps_to_attractor.txt",
    "report_normalized_steps": "report_normalized_steps.txt",
    "tsne_plot": "t-SNE_Plot_Colored.png",
    "tsne_plot_pseudotime": "t-SNE_Plot_Colored_with_Pseudotime.png",
    "condition_bias_histogram": "Condition_Bias_Histogram.png",
    "bifurcation_diagram_ki": "Bifurcation_Diagram_Perturbed_KI_{gene}.png",
    "bifurcation_diagram_ko": "Bifurcation_Diagram_Perturbed_KO_{gene}.png",
    "condition_bias_histogram_ki": "Condition_Bias_Histogram_Perturbed_KI_{gene}.png",
    "condition_bias_histogram_ko": "Condition_Bias_Histogram_Perturbed_KO_{gene}.png",
}

def main():
    # Load importance scores
    gene_names = []
    importance_scores = []
    with open(INPUT_FILES["importance_scores"], 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                gene, score = parts
                try:
                    importance_scores.append(float(score))
                    gene_names.append(gene)
                except ValueError:
                    print(f"Warning: Could not convert score '{score}' for gene '{gene}' to float. Skipping.")
    
    print(f"Loaded importance scores for {len(importance_scores)} genes.")
    print(f"Importance scores: {importance_scores[:10]}...")
    
    # Print top thirty important genes
    top_30_indices = np.argsort(importance_scores)[-30:][::-1]
    print("Top 30 important genes and their scores:")
    for idx in top_30_indices:
        gene = gene_names[idx] if idx < len(gene_names) else f"Gene_{idx}"
        score = importance_scores[idx]
        print(f"{gene}: {score:.4f}")
    
    # Read fold change values
    foldchange_dict = {}
    with open(INPUT_FILES["foldchange"]) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 4:
                key = parts[0]
                try:
                    value = float(parts[4].strip())
                except ValueError:
                    value = 0.0
                foldchange_dict[key] = np.log2(value + 1e-6)

    # Plot network graph
    G = nx.read_graphml(INPUT_FILES["network"])
    node_list = [n for n in G.nodes() if n in gene_names]
    node_multiplier = 1
    node_sizes = [300 + 7000 * (importance_scores[gene_names.index(n)] / max(importance_scores)) if n in gene_names else 300 for n in node_list]
    node_sizes = np.array(node_sizes) * node_multiplier
    
    log_fold_changes = np.array([foldchange_dict.get(n, 0) for n in node_list])
    log_fold_changes = np.clip(log_fold_changes, -1.5, 1.5)
    print(f"Log fold changes (first 10): {log_fold_changes[:10]}")
    print(f"Smallest log fold change: {log_fold_changes.min()}, Largest log fold change: {log_fold_changes.max()}")

    cmap = plt.cm.coolwarm
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
    node_colors = [cmap(norm(val)) for val in log_fold_changes]

    plt.figure(figsize=(10, 7.5))
    pos = nx.spring_layout(G, seed=42, k=1.2)  
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=1.0, arrows=True, arrowsize=40, arrowstyle='-|>')
    nx.draw_networkx_labels(G, pos, labels={n: n for n in node_list}, font_size=12, font_color='black')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.axis('off')
    plt.tight_layout()
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=18)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    cbar.set_label("Log2 Fold Change", fontsize=18, fontweight = 'bold')
    if hasattr(cbar, 'ax'):
        cbar.ax.tick_params(labelsize=18)
    # Make colorbar outline bold
    cbar.outline.set_linewidth(2)    
    plt.savefig("Figure2ANetwork.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Top 10 node sizes and their corresponding genes:")
    top_10_size_indices = np.argsort(node_sizes)[-10:][::-1]
    for idx in top_10_size_indices:
        gene = node_list[idx] if idx < len(node_list) else f"Gene_{idx}"
        size = node_sizes[idx]
        print(f"{gene}: {size:.2f}")
    
    # Read trajectories
    trajectories = read_trajectories(INPUT_FILES["trajectories"])
    
    # Clean and save trajectories
    cleaned_trajectories = {}
    with open("cleaned_trajectory.txt", 'w') as f:
        for key, value in trajectories.items():
            cleaned_trajectory = clean_trajectory(value)
            cleaned_trajectories[key] = cleaned_trajectory
            f.write(f"{key}: {cleaned_trajectory}\n")
    
    # Plot state transition graph
    stg_combined = plot_stg(trajectories)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(stg_combined)
    nx.draw(stg_combined, pos, with_labels=True, node_size=500, node_color='lightgray', 
            edge_color='blue', font_size=8, font_color='black', connectionstyle="arc3,rad=0.11")
    plt.savefig("FigureSTG.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Read unique states
    with open("Combined_Unique_States.txt", 'r') as f:
        columns_dict = {key.strip(): list(map(int, value.strip().strip('[]').replace('(', '').replace(')', '').split(','))) 
                       for key, value in (line.strip().split(':') for line in f)}

    unique_states = list(columns_dict.values())
    state_labels = list(columns_dict.keys())
    data = np.array(unique_states)
    tsne_results = plot_tsne(data, "t-SNE Plot of Unique States", "t-SNE_Plot_Colored.png", marker='s')

    # Find attractors
    attractors = {}
    attractor_value_dict = {}
    with open("cell_and_attractor.txt", 'w') as f:
        for cell, trajectory in trajectories.items():
            cleaned_trajectory = clean_trajectory(trajectory)
            single_attractor, cyclic_attractor, attractor_state, _ = find_attractors(cleaned_trajectory)
            attractor = single_attractor if single_attractor is not None else cyclic_attractor[0]
            if attractor not in attractors:
                attractors[attractor] = []
            attractors[attractor].append(cell)
            formatted_attractor_state = f"State {attractor_state}"
            if formatted_attractor_state in columns_dict:
                attractor_value = columns_dict[formatted_attractor_state]
                f.write(f"{cell}: {trajectory} -> Attractor: {attractor_state}, Value: {attractor_value}\n")
                attractor_value_dict[int(cell.split('_')[1])] = attractor_value
            else:
                f.write(f"{cell}: {trajectory} -> Attractor: {attractor_state}\n")
                attractor_value_dict[int(cell.split('_')[1])] = attractor_state
    
    with open("attractor_value_dict.txt", 'w') as f:
        print(attractor_value_dict, file=f)

    # Color by attractors
    attractor_probabilities = {attractor: len(cells) / len(trajectories) for attractor, cells in attractors.items()}
    unique_attractors = list(attractors.keys())
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_attractors)))
    attractor_colors = {attractor: colors[i] for i, attractor in enumerate(unique_attractors)}
    
    fig2 = plt.figure()
    for i, state in enumerate(state_labels):
        state_index = int(state.split()[1])
        color = attractor_colors.get(state_index, 'gray')
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=color, alpha=0.8, label=f"Attractor {state_index}", marker='s')
    plt.title("t-SNE Plot of Unique States Colored by Attractors")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

    # Basin analysis
    basins = {attractor: [cell for cell, trajectory in trajectories.items() if attractor in trajectory] for attractor in unique_attractors}
    sorted_attractors = sorted(basins, key=lambda x: len(basins[x]), reverse=True)
    sorted_sizes = [len(basins[attractor]) for attractor in sorted_attractors]

    # Figure 2C: Bar plot
    cmap_blues = plt.get_cmap('Blues')
    colors_bars = [cmap_blues(i / 4) for i in reversed(range(5))]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    bars = ax.bar(range(5), sorted_sizes[:5], tick_label=sorted_attractors[:5], color=colors_bars, edgecolor='black', linewidth=2)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=18, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Attr {i+1}" for i in range(5)], rotation=45, fontsize=18)
    ax.set_xlabel("Top Attractors", fontsize=18, fontweight='bold')
    ax.set_ylabel("Number of Cells", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig("Figure2C.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 2D: Heatmap - FIXED VERSION
    state_genes = {}
    with open("Combined_Unique_States.txt", 'r') as f:
        for line in f:
            key, value = line.strip().split(':')
            state_genes[int(key.split()[1])] = np.array(list(map(int, value.strip().strip('[]').replace('(', '').replace(')', '').split(','))))
    
    # Read gene names from cell trajectory file
    current_dir = r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\scBONITA_output\trajectories\tutorial_dataset_hsa05417\text_files\cell_trajectories"
    print(f"Current directory: {current_dir}")
    cell_trajectory_path = os.path.join(current_dir, "cell_1_trajectory.csv")
    if not os.path.exists(cell_trajectory_path):
        parent_dir = os.path.dirname(current_dir)
        cell_trajectory_path = os.path.join(parent_dir, "cell_1_trajectory.csv")
        if not os.path.exists(cell_trajectory_path):
            raise FileNotFoundError("cell_1_trajectory not found in current or parent directory.")
    
    with open(cell_trajectory_path, 'r') as f:
        lines = f.readlines()
        gene_names_list = [line.strip().split(',')[0] for line in lines]
    
    print(f"Total genes in gene_names_list: {len(gene_names_list)}")
    
    # Create heatmap for genes that are ON in at least one of top 5 attractors
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Build matrix for top 5 attractors: columns = attractors, rows = all genes
    num_genes_total = len(gene_names_list)
    attractor_gene_matrix = np.zeros((num_genes_total, 5), dtype=int)
    
    for col_idx, attractor in enumerate(sorted_attractors[:5]):
        if attractor in state_genes:
            state_vector = state_genes[attractor]
            # Copy the state vector values to the column
            # Make sure we don't go out of bounds
            copy_length = min(len(state_vector), num_genes_total)
            attractor_gene_matrix[:copy_length, col_idx] = state_vector[:copy_length]
    
    # Filter to only show genes that are ON (=1) in at least one attractor
    genes_on_mask = np.any(attractor_gene_matrix == 1, axis=1)
    filtered_matrix = attractor_gene_matrix[genes_on_mask]
    filtered_gene_names = [gene_names_list[i] for i in range(len(gene_names_list)) if genes_on_mask[i]]
    
    print(f"Number of genes that are ON in at least one of top 5 attractors: {len(filtered_gene_names)}")
    
    # Limit to top 100 genes for visualization
    if len(filtered_gene_names) > 100:
        filtered_matrix = filtered_matrix[:100]
        filtered_gene_names = filtered_gene_names[:100]
    
    # Create colormap
    set1_colors = sns.color_palette("Set1", 10)
    cmap_heatmap = ListedColormap([set1_colors[8], set1_colors[1]])
    bounds = [-0.5, 0.5, 1.5]
    norm_heatmap = plt.cm.colors.BoundaryNorm(bounds, cmap_heatmap.N)
    
    # Plot heatmap (transpose to have genes as rows, attractors as columns)
    im = ax.imshow(filtered_matrix, cmap=cmap_heatmap, norm=norm_heatmap, aspect='auto')
    
    # Set x-axis labels (attractors)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Attr {i+1}" for i in range(5)], rotation=45, fontsize=18)
    ax.set_xlabel("Top Attractors", fontsize=18, fontweight='bold')
    
    # Set y-axis labels (genes with importance scores)
    gene_labels_with_scores = []
    for gene in filtered_gene_names:
        if gene in gene_names:
            score = importance_scores[gene_names.index(gene)]
            gene_labels_with_scores.append(f"{gene} ({score:.2f})")
        else:
            gene_labels_with_scores.append(gene)
    
    ax.set_yticks(range(len(filtered_gene_names)))
    ax.set_yticklabels(gene_labels_with_scores, fontsize=10)
    ax.set_ylabel("Selected Genes with Importance Scores", fontsize=18, fontweight = 'bold')
    
    # Add black borders around cells
    for (i, j), val in np.ndenumerate(filtered_matrix):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', lw=2))
    
    # Add legend
    legend_elements = [
        Patch(facecolor=set1_colors[8], edgecolor='black', label='Off'),
        Patch(facecolor=set1_colors[1], edgecolor='black', label='On')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18)
    
    # Set spine linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig("Figure2D.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Genes shown in Figure 2D: {filtered_gene_names}")

    #Figure 2B
    # Figure 2B: State transition graph for 5th largest attractor
    if len(sorted_attractors) >= 5:
        attractor = sorted_attractors[4]  # 5th largest (0-indexed)
        attractor_color = 'blue'
        attractor_cells = basins[attractor]
        attractor_trajs = [cleaned_trajectories[cell] for cell in attractor_cells if cell in cleaned_trajectories]
        
        # Build graph and track which trajectory each edge belongs to
        G_attractor = nx.DiGraph()
        edge_to_traj = {}  # Map edges to trajectory index
        
        # IMPORTANT: Track all terminal states (attractors) in these trajectories
        terminal_states = set()
        
        for traj_idx, traj in enumerate(attractor_trajs):
            if len(traj) == 0:
                continue
            
            # Add the last state as a terminal state
            terminal_states.add(traj[-1])
            
            for i in range(len(traj) - 1):
                edge = (traj[i], traj[i + 1])
                G_attractor.add_edge(traj[i], traj[i + 1])
                edge_to_traj[edge] = traj_idx
            
            # Add self-loop for attractor state if it exists
            if len(traj) > 0:
                final_state = traj[-1]
                if final_state == attractor or traj.count(final_state) > 1:
                    G_attractor.add_edge(final_state, final_state)
        
        print(f"\nFigure 2B: Attractor {attractor} basin")
        print(f"Number of trajectories: {len(attractor_trajs)}")
        print(f"Terminal states found: {terminal_states}")
        print(f"Expected attractor state: {attractor}")
        
        # Verify all trajectories end at the attractor
        mismatch_count = 0
        for traj in attractor_trajs:
            if len(traj) > 0 and traj[-1] != attractor:
                mismatch_count += 1
                print(f"Warning: Trajectory ends at {traj[-1]} instead of {attractor}")
        
        if mismatch_count > 0:
            print(f"WARNING: {mismatch_count} trajectories don't end at attractor {attractor}!")
            print(f"This suggests the basin assignment might be incorrect.")

    if len(G_attractor) > 0:
        plt.figure(figsize=(20, 20))
        pos = nx.kamada_kawai_layout(G_attractor)
        pos = jitter_positions(pos, jitter=0.15, min_dist=0.08, max_iter=200)
        
        # Identify attractor states (should be the terminal states)
        attractor_states = terminal_states.copy()
        
        # Also check for self-loops
        for node in G_attractor.nodes():
            if G_attractor.has_edge(node, node):
                attractor_states.add(node)
        
        print(f"Attractor states to highlight: {attractor_states}")
        
        # Move attractor states to center
        if attractor_states:
            center = np.array([0.0, 0.0])
            if len(attractor_states) == 1:
                # Single attractor - place at exact center
                attractor_state = list(attractor_states)[0]
                pos[attractor_state] = center
            else:
                # Multiple attractor states - arrange in small circle around center
                radius = 0.1
                for i, state in enumerate(attractor_states):
                    angle = 2 * np.pi * i / len(attractor_states)
                    pos[state] = center + radius * np.array([np.cos(angle), np.sin(angle)])
        
        labels = {node: f"S{node}" for node in G_attractor.nodes()}
        
        # Calculate node alphas based on proximity (density)
        node_alphas = []
        positions = np.array([pos[node] for node in G_attractor.nodes()])
        for i, node in enumerate(G_attractor.nodes()):
            # Calculate distances to all other nodes
            distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
            # Count nearby nodes (within threshold)
            nearby = np.sum(distances < 0.2)
            # More nearby nodes = lower alpha (more transparent)
            # BUT ensure attractor states are always fully opaque
            if node in attractor_states:
                alpha = 1.0
            else:
                alpha = max(0.3, 1.0 - (nearby - 1) * 0.05)
            node_alphas.append(alpha)
        
        # Separate regular and attractor nodes
        regular_nodes = [(node, alpha) for node, alpha in zip(G_attractor.nodes(), node_alphas) if node not in attractor_states]
        attractor_nodes = [(node, alpha) for node, alpha in zip(G_attractor.nodes(), node_alphas) if node in attractor_states]
        
        # STEP 1: Draw regular nodes first
        for node, alpha in regular_nodes:
            nx.draw_networkx_nodes(
                G_attractor, pos, nodelist=[node],
                node_size=3000, node_color=attractor_color,
                alpha=alpha, node_shape='s'
            )
        
        # STEP 2: Draw edges
        num_trajs = len(attractor_trajs)
        rainbow_colors = cm.rainbow(np.linspace(0, 1, num_trajs))
        
        for traj_idx in range(num_trajs):
            traj_edges = [edge for edge, idx in edge_to_traj.items() if idx == traj_idx]
            
            if traj_edges:
                nx.draw_networkx_edges(
                    G_attractor, pos, edgelist=traj_edges,
                    edge_color='black',
                    width=6, arrows=True, 
                    arrowsize=40,
                    arrowstyle='-|>',
                    connectionstyle='arc3,rad=0.1',
                    alpha=0.6,
                    min_source_margin=15,
                    min_target_margin=15
                )
        
        # STEP 3: Draw regular node labels
        for node, alpha in regular_nodes:
            nx.draw_networkx_labels(
                G_attractor, pos, labels={node: labels[node]},
                font_size=12, font_color='white',
                alpha=min(1.0, alpha + 0.2)
            )
        
        # STEP 4: Draw attractor nodes (ON TOP OF EVERYTHING)
        for node, alpha in attractor_nodes:
            nx.draw_networkx_nodes(
                G_attractor, pos, nodelist=[node],
                node_size=4000,
                node_color='gold',
                alpha=1.0,
                node_shape='s',
                edgecolors='darkgoldenrod',
                linewidths=3
            )
        
        # STEP 5: Draw attractor labels (LAST - on top of everything)
        for node, alpha in attractor_nodes:
            nx.draw_networkx_labels(
                G_attractor, pos, labels={node: labels[node]},
                font_size=14, font_color='black',
                font_weight='bold',
                alpha=1.0
            )
        
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("Figure2B.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Pseudotime calculations
    steps_to_attractor = {state: ((len(clean_trajectory(trajectory)) - clean_trajectory(trajectory).index(state)) - 1)
                          for trajectory in cleaned_trajectories.values()
                          for state in clean_trajectory(trajectory)
                          if state in clean_trajectory(trajectory) and len(clean_trajectory(trajectory)) > 1}
    
    normalized_steps_to_attractor = {state: ((len(clean_trajectory(trajectory)) - clean_trajectory(trajectory).index(state)) - 1) / len(clean_trajectory(trajectory))
                                     for trajectory in cleaned_trajectories.values()
                                     for state in clean_trajectory(trajectory)
                                     if state in clean_trajectory(trajectory) and len(clean_trajectory(trajectory)) > 1}
    
    with open("report_steps_to_attractor.txt", "w") as f:
        for state, steps in steps_to_attractor.items():
            f.write(f"State {state}: Steps to Attractor: {steps}\n")
    
    with open("report_normalized_steps.txt", "w") as f:
        for state, steps in normalized_steps_to_attractor.items():
            f.write(f"State {state}: Normalized Steps to Attractor: {steps}\n")
    
    state_pseudotime_dict = {state: 1 - normalized_steps_to_attractor[state] for state in normalized_steps_to_attractor}
    state_vector_with_pseudotime = np.array([
        np.array(unique_states[i]) * (1 - normalized_steps_to_attractor.get(i, 0))
        for i in range(len(unique_states))
    ])
    
    tsne_results = plot_tsne(state_vector_with_pseudotime, "t-SNE Plot of Unique States with Pseudotime", "t-SNE_Plot_Colored_with_Pseudotime.png", marker='s')
    
    # Figure 2E: 3D t-SNE plot with transitions
    fig = plt.figure(figsize=(12, 10))
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=min(30, len(unique_states)-1), 
                   n_iter=1000, learning_rate=200, early_exaggeration=12.0)
    
    tsne_results_3d = tsne_3d.fit_transform(state_vector_with_pseudotime)

    # APPLY THE SPREAD FACTOR
    spread_factor = 5.0
    tsne_results_3d = tsne_results_3d * spread_factor
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot states colored by pseudotime
    for i, state in enumerate(state_labels):
        state_index = int(state.split()[1])
        color = plt.cm.coolwarm(1 - normalized_steps_to_attractor.get(state_index, 0))
        ax.scatter(tsne_results_3d[i, 0], tsne_results_3d[i, 1], tsne_results_3d[i, 2], 
                   color=color, alpha=0.6, label=f"State {state_index}", marker='s', edgecolors='none', s=100)
    
    # Set axis limits
    x_min, x_max = np.min(tsne_results_3d[:, 0]), np.max(tsne_results_3d[:, 0])
    y_min, y_max = np.min(tsne_results_3d[:, 1]), np.max(tsne_results_3d[:, 1])
    z_min, z_max = np.min(tsne_results_3d[:, 2]), np.max(tsne_results_3d[:, 2])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.view_init(elev=15, azim=-110)

    # Find attractor coordinates in 3D t-SNE space
    print("\nFinding attractor coordinates in 3D...")
    attractor_coordinates_3d = {}

    # Get indices where pseudotime = 0 (attractors)
    indices_pseudotime_0 = [
        i for i, state in enumerate(state_labels)
        if normalized_steps_to_attractor.get(int(state.split()[1]), 0) == 0
    ]

    for i in indices_pseudotime_0:
        state_index = int(state_labels[i].split()[1])
        x, y, z = tsne_results_3d[i, 0], tsne_results_3d[i, 1], tsne_results_3d[i, 2]
        attractor_coordinates_3d[state_index] = (x, y, z)

    print(f"Identified {len(attractor_coordinates_3d)} attractor coordinates in 3D")

    # # Draw attractors with MUCH LARGER markers and labels
    attractor_locations = {}
    label_positions = []

    for idx, attractor in enumerate(sorted_attractors[:5]):  # Top 5 attractors
        rank = idx + 1
        attractor_cells = basins[attractor]
        
        # attractor is already an integer state index - use it directly
        if attractor not in attractor_coordinates_3d:
            print(f"Warning: Attractor {rank} (state {attractor}) missing 3D coordinates → skipped.")
            continue

        text_x, text_y, text_z = attractor_coordinates_3d[attractor]
        attractor_locations[attractor] = (text_x, text_y, text_z)

        # Plot a VERY LARGE gold square
        ax.scatter(
            [text_x], [text_y], [text_z],
            marker='s',
            s=2000,
            facecolors='gold',
            edgecolors='black',
            linewidths=3,
            alpha=1.0,
            depthshade=False
        )

        label_positions.append((text_x, text_y, text_z, rank))

        print(
            f"Attractor {rank}: (State idx {attractor}) "
            f"Position=({text_x:.2f}, {text_y:.2f}, {text_z:.2f}), "
            f"Basin size={len(attractor_cells)}"
        )

    # ----------------------------------------------------------
    # Add labels with simple overlap avoidance
    # ----------------------------------------------------------
    # for i, (x, y, z, rank) in enumerate(label_positions):
    #     offset_x, offset_y, offset_z = 0, 0, 0

    #     # Push labels apart if they're too close
    #     for j, (ox, oy, oz, other_rank) in enumerate(label_positions):
    #         if i == j:
    #             continue
    #         dist = np.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
            
    #         if dist < 10:  # If closer than 10 units
    #             # Push away from the other label
    #             direction = np.array([x - ox, y - oy, z - oz])
    #             if np.linalg.norm(direction) > 0:
    #                 direction = direction / np.linalg.norm(direction)
    #                 offset_x += direction[0] * 5
    #                 offset_y += direction[1] * 5
    #                 offset_z += direction[2] * 5
    # ----------------------------------------------------------
    # Check distances between all pairs to identify overlaps
    # ----------------------------------------------------------
    print("\nDistances between attractors:")
    for i in range(len(label_positions)):
        for j in range(i+1, len(label_positions)):
            x1, y1, z1, rank1 = label_positions[i]
            x2, y2, z2, rank2 = label_positions[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            print(f"  Attr {rank1} <-> Attr {rank2}: distance = {dist:.2f}")

    #----------------------------------------------------------
    #Manual offsets for each attractor (adjust based on distances above)
    #----------------------------------------------------------
    manual_offsets = {
    1: (0, 0, 0),       # Far from others, no offset needed
    2: (0, 0, 0),       # Far from others, no offset needed
    3: (-10, -5, 10),   # Move left, down, and forward
    4: (10, 5, -10),    # Move right, up, and back (opposite of 3)
    5: (0, 0, 0),       # Far from others, no offset needed
    }

    # Add labels with manual offsets
    for i, (x, y, z, rank) in enumerate(label_positions):
        offset_x, offset_y, offset_z = manual_offsets.get(rank, (0, 0, 0))
        
        ax.text(
            x + offset_x,
            y + offset_y,
            z + offset_z,
            f"Attr {rank}",
            fontsize=18,
            fontweight='bold',
            color='black',
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='black', linewidth=2, pad=3),
            zorder=10000
        )    
        print(f"\nLabeled {len(label_positions)} attractors")

    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis] if hasattr(ax, 'w_xaxis') else [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis.line.set_linewidth(2)
        except AttributeError:
            try:
                axis.pane.set_linewidth(2)
            except Exception:
                pass

    for line in ax.get_lines():
        line.set_linewidth(2)
        
    ax.set_xlabel("t-SNE Component 1", fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel("t-SNE Component 2", fontsize=18, fontweight='bold', labelpad=15)
    ax.set_zlabel("t-SNE Component 3", fontsize=18, fontweight='bold', labelpad=15)
        
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='z', which='major', labelsize=18)
        
    # Add colorbar with bold labels and ticks
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Normalized Pseudotime", shrink=0.5, pad=0)
    cbar.ax.tick_params(labelsize=18)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    cbar.set_label("Normalized Pseudotime", fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("Figure2E.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Combine all figures into one panel
    images = [
        "Figure2ANetwork.png",
        "Figure2B.png",
        "Figure2C.png",
        "Figure2D.png",
        "Figure2E.png"
    ]

    img_data = [mpimg.imread(f) for f in images]
    labels = ["a.", "b.", "c.", "d.", "e."]

    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0, wspace=0)

    # First 4 images (2x2 grid)
    for i in range(4):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img_data[i])
        ax.axis('off')

        # Label placement logic
        if i == 1:  # second image, special label position
            ax.text(
                0.03, 0.98, labels[i], transform=ax.transAxes,
                fontsize=20, fontweight='bold', color='black',
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
            )
        else:
            ax.text(
                0.03, 1.1, labels[i], transform=ax.transAxes,
                fontsize=20, fontweight='bold', color='black',
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
            )

    # Last image spans both columns
    ax5 = fig.add_subplot(gs[2, :])
    ax5.imshow(img_data[4])
    ax5.axis('off')
    ax5.text(
        0.03, 0.88, labels[4], transform=ax5.transAxes,
        fontsize=20, fontweight='bold', color='black',
        ha='left', va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
    )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.margins(0)
    plt.savefig("Figure2_Combined.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    print("Analysis complete! All figures generated.")

# Run the main function
if __name__ == "__main__":
    main()