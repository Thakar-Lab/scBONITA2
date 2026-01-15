import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import pickle
from sklearn.cluster import AgglomerativeClustering
from matplotlib.patches import Rectangle, FancyBboxPatch, PathPatch
import time
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from matplotlib.sankey import Sankey
from matplotlib.path import Path
from collections import Counter
from matplotlib.path import Path
import csv
# -------------------- Utility Functions --------------------

def compute_state_and_cell_transition_matrices(cleaned_trajectories, state_dict, trajectories):
    """Compute state and cell transition matrices from trajectories."""
    unique_states = [state_dict[key] for key in sorted(state_dict.keys(), key=lambda x: int(x.split()[1]))]
    num_states = len(unique_states)
    num_cells = len(cleaned_trajectories)
    state_transition_matrix = np.zeros((num_states, num_states))
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
    
    # VECTORIZED cell transition matrix calculation (MUCH faster!)
    cell_transition_matrix = state_occupancy @ state_transition_matrix @ state_occupancy.T
    np.fill_diagonal(cell_transition_matrix, 0)  # Set diagonal to 0
    
    # Normalize rows
    row_sums_cell = cell_transition_matrix.sum(axis=1, keepdims=True)
    non_zero_rows_cell = row_sums_cell.flatten() != 0
    if non_zero_rows_cell.any():
        cell_transition_matrix[non_zero_rows_cell] /= row_sums_cell[non_zero_rows_cell]
    
    return state_transition_matrix, state_occupancy, cell_transition_matrix, weighted_transition_matrix


def plot_binary_connectivity_only(cleaned_trajectories, cell_labels, cell_transition_matrix, output_file='Cell_Transition_Matrix.png',
                                   threshold=0.05, max_cells=500):
    """
    Save only the binary connectivity plot from cell transition matrix.
    Binary visualization: Black = Connection exists, White = No connection
    
    Args:
        cell_transition_matrix: Cell-to-cell transition probability matrix
        output_file: Output filename (default: 'Cell_Transition_Matrix.png')
        threshold: Minimum transition probability to show
        max_cells: Maximum cells to display
    """
    print(f"\nCreating binary connectivity visualization...")
    
    num_cells = cell_transition_matrix.shape[0]
    
    # Threshold the matrix
    binary_matrix = (cell_transition_matrix > threshold).astype(int)
    
    # Calculate connectivity for each cell
    out_degree = binary_matrix.sum(axis=1)
    in_degree = binary_matrix.sum(axis=0)
    total_degree = out_degree + in_degree
    
    print(f"   Threshold: {threshold}")
    print(f"   Total connections above threshold: {binary_matrix.sum():,}")
    print(f"   Cells with outgoing connections: {(out_degree > 0).sum()}")
    print(f"   Cells with incoming connections: {(in_degree > 0).sum()}")
    
    # Select most connected cells (for visualization)
    if num_cells > max_cells:
        print(f"   Selecting top {max_cells} most connected cells for visualization...")
        top_indices = np.argsort(total_degree)[-max_cells:]
    else:
        top_indices = np.arange(num_cells)
        max_cells = num_cells
    
    # Get submatrix
    plot_matrix = binary_matrix[np.ix_(top_indices, top_indices)]
    
    # Create single figure for binary connectivity
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Binary connectivity plot 
    im = ax.imshow(plot_matrix, cmap='binary', aspect='auto', interpolation='nearest')
    
    ax.set_xlabel('To Cell', fontsize=12, fontweight='bold')
    ax.set_ylabel('From Cell', fontsize=12, fontweight='bold')
    
    # Add legend instead of colorbar
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='black', label='Connection Present'),
        Rectangle((0, 0), 1, 1, fc='white', ec='black', label='No Connection')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             framealpha=0.9, edgecolor='black')
    
    # Statistics text
    stats_text = (f"Total cells: {num_cells:,}\n"
                  f"Shown: {max_cells}\n"
                  f"Connections: {plot_matrix.sum()}\n"
                  f"Density: {plot_matrix.sum()/(max_cells**2)*100:.2f}%")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    #==============Add this to explain one of the figures and debug==================================
    if num_cells > max_cells:
        print(f"   Selecting top {max_cells} most connected cells for visualization...")
        top_indices = np.argsort(total_degree)[-max_cells:]
    else:
        top_indices = np.arange(num_cells)
        max_cells = num_cells

    # Get submatrix
    plot_matrix = binary_matrix[np.ix_(top_indices, top_indices)]

    # --- Find most outgoing and most incoming cells ---
    most_out_idx = plot_matrix.sum(axis=1).argmax()  # row with max sum
    most_out_count = binary_matrix[most_out_idx,:].sum()
    most_in_idx  = plot_matrix.sum(axis=0).argmax()  # column with max sum
    most_in_count = binary_matrix[:,most_in_idx].sum()

    # --- Print the stat ---
    print(f"Most out index : {most_out_idx:,}\n")
    print(f"Most out count : {most_out_count:,}\n")
    print(f"Most in index : {most_in_idx:,}\n")
    print(f"Most in count : {most_in_count:,}\n")

    # # --- Plot ---
    # fig, ax = plt.subplots(figsize=(8, 6))
    # im = ax.imshow(plot_matrix, cmap='binary', aspect='auto', interpolation='nearest')

    # ax.set_xlabel('To Cell', fontsize=12, fontweight='bold')
    # ax.set_ylabel('From Cell', fontsize=12, fontweight='bold')

    # # Add legend instead of colorbar
    # legend_elements = [
    #     Rectangle((0, 0), 1, 1, fc='black', label='Connection Present'),
    #     Rectangle((0, 0), 1, 1, fc='white', ec='black', label='No Connection')
    # ]
    # ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
    #         framealpha=0.9, edgecolor='black')

    # # Statistics text
    # stats_text = (f"Total cells: {num_cells:,}\n"
    #             f"Shown: {max_cells}\n"
    #             f"Connections: {plot_matrix.sum()}\n"
    #             f"Density: {plot_matrix.sum()/(max_cells**2)*100:.2f}%")
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
    #         fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # # --- Draw rectangles ---
    # # Full row for most outgoing
    # rect_row = Rectangle((-0.5, most_out_idx-0.5), plot_matrix.shape[1], 1, 
    #                     linewidth=2, edgecolor='blue', facecolor='none', label='Most Outgoing')
    # ax.add_patch(rect_row)

    # # Full column for most incoming
    # rect_col = Rectangle((most_in_idx-0.5, -0.5), 1, plot_matrix.shape[0], 
    #                     linewidth=2, edgecolor='red', facecolor='none', label='Most Incoming')
    # ax.add_patch(rect_col)

    # # Optional: add labels for clarity
    # ax.text(plot_matrix.shape[1]-1, most_out_idx, "Most Out", color='blue', 
    #         fontsize=10, verticalalignment='center', horizontalalignment='right', fontweight='bold')
    # ax.text(most_in_idx, plot_matrix.shape[0]-1, "Most In", color='red', 
    #         fontsize=10, verticalalignment='bottom', horizontalalignment='center', fontweight='bold')
    
    # # Save the data
    # summary_stats = {
    #     "most_out_cell" : most_out_idx,
    # }
    # plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # print(f"   Saved binary connectivity plot to {output_file}")
    # plt.show()
    # plt.close()
    
    return fig


def plot_binary_connectivity_with_regions(cell_transition_matrix, cell_labels, cell_condition_dict,
                                          output_file='Cell_Transition_Matrix_Region.png',
                                          threshold=0.05, max_cells=500):
    """
    Save binary connectivity plot with AS-, Mixed, AS+ region annotations.
    Cells are sorted by condition and regions are marked with dashed lines.
    
    Args:
        cell_transition_matrix: Cell-to-cell transition probability matrix
        cell_labels: List of cell labels (e.g., 'cell_1_trajectory')
        cell_condition_dict: Dictionary mapping cell index to condition ('AS+', 'AS-')
        output_file: Output filename
        threshold: Minimum transition probability to show
        max_cells: Maximum cells to display
    """
    print(f"\nCreating binary connectivity visualization with regions...")
    
    num_cells = cell_transition_matrix.shape[0]
    
    # Get condition for each cell from metadata
    cell_conditions = []
    for cell in cell_labels:
        try:
            cell_idx = int(cell.split('_')[1])
            condition = cell_condition_dict.get(cell_idx, 'Mixed')
            cell_conditions.append(condition)
        except:
            cell_conditions.append('Mixed')
    
    # Categorize cells based on metadata labels
    as_minus_cells = [i for i, c in enumerate(cell_conditions) if c == 'AS-']
    as_plus_cells = [i for i, c in enumerate(cell_conditions) if c == 'AS+']
    mixed_cells = [i for i, c in enumerate(cell_conditions) if c not in ['AS-', 'AS+']]
    
    print(f"   Using metadata labels:")
    print(f"   AS-: {len(as_minus_cells)} cells")
    print(f"   Mixed: {len(mixed_cells)} cells")
    print(f"   AS+: {len(as_plus_cells)} cells")
    
    # Sort cells: AS- first, then Mixed, then AS+
    sorted_indices = as_minus_cells + mixed_cells + as_plus_cells
    
    # Limit to max_cells if needed
    if len(sorted_indices) > max_cells:
        # Proportionally sample from each group
        total = len(sorted_indices)
        n_as_minus = min(len(as_minus_cells), int(max_cells * len(as_minus_cells) / total))
        n_mixed = min(len(mixed_cells), int(max_cells * len(mixed_cells) / total))
        n_as_plus = min(len(as_plus_cells), int(max_cells * len(as_plus_cells) / total))
        
        # Adjust to reach max_cells
        remaining = max_cells - n_as_minus - n_mixed - n_as_plus
        if remaining > 0:
            if len(as_plus_cells) > n_as_plus:
                n_as_plus += remaining
            elif len(mixed_cells) > n_mixed:
                n_mixed += remaining
            else:
                n_as_minus += remaining
        
        sorted_indices = as_minus_cells[:n_as_minus] + mixed_cells[:n_mixed] + as_plus_cells[:n_as_plus]
        n_as_minus_plot = n_as_minus
        n_mixed_plot = n_mixed
        n_as_plus_plot = n_as_plus
    else:
        n_as_minus_plot = len(as_minus_cells)
        n_mixed_plot = len(mixed_cells)
        n_as_plus_plot = len(as_plus_cells)
    
    # Threshold the matrix
    binary_matrix = (cell_transition_matrix > threshold).astype(int)
    
    # Get submatrix with sorted indices
    plot_matrix = binary_matrix[np.ix_(sorted_indices, sorted_indices)]
    
    # Calculate statistics
    num_shown = len(sorted_indices)
    
    print(f"   Threshold: {threshold}")
    print(f"   Total cells: {num_cells}")
    print(f"   Shown: {num_shown}")
    print(f"   AS-: {n_as_minus_plot}, Mixed: {n_mixed_plot}, AS+: {n_as_plus_plot}")
    print(f"   Connections: {plot_matrix.sum()}")
    print(f"   Density: {plot_matrix.sum()/(num_shown**2)*100:.2f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Binary connectivity plot
    im = ax.imshow(plot_matrix, cmap='binary', aspect='auto', interpolation='nearest')
    
    ax.set_xlabel('To Cell', fontsize=12, fontweight='bold')
    ax.set_ylabel('From Cell', fontsize=12, fontweight='bold')
    
    # Draw region boundary lines
    boundary1 = n_as_minus_plot  # End of AS- region
    boundary2 = n_as_minus_plot + n_mixed_plot  # End of Mixed region
    
    # Vertical dashed lines
    if boundary1 > 0:
        ax.axvline(x=boundary1 - 0.5, color='gray', linestyle='--', linewidth=1.5)
    if boundary2 > 0 and boundary2 < num_shown:
        ax.axvline(x=boundary2 - 0.5, color='gray', linestyle='--', linewidth=1.5)
    
    # Horizontal dashed lines
    if boundary1 > 0:
        ax.axhline(y=boundary1 - 0.5, color='gray', linestyle='--', linewidth=1.5)
    if boundary2 > 0 and boundary2 < num_shown:
        ax.axhline(y=boundary2 - 0.5, color='gray', linestyle='--', linewidth=1.5)
    
    # Add region labels on y-axis (left side)
    if n_as_minus_plot > 0:
        ax.text(-0.02, (0 + boundary1) / 2 / num_shown, 'AS-', transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='right', va='center')
    if n_mixed_plot > 0:
        ax.text(-0.02, (boundary1 + boundary2) / 2 / num_shown, 'Mixed', transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='right', va='center')
    if n_as_plus_plot > 0:
        ax.text(-0.02, (boundary2 + num_shown) / 2 / num_shown, 'AS+', transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='right', va='center')
    
    # Add region labels on x-axis (bottom)
    if n_as_minus_plot > 0:
        mid_as_minus = boundary1 / 2
        ax.text(mid_as_minus, -0.08 * num_shown, 'AS-', fontsize=12, fontweight='bold', ha='center', va='top')
    if n_mixed_plot > 0:
        mid_mixed = (boundary1 + boundary2) / 2
        ax.text(mid_mixed, -0.08 * num_shown, 'Mixed', fontsize=12, fontweight='bold', ha='center', va='top')
    if n_as_plus_plot > 0:
        mid_as_plus = (boundary2 + num_shown) / 2
        ax.text(mid_as_plus, -0.08 * num_shown, 'AS+', fontsize=12, fontweight='bold', ha='center', va='top')
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='black', label='Connection Present'),
        Rectangle((0, 0), 1, 1, fc='white', ec='black', label='No Connection')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             framealpha=0.9, edgecolor='black')
    
    # Statistics text
    stats_text = (f"Total cells: {num_cells:,}\n"
                  f"Shown: {num_shown}\n"
                  f"Connections: {plot_matrix.sum()}\n"
                  f"Density: {plot_matrix.sum()/(num_shown**2)*100:.2f}%\n"
                  f"\n"
                  f"AS-: {n_as_minus_plot}\n"
                  f"Mixed: {n_mixed_plot}\n"
                  f"AS+: {n_as_plus_plot}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    #plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved region-annotated connectivity plot to {output_file}")
    plt.close()
    
    return fig

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

def compute_weighted_pseudotime(trajectories, attractor_value_dict):
    pseudotime_dict = {}
    observed_steps_dict = {}
    
    for cell, trajectory in trajectories.items():
        cleaned_trajectory = clean_trajectory(trajectory)
        single_attractor, cyclic_attractor, attractor_state, initial_state = find_attractors(cleaned_trajectory)
        attractor = single_attractor if single_attractor is not None else cyclic_attractor[0]
        observed_steps = cleaned_trajectory.index(attractor) + 1
        observed_steps_dict[cell] = observed_steps
    
    max_observed_steps = max(observed_steps_dict.values())
    min_observed_steps = min(observed_steps_dict.values())
    
    for cell, observed_steps in observed_steps_dict.items():
        if max_observed_steps != min_observed_steps:
            pseudotime = 1 - (np.abs(observed_steps - min_observed_steps) / (max_observed_steps - min_observed_steps))
        else:
            pseudotime = 0
        pseudotime_dict[cell] = pseudotime
    
    return pseudotime_dict

def find_most_traveled_paths(basins, trajectories, output_file="most_traveled_paths.txt"):
    most_traveled_paths = {}
    for attractor, cells in basins.items():
        path_counts = {}
        for cell in cells:
            traj = trajectories[cell]
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

# -------------------- Input Files --------------------
INPUT_FILES = {
    "trajectories": "Combined_Trajectory.txt",
    "unique_states": "Combined_Unique_States.txt",
    "metadata": r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\input\tutorial_data\combined_metadata.txt",
    "metadata_extra": r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\input\tutorial_data\HIV_metadata_extra.txt",
    "knockout_results": r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\scBONITA_output\importance_score_output\tutorial_dataset\intermediate_files\hsa05417\knockout_results_{gene}.pkl",
    "knockin_results": r"C:\Users\plwin\MayTrajectory_MDataRuns_MultipleTrajectoryRuns\T_cells_all_July_2025\scBONITA_output\importance_score_output\tutorial_dataset\intermediate_files\hsa05417\knockin_results_{gene}.pkl",
}

# -------------------- Main Function --------------------
def generate_all_figures3():
    print("\n" + "="*60)
    print("Generating Figure 3 : State Transition and Cell Transition Matrices with Clusters")
    print("="*60)
    
    
    # Load trajectories and clean them
    trajectories = read_trajectories(INPUT_FILES["trajectories"])
    cleaned_trajectories = {key: clean_trajectory(value) for key, value in trajectories.items()}
    
    # Read unique states
    with open(INPUT_FILES["unique_states"], 'r') as f:
        columns_dict = {
            key.strip(): list(map(int, value.strip().strip('[]').replace('(', '').replace(')', '').split(','))) 
            for key, value in (line.strip().split(':') for line in f)
        }
    
    unique_states = list(columns_dict.values())
    state_labels = list(columns_dict.keys())
    
    # Build attractor information and basins
    attractors = {}
    basins = {}
    attractor_value_dict = {}
    
    for cell, trajectory in trajectories.items():
        cleaned_trajectory = clean_trajectory(trajectory)
        single_attractor, cyclic_attractor, attractor_state, _ = find_attractors(cleaned_trajectory)
        attractor = single_attractor if single_attractor is not None else cyclic_attractor[0]
        
        if attractor not in attractors:
            attractors[attractor] = []
            basins[attractor] = []
        attractors[attractor].append(cell)
        basins[attractor].append(cell)
        
        formatted_attractor_state = f"State {attractor_state}"
        if formatted_attractor_state in columns_dict:
            attractor_value = columns_dict[formatted_attractor_state]
            attractor_value_dict[int(cell.split('_')[1])] = attractor_value
        else:
            attractor_value_dict[int(cell.split('_')[1])] = attractor_state
    
    # Calculate normalized steps to attractor
    normalized_steps_to_attractor = {
        state: ((len(clean_trajectory(trajectory)) - clean_trajectory(trajectory).index(state)) - 1) / len(clean_trajectory(trajectory))
        for trajectory in cleaned_trajectories.values()
        for state in clean_trajectory(trajectory)
        if state in clean_trajectory(trajectory) and len(clean_trajectory(trajectory)) > 1
    }
    
    # Prepare data for t-SNE
    state_vector_with_pseudotime = np.array([
        np.array(unique_states[i]) * (1 - normalized_steps_to_attractor.get(i, 0))
        for i in range(len(unique_states))
    ])
    
    # Perform t-SNE
    tsne_results = TSNE(n_components=2, random_state=24, perplexity=150).fit_transform(state_vector_with_pseudotime)
    
    # ==================== FIGURE 3A ====================
    indices_pseudotime1 = [
        i for i, state in enumerate(state_labels) 
        if normalized_steps_to_attractor.get(int(state.split()[1]), 0) == 0
    ]
    
    attractor_coordinates = {}
    
    # Create ScalarMappable for colorbar (used in multiple figures)
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    sm.set_array([])
    
    if indices_pseudotime1:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        
        for i, state in enumerate(state_labels):
            state_index = int(state.split()[1])
            color = plt.cm.coolwarm(1 - normalized_steps_to_attractor.get(state_index, 0))
            plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=color, alpha=0.3, marker='s', s=120)
        
        for i in indices_pseudotime1:
            state_index = int(state_labels[i].split()[1])
            plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color='gold', edgecolor='black', 
                   s=120, marker='s', linewidths=2)
        
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Normalized Pseudotime", fontsize=18, fontweight='bold')
        cbar.ax.tick_params(labelsize=18)
        cbar.outline.set_linewidth(2)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.xlabel("t-SNE Component 1", fontsize=18, fontweight='bold')
        plt.ylabel("t-SNE Component 2", fontsize=18, fontweight='bold')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        #plt.tight_layout()
        plt.savefig('Figure3A.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        for i in indices_pseudotime1:
            state_index = int(state_labels[i].split()[1])
            x, y = tsne_results[i, 0], tsne_results[i, 1]
            attractor_coordinates[state_index] = (x, y)
    
    # ==================== FIGURE 3B ====================
    cell_tsne_coords = np.zeros((len(trajectories), 2))
    cell_labels = list(trajectories.keys())
    
    for idx, cell in enumerate(cell_labels):
        cleaned_traj = clean_trajectory(trajectories[cell])
        attractor_state = find_attractors(cleaned_traj)[0]
        if attractor_state is None:
            attractor_state = find_attractors(cleaned_traj)[1][0]
        attractor_coord = attractor_coordinates.get(attractor_state, (0, 0))
        cell_tsne_coords[idx] = attractor_coord
    
    jitter_strength = 1.0
    rng = np.random.default_rng(seed=42)
    jitter = rng.normal(loc=0, scale=jitter_strength, size=cell_tsne_coords.shape)
    cell_tsne_coords_jittered = cell_tsne_coords + jitter
    
    pseudotime_dict = compute_weighted_pseudotime(trajectories, attractor_value_dict)
    cell_pseudotime_values = [pseudotime_dict.get(cell, 0) for cell in cell_labels]
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    scatter = plt.scatter(
        cell_tsne_coords_jittered[:, 0], cell_tsne_coords_jittered[:, 1],
        c=cell_pseudotime_values, cmap='plasma', alpha=0.7, edgecolor='white', s=120, linewidths=2
    )
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("1-Normalized Steps to Attractors", fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_linewidth(2)
    
    plt.xlabel("t-SNE Component 1", fontsize=18, fontweight='bold')
    plt.ylabel("t-SNE Component 2", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.tight_layout()
    plt.savefig('Figure3B.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== FIGURE 3C Spectral Clustering ====================
    sorted_attractors = sorted(basins.keys(), key=lambda x: len(basins[x]), reverse=True)
    top_attractors = sorted_attractors[:5]
    
    # Using dissimilarity matrix from pickle file as input X
    n_clusters = 5

    # Load precomputed dissimilarity matrix from pickle file
    print("Loading dissimilarity matrix from cell_dissimilarity_matrix.pkl...")
    with open('cell_dissimilarity_matrix.pkl', 'rb') as f:
        dissimilarity_matrix = pickle.load(f)
    print(f"Dissimilarity matrix shape: {dissimilarity_matrix.shape}")
    
    # Convert dissimilarity to similarity (affinity) matrix for Spectral Clustering
    # Using Gaussian kernel: affinity = exp(-dissimilarity^2 / (2 * sigma^2))
    sigma = np.median(dissimilarity_matrix[dissimilarity_matrix > 0])  # Use median as sigma
    affinity_matrix = np.exp(-dissimilarity_matrix**2 / (2 * sigma**2))
    
    # Also convert to similarity score matrix (1 - dissimilarity for normalized dissimilarity)
    similarity_matrix = 1 - (dissimilarity_matrix / np.max(dissimilarity_matrix))
    
    # Keep the t-SNE coordinates for visualization
    X_coords = cell_tsne_coords_jittered
    
    # Use Spectral clustering on affinity matrix
    print("Running Spectral clustering on affinity matrix...")
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42, n_init=10)
    cluster_labels_spectral = spectral.fit_predict(affinity_matrix)
    print(f"Spectral cluster labels shape: {cluster_labels_spectral.shape}")
    
    # Also run K-means on similarity matrix for comparison
    print("Running K-means clustering on similarity matrix...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels_kmeans = kmeans.fit_predict(similarity_matrix)
    print(f"K-means cluster labels shape: {cluster_labels_kmeans.shape}")
    
    # Calculate centroids for Spectral clustering (using t-SNE coordinates)
    spectral_centroids = np.zeros((n_clusters, 2))
    spectral_cluster_counts = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels_spectral == i
        cluster_points = X_coords[cluster_mask]
        spectral_cluster_counts.append(np.sum(cluster_mask))
        if len(cluster_points) > 0:
            spectral_centroids[i] = cluster_points.mean(axis=0)
    print(f"Spectral cluster counts: {spectral_cluster_counts}")
    print(f"Spectral centroids:\n{spectral_centroids}")
    
    # Calculate centroids for K-means clustering (using t-SNE coordinates)
    kmeans_centroids = np.zeros((n_clusters, 2))
    kmeans_cluster_counts = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels_kmeans == i
        cluster_points = X_coords[cluster_mask]
        kmeans_cluster_counts.append(np.sum(cluster_mask))
        if len(cluster_points) > 0:
            kmeans_centroids[i] = cluster_points.mean(axis=0)
    print(f"K-means cluster counts: {kmeans_cluster_counts}")
    print(f"K-means centroids:\n{kmeans_centroids}")

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Plot all cells colored by Spectral cluster (using t-SNE coordinates for visualization)
    cluster_colors = ListedColormap(['red', 'green', 'blue', 'orange', 'purple'])
    scatter = plt.scatter(
        X_coords[:, 0], X_coords[:, 1],
        c=cluster_labels_spectral, cmap=cluster_colors, alpha=0.7, edgecolor='white', s=120, linewidths=2
    )

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Draw convex hulls for each attractor (not clusters)
    hull_colors = ['r', 'g', 'b', 'm', 'c']
    legend_elements = []
    for i, attractor in enumerate(top_attractors):
        indices = [idx for idx, cell in enumerate(cell_labels)
               if find_attractors(clean_trajectory(trajectories[cell]))[0] == attractor]
        if len(indices) >= 3:
            points = X_coords[indices]
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                plt.plot(
                    np.append(hull_points[:, 0], hull_points[0, 0]),
                    np.append(hull_points[:, 1], hull_points[0, 1]),
                    color=hull_colors[i % len(hull_colors)], linestyle='--', lw=3,
                )
                legend_elements.append(Line2D([0], [0], color=hull_colors[i % len(hull_colors)], lw=3, linestyle='--',
                                label=f'Attractor {i+1}'))
            except:
                pass

    # Mark attractor points (gold squares)
    for attractor in top_attractors:
        attractor_coord = attractor_coordinates.get(attractor, None)
        if attractor_coord is not None:
            plt.scatter(attractor_coord[0], attractor_coord[1], color='gold', edgecolor='black', s=180, marker='s')
    
    # Plot Spectral cluster centroids (large X markers with cluster colors)
    centroid_colors = ['red', 'green', 'blue', 'orange', 'purple']
    print(f"Plotting {n_clusters} spectral centroids...")
    for i in range(n_clusters):
        if spectral_cluster_counts[i] > 0:  # Only plot if cluster has points
            print(f"  Centroid {i}: ({spectral_centroids[i, 0]:.2f}, {spectral_centroids[i, 1]:.2f})")
            plt.scatter(spectral_centroids[i, 0], spectral_centroids[i, 1], 
                       color=centroid_colors[i], edgecolor='black', s=400, marker='X', linewidths=3, zorder=10)
    
    # Add centroid legend element
    legend_elements.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='black', 
                                  markeredgecolor='black', markersize=15, label='Spectral Centroid'))

    # Colorbar for clusters
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_clusters))
    cbar.set_label('Cluster Label (Spectral)', fontsize=16, fontweight='bold')
    cbar.ax.set_yticklabels([f'C{i}' for i in range(n_clusters)])
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(2)
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='lower right', fontsize=14)

    plt.xlabel("t-SNE Component 1", fontsize=18, fontweight='bold')
    plt.ylabel("t-SNE Component 2", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Set xticks and yticks based on t-SNE range
    x_min, x_max = X_coords[:, 0].min(), X_coords[:, 0].max()
    y_min, y_max = X_coords[:, 1].min(), X_coords[:, 1].max()
    x_tick_max = int(np.ceil(x_max / 10.0) * 10)
    x_tick_min = int(np.floor(x_min / 10.0) * 10)
    y_tick_max = int(np.ceil(y_max / 10.0) * 10)
    y_tick_min = int(np.floor(y_min / 10.0) * 10)
    plt.xticks(np.arange(x_tick_min, x_tick_max + 1, 10))
    plt.yticks(np.arange(y_tick_min, y_tick_max + 1, 10))
    
    #plt.tight_layout()
    plt.savefig('Figure3C.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save cluster assignments for Spectral clustering
    with open("Cluster_Assignments_Spectral.txt", "w") as f:
        f.write("Cell_ID\tCluster_Label\n")
        for idx, cell in enumerate(cell_labels):
            f.write(f"{cell}\tCluster_{cluster_labels_spectral[idx]}\n")
    
    # Save cluster assignments for K-means clustering
    with open("Cluster_Assignments_KMeans.txt", "w") as f:
        f.write("Cell_ID\tCluster_Label\n")
        for idx, cell in enumerate(cell_labels):
            f.write(f"{cell}\tCluster_{cluster_labels_kmeans[idx]}\n")
    
    # Save centroids for both methods
    with open("Cluster_Centroids.txt", "w") as f:
        f.write("Method\tCluster\tCentroid_X\tCentroid_Y\n")
        for i in range(n_clusters):
            f.write(f"Spectral\t{i}\t{spectral_centroids[i, 0]:.4f}\t{spectral_centroids[i, 1]:.4f}\n")
        for i in range(n_clusters):
            f.write(f"KMeans\t{i}\t{kmeans_centroids[i, 0]:.4f}\t{kmeans_centroids[i, 1]:.4f}\n")
    
    # Save attractor assignments for each cell
    with open("Attractor_Assignments_AllCells.txt", "w") as f:
        for idx, cell in enumerate(cell_labels):
            attractor_state = find_attractors(clean_trajectory(trajectories[cell]))[0]
            f.write(f"{cell}\tAttractor_{attractor_state}\n")

    # ==================== Cell Transition Matrix ====================
    # Print cleaned trajectories first to debug
    print("\n=== Cleaned Trajectories (first 10) ===")
    for i, (cell, traj) in enumerate(cleaned_trajectories.items()):
        if i >= 10:  # Only print first 10
            break
        print(f"{cell}: {traj}")

    print(f"\nTotal cleaned trajectories: {len(cleaned_trajectories)}")
    print("Computing cell transition matrix...")
    state_transition_matrix, state_occupancy, cell_transition_matrix, weighted_transition_matrix = \
        compute_state_and_cell_transition_matrices(cleaned_trajectories, columns_dict, trajectories)
    
    # Save the cell transition matrix for future use
    np.save('state_transition_matrix.npy', state_transition_matrix)
    np.save('state_occpancy_matrix.npy', state_occupancy)
    np.save('weighted_transition_matrix.npy', weighted_transition_matrix)
    np.save('cell_transition_matrix.npy', cell_transition_matrix)
    print(f"Saved cell_transition_matrix.npy with shape: {cell_transition_matrix.shape}")

    # ==================== DEBUG: Cell Transition Matrix Calculation ====================
    print("\n=== Debugging Cell Transition Matrix Calculation ===")

    # First, let's understand what's in the occupancy matrix
    print("\n1. State Occupancy Matrix Analysis:")
    print(f"   Shape: {state_occupancy.shape}")  # Should be (num_cells, num_states)
    print(f"   Total entries: {state_occupancy.sum():.0f}")
    print(f"   Non-zero entries: {np.count_nonzero(state_occupancy)}")

    # Which states are most frequently occupied?
    state_visit_counts = state_occupancy.sum(axis=0)  # Sum over cells
    top_visited_states = np.argsort(state_visit_counts)[::-1][:10]

    print("\n2. Top 10 Most Visited States:")
    for i, state_idx in enumerate(top_visited_states):
        num_cells_visiting = np.count_nonzero(state_occupancy[:, state_idx])
        total_visits = state_visit_counts[state_idx]
        print(f"   {i+1}. State {state_idx}: visited by {num_cells_visiting} cells, total visits = {total_visits:.0f}")
    
    # Generate the binary connectivity plot
    plot_binary_connectivity_only(
        cleaned_trajectories, 
        cell_labels,
        cell_transition_matrix,
        output_file='Cell_Transition_Matrix.png',
        threshold=0.05,
        max_cells=500
    )
    # Save the top transitions 
    # Get the top 500 transitions
    flat_matrix = cell_transition_matrix.flatten()
    top_500_indices = np.argsort(flat_matrix)[::-1][:500]

    # Convert flat indices to (row, col) pairs
    n_cells = cell_transition_matrix.shape[0]
    top_500_transitions = []
    attractor_counts = {}

    for flat_idx in top_500_indices:
        row, col = np.unravel_index(flat_idx, cell_transition_matrix.shape)
        if cell_transition_matrix[row, col] > 0:
            cell_from = cell_labels[row]
            cell_to = cell_labels[col]
            
            # Find attractors for both cells
            traj_from = clean_trajectory(trajectories[cell_from])
            traj_to = clean_trajectory(trajectories[cell_to])
            
            att_from = find_attractors(traj_from)[0] or find_attractors(traj_from)[1][0]
            att_to = find_attractors(traj_to)[0] or find_attractors(traj_to)[1][0]
            
            # Count this attractor pair
            att_pair = (att_from, att_to)
            attractor_counts[att_pair] = attractor_counts.get(att_pair, 0) + 1
            
            top_500_transitions.append({
                'from': cell_from,
                'to': cell_to,
                'prob': cell_transition_matrix[row, col],
                'att_from': att_from,
                'att_to': att_to
            })

    # Analyze the results
    print(f"\n=== Top 500 Transition Attractor Distribution ===")
    print(f"Total transition pairs found: {len(top_500_transitions)}")

    # Count by source attractor
    from_attractor_freq = {}
    for trans in top_500_transitions:
        att = trans['att_from']
        from_attractor_freq[att] = from_attractor_freq.get(att, 0) + 1

    print("\nFrequency by source attractor:")
    for att, count in sorted(from_attractor_freq.items(), key=lambda x: x[1], reverse=True):
        print(f"  Attractor {att}: {count} transitions ({100*count/len(top_500_transitions):.1f}%)")

    # Save to file
    # Save detailed transitions to file (one row per transition)
    with open('top_500_transitions_by_attractor.txt', 'w') as f:
        f.write("Source_Cell\tTarget_Cell\tSource_Attractor\tTarget_Attractor\tTransition_Probability\n")
        for trans in top_500_transitions:
            f.write(f"{trans['from']}\t{trans['to']}\t{trans['att_from']}\t{trans['att_to']}\t{trans['prob']:.6f}\n")
    print("\nSaved: top_500_transitions_by_attractor.txt")

    # ==================== FIGURE 3E and 3F ====================
    # Calculate condition bias for each state
    metadata_file = INPUT_FILES["metadata"]
    metadata = pd.read_csv(metadata_file, sep=" ", header=None, skiprows=0)
    
    cell_sample_indices = []
    participant = []
    conditions = []
    
    for _, row in metadata.iterrows():
        cell_sample_indices.append(int(str(row[0]).strip('"')) - 1)
        participant.append(row[1])
        conditions.append(row[2])
    
    cell_condition_dict = {cell_sample_indices[i]: conditions[i] for i in range(len(cell_sample_indices))}
    
    # Generate the binary connectivity plot (simple version without regions)
    plot_binary_connectivity_only(
        cleaned_trajectories,
        cell_labels,
        cell_transition_matrix,
        output_file='Cell_Transition_Matrix_Region.png',
        threshold=0.05,
        max_cells=500
    )
    
    # Create state_pseudotime_dict for sorting
    state_pseudotime_dict = {state: 1 - normalized_steps_to_attractor.get(state, 0) for state in normalized_steps_to_attractor}
    
    # Sort states by pseudotime (exactly like original code)
    state_pseudotime_items = [
        (state_index, state_pseudotime_dict.get(state_index, 0), unique_states[i])
        for i, state_index in enumerate([int(s.split()[1]) for s in state_labels])
    ]
    state_pseudotime_items.sort(key=lambda x: x[1])
    num_states = len(state_pseudotime_items)
    print(f"Number of unique states: {num_states}")

    sorted_indices = [x[0] for x in state_pseudotime_items]
    sorted_pseudotimes = [x[1] for x in state_pseudotime_items]
    sorted_state_vectors = [x[2] for x in state_pseudotime_items]
    
    # Calculate condition bias for each state (using sorted_indices like original)
    condition_bias_array = np.zeros(num_states)
    y_values = np.zeros(num_states)
    
    for i in range(num_states):
        state_index = sorted_indices[i]
        cell_indices = [cell for cell, traj in cleaned_trajectories.items() if state_index in traj]
        cleaned_cell_indices = []
        for cell in cell_indices:
            try:
                idx = int(cell.split('_')[1])
                cleaned_cell_indices.append(idx)
            except (IndexError, ValueError):
                continue
        state_to_cells = {}
        state_to_cells[state_index] = cleaned_cell_indices
        healthy_cells = 0
        diseased_cells = 0
        total_cells = 0
        for state_index_inner, cleaned_cell_indices_inner in state_to_cells.items():
            for cell in cleaned_cell_indices_inner:
                if cell in cell_condition_dict:
                    condition = cell_condition_dict[cell]
                    if condition == "AS-":
                        healthy_cells += 1
                    elif condition == "AS+":
                        diseased_cells += 1
                    total_cells += 1
            if total_cells > 0:
                healthy_percentage = healthy_cells / total_cells
                diseased_percentage = diseased_cells / total_cells
                condition_bias = healthy_percentage - diseased_percentage
                condition_bias_array[i] = condition_bias
            else:
                condition_bias = 0
                condition_bias_array[i] = 0
            y_values[i] = condition_bias

    #========= Here to do a cluster analysis====================
    # Before perturbation only
    # Cluster the states based on AS+, AS- and mixed
    # Cluster states into AS+, AS-, and mixed based on condition bias
    as_plus_states = []
    as_minus_states = []
    mixed_states = []

    for i, bias in enumerate(condition_bias_array):
        state_index = sorted_indices[i]
        if bias > 0.5:
            as_minus_states.append(state_index)
        elif bias < -0.5:
            as_plus_states.append(state_index)
        else:
            mixed_states.append(state_index)

    print(f"AS- cluster states: {as_minus_states}")
    print(f"AS+ cluster states: {as_plus_states}")
    print(f"Mixed cluster states: {mixed_states}")

    # Make a different clustering
    # Calculate a distance matrix for t-SNE using both condition bias and pseudotime
    # Distance between states i and j: weighted sum of (condition bias difference)^2 and (pseudotime difference)^2
    condition_bias_norm = (condition_bias_array - np.min(condition_bias_array)) / (np.ptp(condition_bias_array) + 1e-8)
    pseudotime_norm = np.array([1 - normalized_steps_to_attractor.get(int(state_labels[i].split()[1]), 0) for i in range(num_states)])
    pseudotime_norm = (pseudotime_norm - np.min(pseudotime_norm)) / (np.ptp(pseudotime_norm) + 1e-8)

    # You can adjust the weights as needed
    w_cb = 0.8  # Weight for condition bias
    w_pt = 1 - w_cb

    dist_matrix = np.zeros((num_states, num_states))
    for i in range(num_states):
        for j in range(num_states):
            d_cb = np.abs(condition_bias_norm[i] - condition_bias_norm[j])
            d_pt = np.abs(pseudotime_norm[i] - pseudotime_norm[j])
            dist_matrix[i, j] = w_cb * d_cb + w_pt * d_pt

    # Use t-SNE with precomputed distance matrix
    exaggeration_num = 12.0  # Increase exaggeration to spread out the points more
    tsne_results_dist = TSNE(
        n_components=2,
        metric='precomputed',
        random_state=24,
        perplexity=350,
        init='random',
        early_exaggeration=exaggeration_num,
    ).fit_transform(dist_matrix)

    # Calculate standard deviation of condition bias for consistent Mixed definition
    # Mixed = states within 1 standard deviation of zero
    condition_bias_std = np.std(condition_bias_array)
    mixed_threshold = condition_bias_std  # 1 standard deviation from zero
    
    print(f"\nCondition Bias Statistics:")
    print(f"  Mean: {np.mean(condition_bias_array):.3f}")
    print(f"  Std Dev: {condition_bias_std:.3f}")
    print(f"  Mixed threshold: ±{mixed_threshold:.3f} (1 std dev from zero)")
    
    # Find indices of AS+ (bias < -threshold), AS- (bias > +threshold), and mixed (within threshold of zero)
    as_plus_indices = [i for i, val in enumerate(condition_bias_array) if val < -mixed_threshold]
    as_minus_indices = [i for i, val in enumerate(condition_bias_array) if val > mixed_threshold]
    mixed_indices = [i for i, val in enumerate(condition_bias_array) if -mixed_threshold <= val <= mixed_threshold]
    
    print(f"  AS- states (bias > {mixed_threshold:.3f}): {len(as_minus_indices)}")
    print(f"  Mixed states (|bias| <= {mixed_threshold:.3f}): {len(mixed_indices)}")
    print(f"  AS+ states (bias < -{mixed_threshold:.3f}): {len(as_plus_indices)}")
    
    # Save the threshold for Figure 4
    np.save('mixed_threshold.npy', np.array([mixed_threshold]))
    # ==================== FIGURE 3E ====================
    # Plot t-SNE colored by pseudotime with convex hulls AND gold attractor highlights
    indices_pseudotime1 = [i for i, state in enumerate(state_labels) if normalized_steps_to_attractor.get(int(state.split()[1]), 0) == 0]
    
    _, ax_dist = plt.subplots(figsize=(8, 6))
    plt.setp(ax_dist.spines.values(), linewidth=2)
    for i, state in enumerate(state_labels):
        pt = pseudotime_norm[i]
        color = plt.cm.coolwarm(pt)
        ax_dist.scatter(tsne_results_dist[i, 0], tsne_results_dist[i, 1],
                        color=color, alpha=1, label=f"State {state}", marker='s', linewidths=2)
    ax_dist.set_xlabel("t-SNE Component 1", fontsize=18)
    ax_dist.set_ylabel("t-SNE Component 2", fontsize=18)
    ax_dist.tick_params(axis='both', labelsize=18)
    
    # Overlay highlights if there are any states with pseudotime 1
    if indices_pseudotime1:
        for i in indices_pseudotime1:
            state_index = int(state_labels[i].split()[1])
            ax_dist.scatter(
                tsne_results_dist[i, 0], tsne_results_dist[i, 1],
                color='gold', edgecolor='black', s=120, marker='s',
                label=f"State {state_index} (pseudotime 1)", linewidths=2
            )
        print(f"Highlighted {len(indices_pseudotime1)} unique states with pseudotime 1.")
    else:
        print("No unique states found with pseudotime 1.")

    # --- Draw convex hulls for AS+, AS-, and mixed condition ---
    # Convex hull for AS+ (bias = -1)
    if as_plus_indices and len(as_plus_indices) >= 3:
        as_plus_points = np.array([tsne_results_dist[i] for i in as_plus_indices])
        try:
            hull_plus = ConvexHull(as_plus_points)
            hull_vertices = np.append(hull_plus.vertices, hull_plus.vertices[0])
            ax_dist.plot(as_plus_points[hull_vertices, 0], as_plus_points[hull_vertices, 1], 'r--', lw=2, label='AS+ Convex Hull')
            ax_dist.fill(as_plus_points[hull_vertices, 0], as_plus_points[hull_vertices, 1], color='red', alpha=0.15)
            # Annotate AS+ region
            centroid_plus = as_plus_points[hull_plus.vertices].mean(axis=0)
            ax_dist.annotate("AS+", xy=centroid_plus, color='red', fontsize=18, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2, alpha=0.7))
        except:
            pass

    # Convex hull for AS- (bias = +1)
    if as_minus_indices and len(as_minus_indices) >= 3:
        as_minus_points = np.array([tsne_results_dist[i] for i in as_minus_indices])
        try:
            hull_minus = ConvexHull(as_minus_points)
            hull_vertices = np.append(hull_minus.vertices, hull_minus.vertices[0])
            ax_dist.plot(as_minus_points[hull_vertices, 0], as_minus_points[hull_vertices, 1], 'g--', lw=2, label='AS- Convex Hull')
            ax_dist.fill(as_minus_points[hull_vertices, 0], as_minus_points[hull_vertices, 1], color='green', alpha=0.15)
            # Annotate AS- region
            centroid_minus = as_minus_points[hull_minus.vertices].mean(axis=0)
            ax_dist.annotate("AS-", xy=centroid_minus, color='green', fontsize=18, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", lw=2, alpha=0.7))
        except:
            pass

    # Convex hull for mixed condition states
    if mixed_indices and len(mixed_indices) >= 3:
        mixed_points = np.array([tsne_results_dist[i] for i in mixed_indices])
        try:
            hull_mixed = ConvexHull(mixed_points)
            hull_vertices = np.append(hull_mixed.vertices, hull_mixed.vertices[0])
            ax_dist.plot(mixed_points[hull_vertices, 0], mixed_points[hull_vertices, 1], 'b--', lw=2, label='Mixed Convex Hull')
            ax_dist.fill(mixed_points[hull_vertices, 0], mixed_points[hull_vertices, 1], color='blue', alpha=0.10)
            # Annotate Mixed region
            centroid_mixed = mixed_points[hull_mixed.vertices].mean(axis=0)
            ax_dist.annotate("Mixed", xy=centroid_mixed, color='blue', fontsize=18, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=2, alpha=0.7))
        except:
            pass

    sm_dist = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))
    sm_dist.set_array([])
    cbar = plt.colorbar(sm_dist, ax=ax_dist)
    cbar.set_label("Normalized Pseudotime", fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=18)
    # Make colorbar outline bold
    cbar.outline.set_linewidth(2)
    plt.xlabel("t-SNE Component 1", fontsize=18, fontweight='bold')
    plt.ylabel("t-SNE Component 2", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.tight_layout()
    plt.savefig("Figure3EOrg.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save condition_bias_array and sorted_pseudotimes for Figure 4
    np.save('condition_bias_array.npy', condition_bias_array)
    np.save('sorted_pseudotimes.npy', np.array(sorted_pseudotimes))
    print("Saved condition_bias_array.npy and sorted_pseudotimes.npy for Figure 4")

    #==============================================================================================================================
    # Save the x and y coordinates of the highlighted states (attractors)
    attractor_coordinates_dist = {}
    if indices_pseudotime1:
        with open("highlighted_states_coordinates_of_attractors.txt", "w") as f:
            for i in indices_pseudotime1:
                state_index = int(state_labels[i].split()[1])
                x, y = tsne_results_dist[i, 0], tsne_results_dist[i, 1]
                attractor_coordinates_dist[state_index] = (x, y)
                f.write(f"{x}\t{y}\n")
        # Also save as a dictionary for later use
        with open("attractor_coordinates.txt", "w") as f:
            for attractor, coords in attractor_coordinates_dist.items():
                f.write(f"{attractor}: {coords}\n")

    # Redo the t-SNE with the cells clustering at the coordinates of the attractors
    # Assign each cell the t-SNE coordinates of its attractor
    cell_tsne_coords_dist = np.zeros((len(trajectories), 2))
    for idx, cell in enumerate(cell_labels):
        cleaned_traj = clean_trajectory(trajectories[cell])
        attractor_state = find_attractors(cleaned_traj)[0]
        if attractor_state in attractor_coordinates_dist:
            cell_tsne_coords_dist[idx] = attractor_coordinates_dist[attractor_state]
        else:
            print(f"Warning: Attractor state {attractor_state} not found in attractor coordinates.")

    # Add jitter to the coordinates for better visualization
    jitter_strength_3f = 2  # Increased jitter for more spread
    rng_3f = np.random.default_rng(seed=42)
    jitter_3f = rng_3f.normal(loc=0, scale=jitter_strength_3f, size=cell_tsne_coords_dist.shape)
    cell_tsne_coords_jittered_3f = cell_tsne_coords_dist + jitter_3f

    # Remove '_trajectory' from cell labels if present
    cell_names = [cell.replace('_trajectory', '') for cell in cell_labels]

    # Save the cell t-SNE embeddings to a file
    with open('cell_tsne_embeddings.txt', 'w') as f:
        for name, (x, y) in zip(cell_names, cell_tsne_coords_jittered_3f):
            f.write(f"{name}\t{x}\t{y}\n")
    print(f"Number of cells plotted at attractor coordinates (with jitter): {len(cell_labels)}")

    # Get participant IDs from metadata
    cell_ids = [cell.split('_')[1] for cell in cell_labels]
    meta_data_file = INPUT_FILES["metadata"]
    participant_ids = []
    with open(meta_data_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                participant_ids.append(parts[1])
    
    # Make a subset of participant IDs that match the cell names
    participant_ids_subset = [participant_ids[int(cell_id) - 1] for cell_id in cell_ids]
    
    # Append the participant IDs as the first column
    cell_tsne_coords_with_ids = np.column_stack((participant_ids_subset, cell_tsne_coords_jittered_3f))
    df = pd.DataFrame(cell_tsne_coords_with_ids)
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
    cell_tsne_coords_with_ids = df.values

    # Write the t-SNE embeddings to a file
    if os.path.exists('cell_tsne_embeddings.txt'):
        os.remove('cell_tsne_embeddings.txt')
    with open('cell_tsne_embeddings.txt', 'w') as f:
        for row in cell_tsne_coords_with_ids:
            f.write('\t'.join(map(str, row)) + '\n')

    # Calculate cell pseudotime values
    print("Calculating cell pseudotime values based on attractor states...")
    cell_pseudotime_dict_3f = {}
    cell_pseudotime_values_3f = []
    
    pseudotime_dict_3f = compute_weighted_pseudotime(trajectories, attractor_value_dict)
    for idx, cell in enumerate(cell_labels):
        pseudotime_value = pseudotime_dict_3f.get(cell, 0)
        cell_pseudotime_dict_3f[cell] = pseudotime_value
        cell_pseudotime_values_3f.append(pseudotime_value)

    # Save cell pseudotime values to file
    with open("cell_pseudotime_values.txt", "w") as f:
        for idx, cell in enumerate(cell_labels):
            participant_id = participant_ids_subset[idx] if idx < len(participant_ids_subset) else ""
            pseudotime_value = cell_pseudotime_dict_3f.get(cell, 0)
            f.write(f"{participant_id}\t{pseudotime_value}\n")

    # ==================== FIGURE 3F ====================
    # Plot the t-SNE coordinates of the cells with jitter, colored by pseudotime
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    scatter = plt.scatter(
        cell_tsne_coords_with_ids[:, 1].astype(float), cell_tsne_coords_with_ids[:, 2].astype(float),
        c=cell_pseudotime_values_3f, cmap='plasma', alpha=0.7, edgecolor='white', s=120, linewidths=2
    )
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("1-Normalized Steps to Attractors", fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=18)
    # Make colorbar outline bold
    cbar.outline.set_linewidth(2)
    plt.xlabel("t-SNE Component 1", fontsize=18, fontweight='bold')
    plt.ylabel("t-SNE Component 2", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.tight_layout()
    plt.savefig("Figure3FOrg.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All individual figures (3A-3F) generated successfully!")


def generate_figure4():
    """Generate Figure 4A and 4B: Condition Bias analysis plots"""
    
    print("\n" + "="*60)
    print("Generating Figure 4: Condition Bias Analysis")
    print("="*60)
    
    # Load saved data from Figure 3
    if not os.path.exists('condition_bias_array.npy') or not os.path.exists('sorted_pseudotimes.npy'):
        print("Error: Required data files not found. Please run generate_all_figures3() first.")
        return
    
    condition_bias_array = np.load('condition_bias_array.npy')
    sorted_pseudotimes = np.load('sorted_pseudotimes.npy')
    num_states = len(condition_bias_array)
    
    # Load or calculate the mixed threshold (1 standard deviation from zero)
    if os.path.exists('mixed_threshold.npy'):
        mixed_threshold = np.load('mixed_threshold.npy')[0]
    else:
        mixed_threshold = np.std(condition_bias_array)
    
    print(f"Loaded condition_bias_array: {condition_bias_array.shape}")
    print(f"Loaded sorted_pseudotimes: {sorted_pseudotimes.shape}")
    print(f"Mixed threshold (1 std dev): ±{mixed_threshold:.3f}")
    
    # Define consistent colors for Figure 4B: AS- = teal, Mixed = blue, AS+ = red
    color_as_minus = 'teal'
    color_mixed = 'blue'
    color_as_plus = 'red'
    
    # ==================== FIGURE 4A: Condition Bias vs Pseudotime Scatter ====================
    print("\nGenerating Figure 4A: Condition Bias vs Pseudotime Scatter...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color states by pseudotime using coolwarm colormap (blue=early, red=late)
    colors = plt.cm.coolwarm(sorted_pseudotimes)
    
    # Plot each state as a square marker
    for i in range(num_states):
        ax.scatter(sorted_pseudotimes[i], condition_bias_array[i], 
                   c=[colors[i]], marker='s', s=100, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # Add horizontal dashed line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    
    #  AS- States
    arrow_start = 0.6
    arrow_length = 0.25
    arrow_tip = arrow_start + arrow_length

    ax.arrow(1.05, arrow_start, 0, arrow_length, transform=ax.transAxes,
            head_width=0.02, head_length=0.02, fc='green', ec='green', clip_on=False)

    ax.annotate("AS-\nStates",
                xy=(1.05, arrow_tip),          # tip of the arrow
                xytext=(1.05, arrow_tip + 0.02), # offset text slightly to the right and above
                textcoords='axes fraction',
                fontsize=14, fontweight='bold', color='green',
                ha='left', va='bottom')

    # AS+ States
    arrow_start = 0.4
    arrow_length = -0.25
    arrow_tip = arrow_start + arrow_length

    ax.arrow(1.05, arrow_start, 0, arrow_length, transform=ax.transAxes,
            head_width=0.02, head_length=0.02, fc='red', ec='red', clip_on=False)

    ax.annotate("AS+\nStates",
                xy=(1.05, arrow_tip),
                xytext=(1.05, arrow_tip - 0.02), # offset text slightly to the right and below
                textcoords='axes fraction',
                fontsize=14, fontweight='bold', color='red',
            ha='left', va='top')
    
    # Set labels and formatting
    ax.set_xlabel("Normalized Pseudotime", fontsize=16, fontweight='bold')
    ax.set_ylabel("Condition Bias (AS- vs. AS+)", fontsize=16, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.1, 1.1)
    ax.tick_params(axis='both', labelsize=14)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    #plt.tight_layout()
    plt.savefig("Figure4A.png", dpi=300)
    plt.close()
    print("Saved Figure4A.png")

    # ==================== FIGURE 4B: Condition Bias Histogram ====================
    print("\nGenerating Figure 4B: Condition Bias Histogram...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate percentages for AS-, AS+, Mixed using standard deviation threshold
    # Mixed = states within 1 standard deviation of zero
    n_as_minus = np.sum(condition_bias_array > mixed_threshold)
    n_as_plus = np.sum(condition_bias_array < -mixed_threshold)
    n_mixed = np.sum((condition_bias_array >= -mixed_threshold) & (condition_bias_array <= mixed_threshold))
    
    pct_as_minus = 100 * n_as_minus / num_states
    pct_as_plus = 100 * n_as_plus / num_states
    pct_mixed = 100 * n_mixed / num_states
    
    print(f"  AS- (bias > {mixed_threshold:.3f}): {n_as_minus} states ({pct_as_minus:.1f}%)")
    print(f"  Mixed (|bias| <= {mixed_threshold:.3f}): {n_mixed} states ({pct_mixed:.1f}%)")
    print(f"  AS+ (bias < -{mixed_threshold:.3f}): {n_as_plus} states ({pct_as_plus:.1f}%)")
    
    # Create histogram bins
    bins = np.linspace(-1.1, 1.1, 25)
    
    # Separate data into AS+, Mixed, AS- for coloring using std threshold
    as_plus_biases = condition_bias_array[condition_bias_array < -mixed_threshold]
    as_minus_biases = condition_bias_array[condition_bias_array > mixed_threshold]
    mixed_biases = condition_bias_array[(condition_bias_array >= -mixed_threshold) & (condition_bias_array <= mixed_threshold)]
    
    # Plot histograms with consistent colors: AS+ = red, Mixed = blue, AS- = teal
    ax.hist(as_plus_biases, bins=bins, color=color_as_plus, alpha=0.8, label='AS+', edgecolor='black')
    ax.hist(mixed_biases, bins=bins, color=color_mixed, alpha=0.8, label='Mixed', edgecolor='black')
    ax.hist(as_minus_biases, bins=bins, color=color_as_minus, alpha=0.8, label='AS-', edgecolor='black')
    
    # Add vertical dashed line at y=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    
    # Add vertical lines for mean and median
    mean_bias = np.mean(condition_bias_array)
    median_bias = np.median(condition_bias_array)
    
    ax.axvline(x=mean_bias, color='navy', linestyle='-', linewidth=2)
    ax.axvline(x=median_bias, color='darkmagenta', linestyle=':', linewidth=2)
    
    # Add statistics text box
    stats_text = f"AS-: {pct_as_minus:.1f}%\nAS+: {pct_as_plus:.1f}%\nMixed: {pct_mixed:.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add legend for mean/median
    legend_elements = [
        Line2D([0], [0], color='navy', linestyle='-', linewidth=2, label=f'Mean: {mean_bias:.3f}'),
        Line2D([0], [0], color='darkmagenta', linestyle=':', linewidth=2, label=f'Median: {median_bias:.3f}')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Set labels and formatting
    ax.set_xlabel("Condition Bias (AS- vs. AS+)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Frequency (Number of States)", fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    #plt.tight_layout()
    plt.savefig("Figure4B.png", dpi=300)
    plt.close()
    print("Saved Figure4B.png")
    
    # ==================== Combine Figure 4A and 4B ====================
    print("\nCombining Figure 4A and 4B...")
    
    images = ["Figure4A.png", "Figure4B.png"]
    missing_images = [img for img in images if not os.path.exists(img)]
    if missing_images:
        print(f"Warning: Missing images: {missing_images}")
        return
    
    img_data = [mpimg.imread(f) for f in images]
    
    fig = plt.figure(figsize=(10, 16))
    gs = GridSpec(2, 1, figure=fig, hspace=0.05)
    
    labels = ['a.', 'b.']
    for i in range(2):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img_data[i])
        ax.axis('off')
        ax.text(0.02, 1.02, labels[i], transform=ax.transAxes,
                fontsize=20, fontweight='bold', color='black',
                ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    #plt.tight_layout()
    plt.savefig("Figure4_Combined.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print("\nFigure 4 generation complete!")
    print("Individual figures: Figure4A.png, Figure4B.png")
    print("Combined panel: Figure4_Combined.png")


def generate_figure5():
    """
    Generate Figure 5: Perturbation Analysis
    - Figure 5A: Delta AS+ percentage bar chart for multiple genes
    - Figure 5B: KI bifurcation diagram with histogram
    - Figure 5C: KO bifurcation diagram with histogram
    """
    
    print("\n" + "="*60)
    print("Generating Figure 5: Perturbation Analysis")
    print("="*60)
    
    # ========== Load Required Data ==========
    print("\n[1/6] Loading trajectories...")
    trajectories = read_trajectories(INPUT_FILES["trajectories"])
    cleaned_trajectories = {cell: clean_trajectory(traj) for cell, traj in trajectories.items()}
    
    print("[2/6] Loading unique states...")
    with open(INPUT_FILES["unique_states"], 'r') as f:
        columns_dict = {
            key.strip(): list(map(int, value.strip().strip('[]').replace('(', '').replace(')', '').split(','))) 
            for key, value in (line.strip().split(':') for line in f)
        }
    
    unique_states = list(columns_dict.values())
    state_labels = list(columns_dict.keys())
    
    # ========== Calculate Pseudotime ==========
    print("[3/6] Calculating pseudotime...")
    normalized_steps_to_attractor = {
        state: ((len(clean_trajectory(trajectory)) - clean_trajectory(trajectory).index(state)) - 1) / len(clean_trajectory(trajectory))
        for trajectory in cleaned_trajectories.values()
        for state in clean_trajectory(trajectory)
        if state in clean_trajectory(trajectory) and len(clean_trajectory(trajectory)) > 1
    }
    
    state_pseudotime_dict = {state: 1 - normalized_steps_to_attractor[state] 
                            for state in normalized_steps_to_attractor}
    
    state_pseudotime_items = [
        (state_index, state_pseudotime_dict.get(state_index, 0), unique_states[i])
        for i, state_index in enumerate([int(s.split()[1]) for s in state_labels])
    ]
    state_pseudotime_items.sort(key=lambda x: x[1])
    num_states = len(state_pseudotime_items)
    
    sorted_indices = [x[0] for x in state_pseudotime_items]
    
    # ========== Calculate Condition Bias ==========
    print("[4/6] Calculating condition bias...")
    metadata = pd.read_csv(INPUT_FILES["metadata"], sep=" ", header=None, skiprows=0)
    cell_sample_indices = []
    conditions = []
    
    for _, row in metadata.iterrows():
        cell_sample_indices.append(int(str(row[0]).strip('"')) - 1)
        conditions.append(row[2])
    
    cell_condition_dict = {cell_sample_indices[i]: conditions[i] 
                          for i in range(len(cell_sample_indices))}
    
    condition_bias_array = np.zeros(num_states)
    
    for i in range(num_states):
        state_index = sorted_indices[i]
        cell_indices = [cell for cell, traj in cleaned_trajectories.items() if state_index in traj]
        
        cleaned_cell_indices = []
        for cell in cell_indices:
            try:
                idx = int(cell.split('_')[1])
                cleaned_cell_indices.append(idx)
            except (IndexError, ValueError):
                continue
        
        healthy_cells = 0
        diseased_cells = 0
        total_cells = 0
        
        for cell in cleaned_cell_indices:
            if cell in cell_condition_dict:
                condition = cell_condition_dict[cell]
                if condition == "AS-":
                    healthy_cells += 1
                elif condition == "AS+":
                    diseased_cells += 1
                total_cells += 1
        
        if total_cells > 0:
            healthy_percentage = healthy_cells / total_cells
            diseased_percentage = diseased_cells / total_cells
            condition_bias = healthy_percentage - diseased_percentage
            condition_bias_array[i] = condition_bias
        else:
            condition_bias_array[i] = 0
    
    max_y_value = np.max(condition_bias_array)
    min_y_value = np.min(condition_bias_array)
    
    # Create bifurcation dictionaries
    bifurcation_y_values = {}
    bifurcation_x_values = {}
    
    for i, state in enumerate(sorted_indices):
        binary_value = tuple(columns_dict[f"State {state}"])
        bifurcation_y_values[binary_value] = condition_bias_array[i]
        bifurcation_x_values[binary_value] = state_pseudotime_dict.get(state, 0)
    
    # ========== Train Model ==========
    print("[5/6] Training prediction model with cross-validation...")
    bifurcation_features = []
    for binary_tuple, x_value in bifurcation_x_values.items():
        y_value = bifurcation_y_values[binary_tuple]
        bifurcation_features.append((list(binary_tuple), x_value, y_value))
    
    X_raw = np.array([binary for binary, x_val, _ in bifurcation_features])
    Y_raw = np.array([[y_val, x_val] for binary, x_val, y_val in bifurcation_features])
    
    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X_raw)
    
    condition_bias_raw = Y_raw[:, 0]
    pseudotime_raw = Y_raw[:, 1]
    
    Y_multi = np.column_stack([condition_bias_raw, pseudotime_raw])
    
    # Create class labels for condition bias (AS-, Mixed, AS+)
    def get_condition_class(bias, threshold):
        if bias > threshold:
            return "AS-"
        elif bias < -threshold:
            return "AS+"
        else:
            return "Mixed"
    mixed_threshold = np.std(condition_bias_array)
    condition_bias_classes = np.array([get_condition_class(cb, mixed_threshold) for cb in condition_bias_raw])
    
    # ========== 5-Fold Cross-Validation ==========
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_scores_pseudotime = []
    cv_precision_macro = []
    cv_precision_weighted = []
    cv_accuracy = []
    
    # For aggregating all predictions for final contingency table
    all_true_classes = []
    all_pred_classes = []
    
    print(f"\n  Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Performing {n_folds}-fold cross-validation...")
        
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y_multi[train_idx], Y_multi[val_idx]
        y_class_val = condition_bias_classes[val_idx]
        
        fold_model = MultiOutputRegressor(
            MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', 
                        max_iter=10000, random_state=10000)
        )
        fold_model.fit(X_train, Y_train)
        
        Y_pred = fold_model.predict(X_val)
        
        # Pseudotime R²
        r2_pt = r2_score(Y_val[:, 1], Y_pred[:, 1])
        cv_scores_pseudotime.append(r2_pt)
        
        # Condition bias classification metrics
        pred_cond_bias = np.tanh(Y_pred[:, 0])
        pred_classes = np.array([get_condition_class(cb, mixed_threshold) for cb in pred_cond_bias])
        
        # Store for final contingency table
        all_true_classes.extend(y_class_val)
        all_pred_classes.extend(pred_classes)
        
        # Calculate precision
        precision_macro = precision_score(y_class_val, pred_classes, labels=["AS-", "Mixed", "AS+"], average='macro', zero_division=0)
        precision_weighted = precision_score(y_class_val, pred_classes, labels=["AS-", "Mixed", "AS+"], average='weighted', zero_division=0)
        acc = accuracy_score(y_class_val, pred_classes)
        
        cv_precision_macro.append(precision_macro)
        cv_precision_weighted.append(precision_weighted)
        cv_accuracy.append(acc)
        
        print(f"    Fold {fold+1}: Precision (macro) = {precision_macro:.4f}, Precision (weighted) = {precision_weighted:.4f}, Accuracy = {acc:.4f}, R² Pseudotime = {r2_pt:.4f}")
    
    mean_r2_pt = np.mean(cv_scores_pseudotime)
    std_r2_pt = np.std(cv_scores_pseudotime)
    mean_precision_macro = np.mean(cv_precision_macro)
    std_precision_macro = np.std(cv_precision_macro)
    mean_precision_weighted = np.mean(cv_precision_weighted)
    std_precision_weighted = np.std(cv_precision_weighted)
    mean_accuracy = np.mean(cv_accuracy)
    std_accuracy = np.std(cv_accuracy)
    
    print(f"\n  Cross-Validation Results ({n_folds}-fold):")
    print(f"    Condition Bias Classification:")
    print(f"      Precision (macro):    {mean_precision_macro:.4f} ± {std_precision_macro:.4f}")
    print(f"      Precision (weighted): {mean_precision_weighted:.4f} ± {std_precision_weighted:.4f}")
    print(f"      Accuracy:             {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"    Pseudotime R²:          {mean_r2_pt:.4f} ± {std_r2_pt:.4f}")
    
    # Generate contingency table from all CV predictions
    print(f"\n  Contingency Table (aggregated from all CV folds):")
    all_true_classes = np.array(all_true_classes)
    all_pred_classes = np.array(all_pred_classes)
    
    class_labels = ["AS-", "Mixed", "AS+"]
    cm = confusion_matrix(all_true_classes, all_pred_classes, labels=class_labels)
    
    # Print contingency table
    print(f"\n                    Predicted")
    print(f"                    {'AS-':>8} {'Mixed':>8} {'AS+':>8}")
    print(f"         " + "-" * 36)
    for i, true_label in enumerate(class_labels):
        print(f"  Actual {true_label:>5} |{cm[i, 0]:>8} {cm[i, 1]:>8} {cm[i, 2]:>8}")
    
    # Calculate per-class precision
    print(f"\n  Per-class Precision:")
    for i, label in enumerate(class_labels):
        col_sum = cm[:, i].sum()
        if col_sum > 0:
            precision = cm[i, i] / col_sum
            print(f"    {label}: {precision:.4f} ({cm[i, i]}/{col_sum})")
        else:
            print(f"    {label}: N/A (no predictions)")
    
    # Train final model on all data
    print("\n  Training final model on all data...")
    model_multi = MultiOutputRegressor(
        MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', 
                    max_iter=10000, random_state=10000)
    )
    model_multi.fit(X, Y_multi)
    
    # Training metrics
    Y_pred_train = model_multi.predict(X)
    pred_cond_bias_train = np.tanh(Y_pred_train[:, 0])
    pred_classes_train = np.array([get_condition_class(cb, mixed_threshold) for cb in pred_cond_bias_train])
    
    train_accuracy = accuracy_score(condition_bias_classes, pred_classes_train)
    train_precision_macro = precision_score(condition_bias_classes, pred_classes_train, labels=["AS-", "Mixed", "AS+"], average='macro', zero_division=0)
    train_r2_pt = r2_score(Y_multi[:, 1], Y_pred_train[:, 1])
    
    print(f"  Training metrics:")
    print(f"    Condition Bias Accuracy:       {train_accuracy:.4f}")
    print(f"    Condition Bias Precision:      {train_precision_macro:.4f}")
    print(f"    Pseudotime R²:                 {train_r2_pt:.4f}")
    
    # ========== Calculate Baseline Distribution with Mixed threshold ==========
    # Use same threshold as Figure 3E and Figure 4 (1 std dev from zero)
    mixed_threshold = np.std(condition_bias_array)
    print(f"\nMixed threshold (1 std dev): ±{mixed_threshold:.3f}")
    
    baseline_as_minus_states = np.sum(condition_bias_array > mixed_threshold)
    baseline_as_plus_states = np.sum(condition_bias_array < -mixed_threshold)
    baseline_mixed_states = np.sum((condition_bias_array >= -mixed_threshold) & (condition_bias_array <= mixed_threshold))
    
    print(f"Baseline: AS- states = {baseline_as_minus_states}, Mixed = {baseline_mixed_states}, AS+ states = {baseline_as_plus_states}")
    
    # ========== Process Perturbations ==========
    print("\n[6/6] Processing perturbations...")
    
    gene_interest = ["RXRB", "NRAS", "HSPA5", "MAP3K5", "EIF2AK3", "ERN1", 
                     "EIF2S1", "DDIT3", "ITPR1", "TRAF2", "XBP1", "ATF4",
                     "PLCG1", "NFATC1", "GSK3B", "AGER"]
    
    all_gene_results = {}
    selected_gene = "GSK3B"
    ki_data_selected = None
    ko_data_selected = None
    
    for gene in gene_interest:
        print(f"  Processing gene: {gene}")
        
        knockout_results_path = INPUT_FILES["knockout_results"].format(gene=gene)
        knockin_results_path = INPUT_FILES["knockin_results"].format(gene=gene)
        
        if not os.path.exists(knockout_results_path) or not os.path.exists(knockin_results_path):
            print(f"    Warning: Perturbation files not found for {gene}, skipping...")
            continue
        
        with open(knockout_results_path, 'rb') as f:
            knockout_results = pickle.load(f)
        with open(knockin_results_path, 'rb') as f:
            knockin_results = pickle.load(f)
        
        # Process knock-in
        y_values_ki = []
        perturbed_x_ki = []
        perturbed_pt_ki = []
        
        for cell_idx in cell_sample_indices:
            if cell_idx in knockin_results:
                knockin_traj = np.array(knockin_results[cell_idx])
                last_col_knockin = knockin_traj[:, -1]
                
                perturbed_binary = [int(bit) for bit in last_col_knockin]
                X_test = x_scaler.transform([perturbed_binary])
                raw_prediction = model_multi.predict(X_test)[0]
                
                cond_pred = np.tanh(raw_prediction[0])
                pseudotime_pred = np.clip(raw_prediction[1], 0, 1)
                
                perturbed_x_ki.append(pseudotime_pred)
                perturbed_pt_ki.append(pseudotime_pred)
                y_values_ki.append(cond_pred)
        
        # Process knock-out
        y_values_ko = []
        perturbed_x_ko = []
        perturbed_pt_ko = []
        
        for cell_idx in cell_sample_indices:
            if cell_idx in knockout_results:
                knockout_traj = np.array(knockout_results[cell_idx])
                last_col_knockout = knockout_traj[:, -1]
                
                perturbed_binary = [int(bit) for bit in last_col_knockout]
                X_test = x_scaler.transform([perturbed_binary])
                raw_prediction = model_multi.predict(X_test)[0]
                
                cond_pred = np.tanh(raw_prediction[0])
                pseudotime_pred = np.clip(raw_prediction[1], 0, 1)
                
                perturbed_x_ko.append(pseudotime_pred)
                perturbed_pt_ko.append(pseudotime_pred)
                y_values_ko.append(cond_pred)
        
        # Store results
        ki_as_minus = np.sum(np.array(y_values_ki) > 0)
        ki_as_plus = np.sum(np.array(y_values_ki) < 0)
        ko_as_minus = np.sum(np.array(y_values_ko) > 0)
        ko_as_plus = np.sum(np.array(y_values_ko) < 0)
        
        all_gene_results[gene] = {
            'baseline': {'as_minus': baseline_as_minus_states, 'as_plus': baseline_as_plus_states},
            'ki': {'as_minus': ki_as_minus, 'as_plus': ki_as_plus},
            'ko': {'as_minus': ko_as_minus, 'as_plus': ko_as_plus}
        }
        
        if gene == selected_gene:
            ki_data_selected = (perturbed_x_ki, y_values_ki, perturbed_pt_ki, gene)
            ko_data_selected = (perturbed_x_ko, y_values_ko, perturbed_pt_ko, gene)

    #========== Save the data for Figure 7 ========================
    np.save("baseline_condition_bias.npy", condition_bias_array)
    np.save("mixed_threshold.npy", np.array([mixed_threshold]))
    # ========== Save selected gene perturbation outputs ==========
    if ki_data_selected is not None and ko_data_selected is not None:
        _, y_values_ki, _, gene = ki_data_selected
        _, y_values_ko, _, _ = ko_data_selected

        np.save(f"{gene}_ki_condition_bias.npy", np.array(y_values_ki))
        np.save(f"{gene}_ko_condition_bias.npy", np.array(y_values_ko))

        print(f"Saved perturbation condition bias arrays for {gene}")
    else:
        print("Warning: Selected gene perturbation data not found.")

        if len(all_gene_results) == 0:
            print("Warning: No perturbation results found. Skipping Figure 5 generation.")
            return
        
    # ========== Generate Figure 5A: Delta AS+ Bar Chart ==========
    print("\nGenerating Figure 5A: Delta AS+ Percentage Bar Chart...")
    
    gene_names = []
    delta_ki_as_plus_list = []
    delta_ko_as_plus_list = []
    avg_delta_list = []
    
    for gene, data in all_gene_results.items():
        baseline_total = data['baseline']['as_minus'] + data['baseline']['as_plus']
        ki_total = data['ki']['as_minus'] + data['ki']['as_plus']
        ko_total = data['ko']['as_minus'] + data['ko']['as_plus']
        
        if baseline_total == 0 or ki_total == 0 or ko_total == 0:
            continue
        
        baseline_as_plus_pct = (data['baseline']['as_plus'] / baseline_total) * 100
        ki_as_plus_pct = (data['ki']['as_plus'] / ki_total) * 100
        ko_as_plus_pct = (data['ko']['as_plus'] / ko_total) * 100
        
        delta_ki_as_plus = baseline_as_plus_pct - ki_as_plus_pct
        delta_ko_as_plus = baseline_as_plus_pct - ko_as_plus_pct
        
        avg_delta = (abs(delta_ki_as_plus) + abs(delta_ko_as_plus)) / 2
        
        gene_names.append(gene)
        delta_ki_as_plus_list.append(delta_ki_as_plus)
        delta_ko_as_plus_list.append(delta_ko_as_plus)
        avg_delta_list.append(avg_delta)
    
    # Sort by average perturbation strength
    sorted_indices_genes = np.argsort(avg_delta_list)[::-1]
    gene_names_sorted = [gene_names[i] for i in sorted_indices_genes]
    delta_ki_as_plus_sorted = [delta_ki_as_plus_list[i] for i in sorted_indices_genes]
    delta_ko_as_plus_sorted = [delta_ko_as_plus_list[i] for i in sorted_indices_genes]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    x = np.arange(len(gene_names_sorted))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, delta_ki_as_plus_sorted, width, label='Knockin', 
                    color='purple', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.barh(x + width/2, delta_ko_as_plus_sorted, width, label='Knockout', 
                    color='yellow', alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.axvline(0, color='black', linewidth=2, linestyle='--')
    ax.axvline(20, color='black', linewidth=2, linestyle='--', alpha=0.5)
    ax.axvline(-20, color='black', linewidth=2, linestyle='--', alpha=0.5)
    ax.axvline(10, color='blue', linewidth=2, linestyle=':', alpha=0.5)
    ax.axvline(-10, color='blue', linewidth=2, linestyle=':', alpha=0.5)
    
    max_abs_delta = max(max(abs(d) for d in delta_ki_as_plus_sorted), 
                        max(abs(d) for d in delta_ko_as_plus_sorted))
    x_limit = max(max_abs_delta * 1.3, 35)
    ax.set_xlim(-x_limit, x_limit)
    
    ax.set_yticks(x)
    ax.set_yticklabels(gene_names_sorted, fontsize=14)
    ax.set_xlabel('Delta AS+ Percentage (Baseline - Perturbed)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Gene', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, loc='lower right', frameon=True, edgecolor='black')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=14)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    #plt.tight_layout()
    plt.savefig("Figure5A.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Figure5A.png")
    
    # ========== Generate Figure 5B and 5C: Bifurcation Diagrams ==========
    if ki_data_selected is not None and ko_data_selected is not None:
        perturbed_x_ki, y_values_ki, perturbed_pt_ki, gene = ki_data_selected
        perturbed_x_ko, y_values_ko, perturbed_pt_ko, _ = ko_data_selected
        
        # Figure 5B: Knockin Bifurcation
        print(f"\nGenerating Figure 5B: {gene} Knockin Bifurcation...")
        
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
        
        # Main scatter plot
        scatter = ax_main.scatter(perturbed_x_ki, y_values_ki, c=perturbed_pt_ki, cmap='coolwarm', 
                        edgecolor='black', s=150, marker='^', alpha=0.6)
        
        mean_val = np.mean(y_values_ki)
        median_val = np.median(y_values_ki)
        ax_main.axhline(mean_val, color='blue', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax_main.axhline(median_val, color='darkblue', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        ax_main.axhline(0, color='black', linestyle='--', linewidth=2)
        ax_main.legend(loc='lower left', fontsize=14)
        
        # Histogram with colored bars based on condition bias (with Mixed region)
        counts, bins_edges, patches = ax_hist.hist(y_values_ki, bins=20, range=(-1, 1), 
                                                    orientation='horizontal', edgecolor='black', alpha=0.85)
        # Color bars: red for AS+ (< -threshold), blue for Mixed, teal for AS- (> threshold)
        for patch, edge in zip(patches, bins_edges[:-1]):
            bin_center = edge + 0.05  # Approximate center of bin
            if bin_center < -mixed_threshold:
                patch.set_facecolor('red')  # AS+
            elif bin_center > mixed_threshold:
                patch.set_facecolor('teal')  # AS-
            else:
                patch.set_facecolor('blue')  # Mixed
        
        ax_hist.axhline(0, color='black', linestyle='--', linewidth=2)
        ax_hist.axhline(mixed_threshold, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax_hist.axhline(-mixed_threshold, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax_hist.axhline(mean_val, color='blue', linestyle=':', linewidth=2)
        ax_hist.axhline(median_val, color='darkblue', linestyle='--', linewidth=2)
        ax_hist.yaxis.set_visible(False)
        
        # Stats with Mixed
        y_arr_ki = np.array(y_values_ki)
        as_minus_pct = (np.sum(y_arr_ki > mixed_threshold) / len(y_values_ki) * 100)
        as_plus_pct = (np.sum(y_arr_ki < -mixed_threshold) / len(y_values_ki) * 100)
        mixed_pct = (np.sum((y_arr_ki >= -mixed_threshold) & (y_arr_ki <= mixed_threshold)) / len(y_values_ki) * 100)
        ax_main.text(0.02, 0.98, f'AS-: {as_minus_pct:.1f}%\nMixed: {mixed_pct:.1f}%\nAS+: {as_plus_pct:.1f}%', 
                     transform=ax_main.transAxes, fontsize=14, verticalalignment='top',
                     fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax_main.set_xlim(-0.1, 1.1)
        ax_main.set_ylim(-1.1, 1.1)
        ax_main.set_xlabel("Normalized Pseudotime", fontsize=16, fontweight='bold')
        ax_main.set_ylabel("Condition Bias (AS- vs. AS+)", fontsize=16, fontweight='bold')
        ax_main.set_title(f'{gene} Knockin', fontsize=18, fontweight='bold')
        ax_main.tick_params(labelsize=14)
        ax_hist.set_xlabel("Frequency", fontsize=14, fontweight='bold')
        ax_hist.tick_params(labelsize=14)
        
        for spine in ax_main.spines.values():
            spine.set_linewidth(2)
        for spine in ax_hist.spines.values():
            spine.set_linewidth(2)
        
        #plt.tight_layout()
        plt.savefig("Figure5B.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Figure5B.png")
        
        # Figure 5C: Knockout Bifurcation
        print(f"Generating Figure 5C: {gene} Knockout Bifurcation...")
        
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
        
        scatter = ax_main.scatter(perturbed_x_ko, y_values_ko, c=perturbed_pt_ko, cmap='coolwarm', 
                        edgecolor='black', s=150, marker='v', alpha=0.6)
        
        mean_val = np.mean(y_values_ko)
        median_val = np.median(y_values_ko)
        ax_main.axhline(mean_val, color='blue', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax_main.axhline(median_val, color='darkblue', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        ax_main.axhline(0, color='black', linestyle='--', linewidth=2)
        ax_main.legend(loc='lower left', fontsize=14)
        
        # Histogram with colored bars based on condition bias (with Mixed region)
        counts, bins_edges, patches = ax_hist.hist(y_values_ko, bins=20, range=(-1, 1), 
                                                    orientation='horizontal', edgecolor='black', alpha=0.85)
        # Color bars: red for AS+ (< -threshold), blue for Mixed, teal for AS- (> threshold)
        for patch, edge in zip(patches, bins_edges[:-1]):
            bin_center = edge + 0.05  # Approximate center of bin
            if bin_center < -mixed_threshold:
                patch.set_facecolor('red')  # AS+
            elif bin_center > mixed_threshold:
                patch.set_facecolor('teal')  # AS-
            else:
                patch.set_facecolor('blue')  # Mixed
        
        ax_hist.axhline(0, color='black', linestyle='--', linewidth=2)
        ax_hist.axhline(mixed_threshold, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax_hist.axhline(-mixed_threshold, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax_hist.axhline(mean_val, color='blue', linestyle=':', linewidth=2)
        ax_hist.axhline(median_val, color='darkblue', linestyle='--', linewidth=2)
        ax_hist.yaxis.set_visible(False)
        
        # Stats with Mixed
        y_arr_ko = np.array(y_values_ko)
        as_minus_pct = (np.sum(y_arr_ko > mixed_threshold) / len(y_values_ko) * 100)
        as_plus_pct = (np.sum(y_arr_ko < -mixed_threshold) / len(y_values_ko) * 100)
        mixed_pct = (np.sum((y_arr_ko >= -mixed_threshold) & (y_arr_ko <= mixed_threshold)) / len(y_values_ko) * 100)
        ax_main.text(0.02, 0.98, f'AS-: {as_minus_pct:.1f}%\nMixed: {mixed_pct:.1f}%\nAS+: {as_plus_pct:.1f}%', 
                     transform=ax_main.transAxes, fontsize=14, verticalalignment='top',
                     fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax_main.set_xlim(-0.1, 1.1)
        ax_main.set_ylim(-1.1, 1.1)
        ax_main.set_xlabel("Normalized Pseudotime", fontsize=16, fontweight='bold')
        ax_main.set_ylabel("Condition Bias (AS- vs. AS+)", fontsize=16, fontweight='bold')
        ax_main.set_title(f'{gene} Knockout', fontsize=18, fontweight='bold')
        ax_main.tick_params(labelsize=14)
        ax_hist.set_xlabel("Frequency", fontsize=14, fontweight='bold')
        ax_hist.tick_params(labelsize=14)
        
        for spine in ax_main.spines.values():
            spine.set_linewidth(2)
        for spine in ax_hist.spines.values():
            spine.set_linewidth(2)
        
        #plt.tight_layout()
        plt.savefig("Figure5C.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved Figure5C.png")
    
    # ========== Combine Figure 5 ==========
    print("\nCombining Figure 5A, 5B, and 5C...")
    
    images = ["Figure5A.png", "Figure5B.png", "Figure5C.png"]
    missing_images = [img for img in images if not os.path.exists(img)]
    if missing_images:
        print(f"Warning: Missing images: {missing_images}")
        return
    
    img_data = [mpimg.imread(f) for f in images]
    
    fig = plt.figure(figsize=(20, 22))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.15, wspace=0.1)
    
    # Top: Figure 5A spanning both columns
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(img_data[0])
    ax_top.axis('off')
    ax_top.text(0.02, 1.02, 'a.', transform=ax_top.transAxes,
                fontsize=24, fontweight='bold', color='black',
                ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Bottom Left: Figure 5B
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bl.imshow(img_data[1])
    ax_bl.axis('off')
    ax_bl.text(0.02, 1.02, 'b.', transform=ax_bl.transAxes,
               fontsize=24, fontweight='bold', color='black',
               ha='left', va='bottom',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Bottom Right: Figure 5C
    ax_br = fig.add_subplot(gs[1, 1])
    ax_br.imshow(img_data[2])
    ax_br.axis('off')
    ax_br.text(0.02, 1.02, 'c.', transform=ax_br.transAxes,
               fontsize=24, fontweight='bold', color='black',
               ha='left', va='bottom',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    #plt.tight_layout()
    plt.savefig("Figure5_Combined.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print("\nFigure 5 generation complete!")
    print("Individual figures: Figure5A.png, Figure5B.png, Figure5C.png")
    print("Combined panel: Figure5_Combined.png")

#========== Helper function for Figure 6 ==========
def calculate_participant_distributions(unique_states, participant_ids, sorted_attractors, columns_dict):
    """Calculate attractor distributions for each participant."""
    participant_distributions = {}
    
    for pid in sorted(set(participant_ids), key=lambda x: int(x) if x.isdigit() else 0):
        # Get cells for this participant
        participant_cell_indices = [i for i, p in enumerate(participant_ids) if p == pid]
        
        # Get their final states
        perturbed_states = [unique_states[i] 
                           for i in participant_cell_indices 
                           if i < len(unique_states)]
        
        if not perturbed_states:
            continue
        
        # Calculate which attractors these states belong to
        attractor_counts = {att: 0 for att in sorted_attractors[:5]}
        
        for state in perturbed_states:
            state_tuple = tuple(int(bit) for bit in state)
            # Find which attractor this state matches
            for att in sorted_attractors[:5]:
                att_key = f"State {att}"
                if att_key in columns_dict:
                    att_tuple = tuple(columns_dict[att_key])
                    if state_tuple == att_tuple:
                        attractor_counts[att] += 1
                        break
        
        # Normalize to percentages
        total_cells = len(perturbed_states)
        attractor_percentages = np.array([attractor_counts[att] / total_cells * 100 if total_cells > 0 else 0
                                          for att in sorted_attractors[:5]])
        participant_distributions[pid] = attractor_percentages
    
    return participant_distributions

# ================== FIGURE 6 FUNCTION  ==================
def generate_figure6():
    """
    Generate Figure 6: Attractor Distribution Analysis for Normal Signaling
    - Heatmap showing percentage of cells in each attractor for each participant
    - Scatter plot showing Attr(1+2) vs Attr(3+4) colored by AS status
    """
    
    print("\n" + "="*60)
    print("Generating Figure 6: Normal Signaling Attractor Distribution")
    print("="*60)
    
    # File paths
    trajectory_file = INPUT_FILES["trajectories"]
    unique_states_file = INPUT_FILES["unique_states"]
    metadata_file = INPUT_FILES["metadata"]
    
    # Check for required files
    for file in [trajectory_file, unique_states_file, metadata_file]:
        if not os.path.exists(file):
            print(f"Error: Required file '{file}' not found")
            return
    
    print("\nLoading data...")
    
    # Load trajectories
    trajectories = read_trajectories(trajectory_file)
    if not trajectories:
        print("Error: No trajectories loaded")
        return
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Load unique states
    with open(unique_states_file, 'r') as f:
        columns_dict = {}
        for line in f:
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            cleaned_value = value.strip().strip('[]').replace('(', '').replace(')', '')
            columns_dict[key.strip()] = list(map(int, cleaned_value.split(',')))
    print(f"Loaded {len(columns_dict)} unique states")
    
    # Load metadata
    metadata = pd.read_csv(metadata_file, sep=" ", header=None, skiprows=0)
    participant_ids = metadata.iloc[:, 1].apply(
        lambda x: str(x).split('.')[-1].replace('Participant_', '')
    ).tolist()
    
    cell_sample_indices = []
    for _, row in metadata.iterrows():
        try:
            cell_sample_indices.append(int(str(row[0]).strip('"')) - 1)
        except Exception as e:
            print(f"Warning: Error processing metadata row: {e}")
            continue
    print(f"Loaded metadata for {len(cell_sample_indices)} cells")
    
    print("\nFinding attractors...")
    
    # Find attractors
    attractors = {}
    for cell, trajectory in trajectories.items():
        cleaned_trajectory = clean_trajectory(trajectory)
        if not cleaned_trajectory:
            continue
        
        single_attractor, cyclic_attractor, attractor_state, _ = find_attractors(cleaned_trajectory)
        attractor = single_attractor if single_attractor is not None else (cyclic_attractor[0] if cyclic_attractor else None)
        
        if attractor is None:
            continue
        
        if attractor not in attractors:
            attractors[attractor] = []
        attractors[attractor].append(cell)
    
    # Sort attractors by basin size
    basins = {
        attractor: [cell for cell, trajectory in trajectories.items() if attractor in trajectory]
        for attractor in attractors.keys()
    }
    sorted_attractors = sorted(basins, key=lambda x: len(basins[x]), reverse=True)
    print(f"Found {len(sorted_attractors)} attractors")
    
    print("\nProcessing normal signaling states...")
    
    # Extract baseline final states from trajectories using columns_dict lookup
    unique_states_normal = []
    for cell_idx in cell_sample_indices:
        cell_key = f"cell_{cell_idx + 1}_trajectory"
        if cell_key in trajectories:
            traj = trajectories[cell_key]
            if traj:
                final_state_num = traj[-1]
                state_key = f"State {final_state_num}"
                if state_key in columns_dict:
                    unique_states_normal.append(columns_dict[state_key])
                else:
                    print(f"Warning: State {final_state_num} not found in columns_dict for {cell_key}")
    
    if not unique_states_normal:
        print("Error: No normal signaling states found")
        return
    
    print(f"Processing {len(unique_states_normal)} normal signaling states")
    
    # Calculate distributions
    participant_distributions_normal = calculate_participant_distributions(
        unique_states_normal, participant_ids, sorted_attractors, columns_dict
    )
    
    pids = sorted(participant_distributions_normal.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    as_status_dict = {pid: 'AS+' if int(pid) > 4 else 'AS-' for pid in pids}
    
    distributions_normal = np.array([participant_distributions_normal[pid] for pid in pids])
    
    print("\nGenerating figures separately...")

    # ========== FIGURE 6A: HEATMAP ==========
    fig_6a, ax1 = plt.subplots(figsize=(10, 8))

    im = ax1.imshow(distributions_normal, cmap='YlOrRd', aspect='auto')
    ax1.set_yticks(range(len(pids)))
    ax1.set_yticklabels([f"P{pid} ({as_status_dict[pid]})" for pid in pids], fontsize=18)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels([f"Attr{i+1}" for i in range(5)], fontsize=18)
    ax1.set_xlabel('Top 5 Attractors', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Participants', fontsize=18, fontweight='bold')

    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    for i in range(len(pids)):
        for j in range(5):
            ax1.text(j, i, f'{distributions_normal[i, j]:.1f}',
                    ha="center", va="center", color="black", fontsize=18)

    cbar = plt.colorbar(im, ax=ax1, label='Percentage (%)')
    cbar.outline.set_linewidth(2)
    cbar.set_label('Percentage (%)', fontsize=18, weight='bold')
    cbar.ax.tick_params(labelsize=18)

    #plt.tight_layout()
    plt.savefig('Figure6A.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure6A.png (Heatmap)")

    # ========== FIGURE 6B: SCATTER PLOT ==========
    fig_6b, ax2 = plt.subplots(figsize=(10, 8))

    # X-axis: (Attr1 + Attr2) %
    x_vals = distributions_normal[:, 0] + distributions_normal[:, 1]
    # Y-axis: (Attr3 + Attr4) %
    y_vals = distributions_normal[:, 2] + distributions_normal[:, 3]

    for i, pid in enumerate(pids):
        as_status = as_status_dict[pid]
        color = 'green' if as_status == 'AS-' else 'red'
        marker = 'o' if as_status == 'AS-' else 's'
        ax2.scatter(x_vals[i], y_vals[i], c=color, marker=marker, s=500, 
                edgecolors='black', linewidths=2, 
                label=as_status if i == 0 or (i > 0 and as_status_dict[pids[i-1]] != as_status) else "")
        ax2.text(x_vals[i], y_vals[i], f"P{pid}", fontsize=18, weight='bold', 
                ha='center', va='center')

    ax2.set_xlabel('Combined Percentage (% Attr1 + Attr2)', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Combined Percentage (% Attr3 + Attr4)', fontsize=18, fontweight='bold')
    ax2.set_xlim(35, 60)
    ax2.set_ylim(5, 20)
    ax2.xaxis.set_tick_params(labelsize=18)
    ax2.yaxis.set_tick_params(labelsize=18)
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    #plt.tight_layout()
    plt.savefig('Figure6B.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure6B.png (Scatter plot)")

    # ========== OPTIONAL: COMBINED FIGURE ==========
    print("\nGenerating combined figure...")

    fig = plt.figure(figsize=(10, 14))

    # Top: Figure 6A
    ax1 = plt.subplot(2, 1, 1)
    im = ax1.imshow(distributions_normal, cmap='YlOrRd', aspect='auto')
    ax1.set_yticks(range(len(pids)))
    ax1.set_yticklabels([f"P{pid} ({as_status_dict[pid]})" for pid in pids], fontsize=18)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels([f"Attr{i+1}" for i in range(5)], fontsize=18)
    ax1.set_xlabel('Top 5 Attractors', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Participants', fontsize=18, fontweight='bold')

    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    for i in range(len(pids)):
        for j in range(5):
            ax1.text(j, i, f'{distributions_normal[i, j]:.1f}',
                    ha="center", va="center", color="black", fontsize=18)

    cbar = plt.colorbar(im, ax=ax1, label='Percentage (%)')
    cbar.outline.set_linewidth(2)
    cbar.set_label('Percentage (%)', fontsize=18, weight='bold')
    cbar.ax.tick_params(labelsize=18)

    # Bottom: Figure 6B
    ax2 = plt.subplot(2, 1, 2)

    x_vals = distributions_normal[:, 0] + distributions_normal[:, 1]
    y_vals = distributions_normal[:, 2] + distributions_normal[:, 3]

    for i, pid in enumerate(pids):
        as_status = as_status_dict[pid]
        color = 'green' if as_status == 'AS-' else 'red'
        marker = 'o' if as_status == 'AS-' else 's'
        ax2.scatter(x_vals[i], y_vals[i], c=color, marker=marker, s=500, 
                edgecolors='black', linewidths=2, 
                label=as_status if i == 0 or (i > 0 and as_status_dict[pids[i-1]] != as_status) else "")
        ax2.text(x_vals[i], y_vals[i], f"P{pid}", fontsize=18, weight='bold', 
                ha='center', va='center')

    ax2.set_xlabel('Combined Percentage (% Attr1 + Attr2)', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Combined Percentage (% Attr3 + Attr4)', fontsize=18, fontweight='bold')
    ax2.set_xlim(35, 60)
    ax2.set_ylim(5, 20)
    ax2.xaxis.set_tick_params(labelsize=18)
    ax2.yaxis.set_tick_params(labelsize=18)
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    #plt.tight_layout()
    combined_filename = 'Figure6_Combined.png'
    plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure: {combined_filename}")

    # Save data
    output_file = "Participant_Attractor_Distribution_Normal.txt"
    with open(output_file, 'w') as f:
        f.write("Participant\tAS_Status\t" + "\t".join([f"Attractor_{i+1}_%" for i in range(5)]) + "\n")
        for pid in pids:
            as_status = as_status_dict[pid]
            attractor_dist = participant_distributions_normal[pid]
            dist_str = "\t".join([f"{v:.2f}" for v in attractor_dist])
            f.write(f"P{pid}\t{as_status}\t{dist_str}\n")
    print(f"Saved data: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("FIGURE 6 SUMMARY")
    print("="*60)
    for pid in pids:
        dist = participant_distributions_normal[pid]
        attr_1_2 = dist[0] + dist[1]
        attr_3_4 = dist[2] + dist[3]
        print(f"P{pid} ({as_status_dict[pid]}): Attr(1+2)={attr_1_2:.1f}%, Attr(3+4)={attr_3_4:.1f}%")

    print("\n" + "="*60)
    print("FIGURE 6 COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("Figure6A.png (Heatmap)")
    print("Figure6B.png (Scatter plot)")
    print("Figure6_Combined_Jan.png (Combined panel)")
    print("Participant_Attractor_Distribution_Normal.txt")
    print("="*60)

def generate_new_figure6(
    fig7a_path,
    fig7b_path,
    fig6a_path,
    fig6b_path,
    output_path="Figure6.png",
    figsize=(10, 10),
    dpi=300
):
    """
    Generate a new Figure 6 composed of:
      Row 1: (a) Figure 7a, (b) Figure 7b
      Row 2: (c) Figure 6a, (d) Figure 6b
    """

    # Load images
    images = [
        mpimg.imread(fig7a_path),
        mpimg.imread(fig7b_path),
        mpimg.imread(fig6a_path),
        mpimg.imread(fig6b_path),
    ]

    labels = ['a.', 'b.', 'c.', 'd.']

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, img, label in zip(axes.flatten(), images, labels):
        ax.imshow(img)
        ax.axis('off')
        ax.text(
            0.02, 1.1, label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top',
            ha='left'
        )

    plt.subplots_adjust(hspace=0.0, wspace=0.02)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def generate_figure7(selected_gene):
    """
    Generate Figure 7: Sankey plots using saved perturbation outputs.
    """
    required_files = [
        "baseline_condition_bias.npy",
        "mixed_threshold.npy",
        "GSK3B_ki_condition_bias.npy",
        "GSK3B_ko_condition_bias.npy",
    ]

    for f in required_files:
        if not os.path.exists(f):
            raise RuntimeError(f"Missing file: {f}")
        
    #========Load data==========
    baseline_bias = np.load("baseline_condition_bias.npy")
    ki_bias = np.load("GSK3B_ki_condition_bias.npy") 
    ko_bias = np.load("GSK3B_ko_condition_bias.npy")
    mixed_threshold = np.load("mixed_threshold.npy")[0]   

    # ---------- Classification ----------
    def classify_state_scalar(val, threshold):
        if val > threshold:
            return "AS-"
        elif val < -threshold:
            return "AS+"
        else:
            return "Mixed"
        
    # ========== Helper function to create Sankey plot ==========
    def create_sankey_plot(baseline_classes, perturbed_classes, title, output_file):
        """Create a Sankey plot showing flow between baseline and perturbed states"""
        
        # Count transitions
        transitions = {}
        for baseline, perturbed in zip(baseline_classes, perturbed_classes):
            key = (baseline, perturbed)
            transitions[key] = transitions.get(key, 0) + 1
        
        # Define node labels and colors
        labels = ["AS- (Before)", "Mixed (Before)", "AS+ (Before)", 
                  "AS- (After)", "Mixed (After)", "AS+ (After)"]
        
        # Colors: AS- = teal, Mixed = blue, AS+ = red
        node_colors = ['teal', 'blue', 'red', 'teal', 'blue', 'red']
        
        # Map state names to indices
        state_to_idx_before = {"AS-": 0, "Mixed": 1, "AS+": 2}
        state_to_idx_after = {"AS-": 3, "Mixed": 4, "AS+": 5}
        
        # Build source, target, value lists for Sankey
        sources = []
        targets = []
        values = []
        link_colors = []
        
        # Color mapping for links (based on source)
        link_color_map = {
            "AS-": "rgba(0, 128, 128, 0.4)",   # teal with transparency
            "Mixed": "rgba(0, 0, 255, 0.4)",    # blue with transparency
            "AS+": "rgba(255, 0, 0, 0.4)"       # red with transparency
        }
        
        for (before, after), count in transitions.items():
            sources.append(state_to_idx_before[before])
            targets.append(state_to_idx_after[after])
            values.append(count)
            link_colors.append(link_color_map[before])
        
        # Create figure using matplotlib (since plotly may not be available)
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculate node positions
        before_counts = {"AS-": 0, "Mixed": 0, "AS+": 0}
        after_counts = {"AS-": 0, "Mixed": 0, "AS+": 0}
        
        for bc in baseline_classes:
            before_counts[bc] += 1
        for pc in perturbed_classes:
            after_counts[pc] += 1
        
        total = len(baseline_classes)
        
        # Calculate proportional heights (scaled to fit in 0.8 of the plot height)
        scale_factor = 0.7
        # Compute total counts per column
        before_total = sum(before_counts.values())
        after_total = sum(after_counts.values())
        before_heights = {k: (v/before_total) * scale_factor for k, v in before_counts.items()}
        after_heights = {k: (v/after_total) * scale_factor for k, v in after_counts.items()}
        
        # Node x positions
        x_before = 0.15
        x_after = 0.85
        node_width = 0.08
        
        # Calculate y positions for nodes (starting from top, with gaps)
        gap = 0.03
        
        # Before nodes - start from 0.9 and go down
        before_y = {}
        current_y = 0.85
        for state in ["AS-", "Mixed", "AS+"]:
            height = before_heights[state]
            before_y[state] = (current_y, current_y - height)  # (top, bottom)
            current_y -= height + gap
        
        # After nodes - start from 0.9 and go down
        after_y = {}
        current_y = 0.85
        for state in ["AS-", "Mixed", "AS+"]:
            height = after_heights[state]
            after_y[state] = (current_y, current_y - height)  # (top, bottom)
            current_y -= height + gap
        
        # Color map
        color_map = {"AS-": "teal", "Mixed": "blue", "AS+": "red"}
        
        # Draw nodes (rectangles)
        for state in ["AS-", "Mixed", "AS+"]:
            # Before node
            height = before_heights[state]
            if height > 0:
                y_top, y_bottom = before_y[state]
                rect = plt.Rectangle((x_before - node_width/2, y_bottom), 
                                     node_width, height, 
                                     facecolor=color_map[state], edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                # Label - position at center of node
                y_center = (y_top + y_bottom) / 2
                ax.text(x_before - node_width/2 - 0.02, y_center, 
                       f"{state}\n({before_counts[state]})", 
                       ha='right', va='center', fontsize=14, fontweight='bold')
            
            # After node
            height = after_heights[state]
            if height > 0:
                y_top, y_bottom = after_y[state]
                rect = plt.Rectangle((x_after - node_width/2, y_bottom), 
                                     node_width, height,
                                     facecolor=color_map[state], edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                # Label - position at center of node
                y_center = (y_top + y_bottom) / 2
                ax.text(x_after + node_width/2 + 0.02, y_center, 
                       f"{state}\n({after_counts[state]})", 
                       ha='left', va='center', fontsize=14, fontweight='bold')
        
        # Draw flows (curved bands)
        # Track y offsets for stacking flows (start from top of each node)
        before_y_offset = {state: before_y[state][0] for state in before_y}
        after_y_offset = {state: after_y[state][0] for state in after_y}
        
        for (before_state, after_state), count in sorted(transitions.items(), key=lambda x: -x[1]):
            if count == 0:
                continue
            
            flow_height = (count / total) * scale_factor
            
            # Calculate y positions for this flow
            y1_top = before_y_offset[before_state]
            y1_bottom = y1_top - flow_height
            before_y_offset[before_state] = y1_bottom
            
            y2_top = after_y_offset[after_state]
            y2_bottom = y2_top - flow_height
            after_y_offset[after_state] = y2_bottom
            
            # Create curved path
            x1 = x_before + node_width/2
            x2 = x_after - node_width/2
            
            # Control points for bezier curve
            ctrl_x = (x1 + x2) / 2
            
            # Create path for flow band
            verts = [
                (x1, y1_top),  # Start top
                (ctrl_x, y1_top),  # Control point top
                (x2, y2_top),  # End top
                (x2, y2_bottom),  # End bottom
                (ctrl_x, y1_bottom),  # Control point bottom
                (x1, y1_bottom),  # Start bottom
                (x1, y1_top),  # Close
            ]
            
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, 
                    Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
            
            path = Path(verts, codes)
            
            # Use source color with transparency
            flow_color = color_map[before_state]
            patch = PathPatch(path, facecolor=flow_color, alpha=0.4, 
                            edgecolor=flow_color, linewidth=0.5)
            ax.add_patch(patch)
            
            # # Add flow count label in the middle
            # mid_y = (y1_top + y1_bottom + y2_top + y2_bottom) / 4
            # if flow_height > 0.03:  # Only label significant flows
            #     ax.text(0.5, mid_y, str(count), ha='center', va='center', 
            #            fontsize=10, fontweight='bold', color='black',
            #            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Add title and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.0)
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.text(x_before, 0.92, "Before\nPerturbation", ha='center', va='bottom', 
               fontsize=16, fontweight='bold')
        ax.text(x_after, 0.92, "After\nPerturbation", ha='center', va='bottom', 
               fontsize=16, fontweight='bold')
        
        ax.axis('off')
        
        #plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_file}")
        
        return transitions        

    baseline_classes = [
        classify_state_scalar(v, mixed_threshold) for v in baseline_bias
    ]
    ki_classes = [
        classify_state_scalar(v, mixed_threshold) for v in ki_bias
    ]
    ko_classes = [
        classify_state_scalar(v, mixed_threshold) for v in ko_bias
    ]

    # ---------- Sanity checks ----------
    valid_states = {"AS-", "Mixed", "AS+"}

    assert set(baseline_classes).issubset(valid_states)
    assert set(ki_classes).issubset(valid_states)
    assert set(ko_classes).issubset(valid_states)

    print("\nBaseline distribution:", Counter(baseline_classes))
    print("Knockin distribution:", Counter(ki_classes))
    print("Knockout distribution:", Counter(ko_classes))

    # ---------- Sankey plots ----------
    print(f"\nGenerating Figure7A: {selected_gene} Knockin Sankey Plot...")
    create_sankey_plot(
        baseline_classes,
        ki_classes,
        f"{selected_gene} Knockin: State Transitions",
        "Figure7A.png",
    )

    print(f"Generating Figure7B: {selected_gene} Knockout Sankey Plot...")
    create_sankey_plot(
        baseline_classes,
        ko_classes,
        f"{selected_gene} Knockout: State Transitions",
        "Figure7B.png",
    )

    # ------------- Save results to CSV -------------
    # Save detailed state information for each cell
    detailed_output = f"{selected_gene}_state_transitions_detailed.csv"
    with open(detailed_output, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Cell_Index",
            "Baseline_Condition_Bias",
            "Baseline_Class",
            "KI_Condition_Bias",
            "KI_Class",
            "KO_Condition_Bias", 
            "KO_Class"
        ])
        
        for i in range(len(baseline_classes)):
            writer.writerow([
                i,
                f"{baseline_bias[i]:.6f}",
                baseline_classes[i],
                f"{ki_bias[i]:.6f}",
                ki_classes[i],
                f"{ko_bias[i]:.6f}",
                ko_classes[i]
            ])
    
    print(f"Saved detailed cell-level data to '{detailed_output}'")
    
    # Save knockin transition summary
    ki_transitions_file = f"{selected_gene}_KI_transitions_summary.csv"
    with open(ki_transitions_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Baseline_Class", "KI_Class", "Count", "Percentage"])
        ki_transitions = Counter(zip(baseline_classes, ki_classes))
        total = len(baseline_classes)
        for (baseline_class, ki_class), count in sorted(ki_transitions.items()):
            percentage = (count / total) * 100
            writer.writerow([baseline_class, ki_class, count, f"{percentage:.2f}%"])
    
    print(f"Saved knockin transition summary to '{ki_transitions_file}'")
    
    # Save knockout transition summary
    ko_transitions_file = f"{selected_gene}_KO_transitions_summary.csv"
    with open(ko_transitions_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Baseline_Class", "KO_Class", "Count", "Percentage"])
        ko_transitions = Counter(zip(baseline_classes, ko_classes))
        for (baseline_class, ko_class), count in sorted(ko_transitions.items()):
            percentage = (count / total) * 100
            writer.writerow([baseline_class, ko_class, count, f"{percentage:.2f}%"])
    
    print(f"Saved knockout transition summary to '{ko_transitions_file}'")

    # ---------- Combine panels ----------
    print("\nCombining Figure7A and 7B...")

    images = ["Figure7A.png", "Figure7B.png"]
    img_data = [mpimg.imread(f) for f in images]

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2, figure=fig, wspace=0.1)

    for i, label in enumerate(["a.", "b."]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img_data[i])
        ax.axis("off")
        ax.text(
            0.02, 1.02, label,
            transform=ax.transAxes,
            fontsize=24, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
        )

    #plt.tight_layout()
    plt.savefig("Figure7_Combined.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print("\nFigure 7 generation complete!")
    print("Files saved: Figure7A.png, Figure7B.png, Figure7_Combined.png")

def combine_figures():
    """Combine all figures into a single panel"""
    images = [
        "Figure3A.png",
        "Figure3B.png",
        "Figure3C.png",
        "Cell_Transition_Matrix_Region.png",
        "Figure3EOrg.png",
        "Figure3FOrg.png"
    ]
    
    # Check if all images exist
    missing_images = [img for img in images if not os.path.exists(img)]
    if missing_images:
        print(f"Warning: Missing images: {missing_images}")
        print("Skipping combine step.")
        return
    
    img_data = [mpimg.imread(f) for f in images]
    
    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0, wspace=0)
    
    label_x = 0.03
    label_y = 1.05
    
    for i in range(6):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img_data[i])
        ax.axis('off')
        label = chr(ord('a') + i) + "."
        ax.text(
            label_x, label_y, label, transform=ax.transAxes,
            fontsize=20, fontweight='bold', color='black',
            ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        )
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.margins(0)
    plt.savefig("Figure3_Combined_Dec.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print("Combined Figure 3 panel created successfully!")
    print("Saved as: Figure3_Combined_Dec.png")

def generate_figure8():
    """
    Generate Figure 8: Attractor vs Cell Type Heatmap
    - Y-axis: Attractors
    - X-axis: Cell Types
    - Shows the distribution of cell types across attractors
    """
    
    print("\n" + "="*60)
    print("Generating Figure 8: Attractor vs Cell Type")
    print("="*60)
    
    # ========== Load Trajectories ==========
    print("\n[1/4] Loading trajectories...")
    trajectories = read_trajectories(INPUT_FILES["trajectories"])
    cleaned_trajectories = {cell: clean_trajectory(traj) for cell, traj in trajectories.items()}
    
    # ========== Load Extra Metadata with Cell Types ==========
    print("[2/4] Loading cell type metadata...")
    
    # Read HIV_metadata_extra.txt
    # Format: "619"	"GCTCTGTTCGGCGGTT.Participant_6"	"AS+"	7	"T cells CD8 - 2"
    cell_barcode_to_type = {}
    cell_barcode_to_condition = {}
    
    with open(INPUT_FILES["metadata_extra"], 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse tab-separated values with quotes
            parts = line.split('\t')
            if len(parts) >= 5:
                # Remove quotes from each part
                cell_idx = parts[0].strip('"')
                cell_barcode = parts[1].strip('"')
                condition = parts[2].strip('"')
                cell_type = parts[4].strip('"')
                
                cell_barcode_to_type[cell_barcode] = cell_type
                cell_barcode_to_condition[cell_barcode] = condition
    
    print(f"  Loaded {len(cell_barcode_to_type)} cells with cell type info")
    
    # Get unique cell types
    unique_cell_types = sorted(set(cell_barcode_to_type.values()))
    print(f"  Found {len(unique_cell_types)} unique cell types: {unique_cell_types}")
    
    # ========== Find Attractors for Each Cell ==========
    print("[3/4] Mapping cells to attractors...")
    
    # Build mapping from trajectory cell key to barcode
    # Trajectory keys are like "Cell_0", "Cell_1", etc.
    # We need to match with barcodes from metadata
    
    attractor_celltype_counts = {}  # {attractor: {cell_type: count}}
    attractor_cells = {}  # {attractor: [cell_barcodes]}
    
    # First, create a mapping from cell index to barcode
    cell_idx_to_barcode = {}
    with open(INPUT_FILES["metadata_extra"], 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                cell_idx = parts[0].strip('"')
                cell_barcode = parts[1].strip('"')
                cell_idx_to_barcode[cell_idx] = cell_barcode
    
    # Process each cell trajectory
    matched_cells = 0
    for cell_key, traj in cleaned_trajectories.items():
        # Extract cell index from key (e.g., "Cell_619" -> "619")
        try:
            cell_idx = cell_key.split('_')[1]
        except:
            continue
        
        # Get the attractor (last state in cleaned trajectory)
        if len(traj) > 0:
            attractor = traj[-1]
            
            # Get barcode for this cell
            if cell_idx in cell_idx_to_barcode:
                barcode = cell_idx_to_barcode[cell_idx]
                
                # Get cell type for this barcode
                if barcode in cell_barcode_to_type:
                    cell_type = cell_barcode_to_type[barcode]
                    
                    # Count
                    if attractor not in attractor_celltype_counts:
                        attractor_celltype_counts[attractor] = {}
                        attractor_cells[attractor] = []
                    
                    if cell_type not in attractor_celltype_counts[attractor]:
                        attractor_celltype_counts[attractor][cell_type] = 0
                    
                    attractor_celltype_counts[attractor][cell_type] += 1
                    attractor_cells[attractor].append(barcode)
                    matched_cells += 1
    
    print(f"  Matched {matched_cells} cells to attractors")
    print(f"  Found {len(attractor_celltype_counts)} unique attractors")
    
    if len(attractor_celltype_counts) == 0:
        print("Warning: No cells matched. Check cell index format.")
        return
    
    # ========== Create Heatmap ==========
    print("[4/4] Creating heatmap...")
    
    # Get sorted list of attractors and cell types
    attractors = sorted(attractor_celltype_counts.keys())
    cell_types = sorted(unique_cell_types)
    
    # Create count matrix
    count_matrix = np.zeros((len(attractors), len(cell_types)))
    
    for i, attractor in enumerate(attractors):
        for j, cell_type in enumerate(cell_types):
            count_matrix[i, j] = attractor_celltype_counts[attractor].get(cell_type, 0)
    
    # Normalize by row (per attractor) to show proportion
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    proportion_matrix = count_matrix / row_sums
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(attractors) * 0.3)))
    
    # Plot heatmap
    im = ax.imshow(proportion_matrix, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Cells', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Set axis labels
    ax.set_xticks(np.arange(len(cell_types)))
    ax.set_yticks(np.arange(len(attractors)))
    ax.set_xticklabels(cell_types, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels([f'Attractor {a}' for a in attractors], fontsize=10)
    
    ax.set_xlabel('Cell Type', fontsize=16, fontweight='bold')
    ax.set_ylabel('Attractor', fontsize=16, fontweight='bold')
    ax.set_title('Cell Type Distribution Across Attractors', fontsize=18, fontweight='bold')
    
    # Add cell counts as text annotations
    for i in range(len(attractors)):
        for j in range(len(cell_types)):
            count = int(count_matrix[i, j])
            if count > 0:
                text_color = 'white' if proportion_matrix[i, j] > 0.5 else 'black'
                ax.text(j, i, str(count), ha='center', va='center', 
                       fontsize=8, color=text_color, fontweight='bold')
    
    #plt.tight_layout()
    plt.savefig("Figure8A.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Figure8A.png")
    
    # ========== Create Bar Chart Version ==========
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Stack bar chart showing cell type composition per attractor
    bottom = np.zeros(len(attractors))
    
    # Color map for cell types
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types)))
    
    for j, cell_type in enumerate(cell_types):
        counts = [attractor_celltype_counts[a].get(cell_type, 0) for a in attractors]
        ax.barh(np.arange(len(attractors)), counts, left=bottom, 
                label=cell_type, color=colors[j], edgecolor='black', linewidth=0.5)
        bottom += counts
    
    ax.set_yticks(np.arange(len(attractors)))
    ax.set_yticklabels([f'Attractor {a}' for a in attractors], fontsize=10)
    ax.set_xlabel('Number of Cells', fontsize=16, fontweight='bold')
    ax.set_ylabel('Attractor', fontsize=16, fontweight='bold')
    ax.set_title('Cell Type Composition by Attractor', fontsize=18, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    #plt.tight_layout()
    plt.savefig("Figure8B.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Figure8B.png")
    
    # ========== Combine Figure 8 ==========
    print("\nCombining Figure 8A and 8B...")
    
    images = ["Figure8A.png", "Figure8B.png"]
    missing_images = [img for img in images if not os.path.exists(img)]
    if missing_images:
        print(f"Warning: Missing images: {missing_images}")
        return
    
    img_data = [mpimg.imread(f) for f in images]
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(1, 2, figure=fig, wspace=0.1)
    
    # Left: Figure 8A (Heatmap)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.imshow(img_data[0])
    ax_left.axis('off')
    ax_left.text(0.02, 1.02, 'a.', transform=ax_left.transAxes,
                fontsize=24, fontweight='bold', color='black',
                ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Right: Figure 8B (Bar chart)
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.imshow(img_data[1])
    ax_right.axis('off')
    ax_right.text(0.02, 1.02, 'b.', transform=ax_right.transAxes,
                 fontsize=24, fontweight='bold', color='black',
                 ha='left', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    #plt.tight_layout()
    plt.savefig("Figure8_Combined.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Total attractors: {len(attractors)}")
    print(f"Total cell types: {len(cell_types)}")
    print(f"Total matched cells: {matched_cells}")
    
    print("\nCells per attractor:")
    for attractor in attractors:
        total = sum(attractor_celltype_counts[attractor].values())
        print(f"  Attractor {attractor}: {total} cells")
    
    print("\nFigure 8 generation complete!")
    print("Individual figures: Figure8A.png, Figure8B.png")
    print("Combined panel: Figure8_Combined.png")

def main():
    global selected_gene
    selected_gene = "GSK3B"

    """Main execution function"""
    print("Generating individual figures 3A-3F...")
    generate_all_figures3()
    
    print("\nCombining Figure 3 panels...")
    combine_figures()
    
    print("\nGenerating Figure 4...")
    generate_figure4()
    
    print("\nGenerating Figure 5...")
    generate_figure5()

    print("\nGenerating Figure 6...")
    generate_figure6()
    
    print("\nGenerating Figure 7 (Sankey)...")
    
    generate_figure7(selected_gene)

    print("\nGenerating the new verions of Figure 6 which is the combination of Fig 6 and Fig 7")
    generate_new_figure6(
    "figure7a.png",
    "figure7b.png",
    "figure6a.png",
    "figure6b.png",
    output_path="Figure6_revised.png")
    
    #print("\nGenerating Figure 8...")
    #generate_figure8()

    
    print("\n=== Process Complete ===")
    print("Figure 3: Figure3A.png through Figure3FOrg.png")
    print("Figure 3 Combined: Figure3_Combined_Dec.png")
    print("Figure 4: Figure4A.png, Figure4B.png")
    print("Figure 4 Combined: Figure4_Combined.png")
    print("Figure 5: Figure5A.png, Figure5B.png, Figure5C.png")
    print("Figure 5 Combined: Figure5_Combined.png")
    print("Figure 6: Figure6A.png, Figure6B.png")
    print("Figure 6 (New Version)")


if __name__ == "__main__":
    main()
