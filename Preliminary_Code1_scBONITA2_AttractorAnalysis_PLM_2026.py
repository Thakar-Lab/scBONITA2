"""
Trajectory Analysis Script by Pancy Lwin-Munger
2025
Used for the publication - Harnessing the single-cell RNA seq data for digital twins

Analyzes cell trajectory data from CSV files, computes state statistics,
and generates t-SNE visualizations.
"""

import os
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE


def read_trajectory_files(trajectory_dir: Path) -> tuple[dict, dict]:
    """
    Read trajectory CSV files and extract unique states.
    
    Args:
        trajectory_dir: Directory containing trajectory CSV files
        
    Returns:
        Tuple of (columns_dict mapping state names to data, trajectory_dict mapping filenames to state sequences)
    """
    columns_dict = {}
    trajectory_dict = {}
    state_counter = 0
    
    # Sort files for consistent ordering
    # Search in the provided directory (not parent)
    trajectory_files = sorted([
        f for f in trajectory_dir.iterdir()
        if f.name.startswith("cell_") and f.name.endswith("_trajectory.csv")
    ])
    
    print(f"Searching in: {trajectory_dir}")
    print(f"Found {len(trajectory_files)} trajectory files")
    
    for filepath in trajectory_files:
        columns = []
        
        with open(filepath, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                # Read columns excluding the first one (index 0)
                for i, value in enumerate(row[1:], start=0):
                    if len(columns) <= i:
                        columns.append([])
                    try:
                        columns[i].append(int(value))
                    except ValueError:
                        print(f"Warning: Skipping non-numeric value '{value}' in {filepath.name}")
        
        # Build trajectory from unique states
        single_trajectory = []
        for column in columns:
            column_tuple = tuple(column)
            
            # Check if this state already exists
            existing_state = None
            for state_name, state_data in columns_dict.items():
                if state_data == column_tuple:
                    existing_state = int(state_name.split()[1])
                    break
            
            if existing_state is None:
                state_counter += 1
                columns_dict[f"State {state_counter}"] = column_tuple
                single_trajectory.append(state_counter)
            else:
                single_trajectory.append(existing_state)
        
        # Store trajectory (remove .csv extension)
        trajectory_dict[filepath.stem] = single_trajectory
    
    return columns_dict, trajectory_dict


def compute_state_statistics(trajectory_dict: dict) -> tuple[dict, dict]:
    """
    Compute occurrence counts and probabilities for each state.
    
    Args:
        trajectory_dict: Dictionary mapping filenames to state sequences
        
    Returns:
        Tuple of (state_occurrence counts, state_probability)
    """
    state_occurrence = defaultdict(int)
    
    for trajectory in trajectory_dict.values():
        for state in trajectory:
            state_occurrence[state] += 1
    
    total_states = sum(state_occurrence.values())
    state_probability = {
        state: count / total_states 
        for state, count in state_occurrence.items()
    }
    
    return dict(state_occurrence), state_probability


def save_results(output_dir: Path, columns_dict: dict, trajectory_dict: dict,
                 state_occurrence: dict, state_probability: dict) -> None:
    """Save analysis results to text files."""
    
    # Save trajectories
    with open(output_dir / "Combined_Trajectory.txt", 'w') as f:
        sorted_keys = sorted(trajectory_dict.keys(), key=lambda x: int(x.split('_')[1]))
        for key in sorted_keys:
            f.write(f"{key}: {trajectory_dict[key]}\n")
    
    # Save state occurrences
    with open(output_dir / "Combined_State_Occurrence.txt", 'w') as f:
        for state in sorted(state_occurrence.keys()):
            f.write(f"State {state}: {state_occurrence[state]}\n")
    
    # Save state probabilities
    with open(output_dir / "Combined_State_Probability.txt", 'w') as f:
        for state in sorted(state_probability.keys()):
            f.write(f"State {state} Probability: {state_probability[state]:.6f}\n")
    
    # Save unique states
    with open(output_dir / "Combined_Unique_States.txt", 'w') as f:
        for key, value in columns_dict.items():
            f.write(f"{key}: {value}\n")


def create_tsne_plots(columns_dict: dict, state_probability: dict, 
                      output_dir: Path, random_state: int = 42) -> None:
    """
    Create 2D and 3D t-SNE visualizations of unique states.
    
    Args:
        columns_dict: Dictionary mapping state names to data
        state_probability: Dictionary mapping state numbers to probabilities
        output_dir: Directory to save plots
        random_state: Random seed for reproducibility
    """
    state_labels = list(columns_dict.keys())
    unique_states = list(columns_dict.values())
    
    # Convert to numpy array
    data = np.array(unique_states)
    n_samples = len(data)
    
    if n_samples < 2:
        print("Warning: Not enough samples for t-SNE visualization (need at least 2)")
        return
    
    # Adjust perplexity based on sample size
    # Perplexity must be less than n_samples
    perplexity = min(30, max(5, n_samples - 1))
    
    print(f"Running t-SNE with {n_samples} samples and perplexity={perplexity}")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    tsne_results = tsne.fit_transform(data)
    
    # Calculate marker sizes based on probabilities
    sizes = [
        state_probability[int(label.split()[1])] * 1000 
        for label in state_labels
    ]
    
    # Get probabilities for coloring
    z_values = [state_probability[int(label.split()[1])] for label in state_labels]
    
    # Custom colormap
    blues_cmap = LinearSegmentedColormap.from_list(
        "custom_blues", ["#add8e6", "#00008b"]
    )
    
    # 2D Plot
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    scatter1 = ax1.scatter(
        tsne_results[:, 0], 
        tsne_results[:, 1], 
        s=sizes, 
        c=z_values,
        cmap=blues_cmap,
        marker='s',
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    ax1.set_title("t-SNE Plot of Unique States", fontsize=14)
    ax1.set_xlabel("t-SNE Component 1", fontsize=12)
    ax1.set_ylabel("t-SNE Component 2", fontsize=12)
    cbar1 = fig1.colorbar(scatter1, ax=ax1)
    cbar1.set_label('State Probability', fontsize=11)
    plt.tight_layout()
    fig1.savefig(output_dir / "t-SNE_Plot.png", dpi=150)
    plt.close(fig1)
    print(f"Saved: {output_dir / 't-SNE_Plot.png'}")
    
    # 3D Plot
    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111, projection='3d')
    scatter2 = ax2.scatter(
        tsne_results[:, 0], 
        tsne_results[:, 1], 
        z_values,
        s=sizes, 
        c=z_values, 
        cmap=blues_cmap, 
        marker='s',
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    ax2.set_title("3D Plot of Unique States with State Probabilities", fontsize=14)
    ax2.set_xlabel("t-SNE Component 1", fontsize=11)
    ax2.set_ylabel("t-SNE Component 2", fontsize=11)
    ax2.set_zlabel("State Probability", fontsize=11)
    cbar2 = fig2.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label('State Probability', fontsize=11)
    plt.tight_layout()
    fig2.savefig(output_dir / "3D_Plot.png", dpi=150)
    plt.close(fig2)
    print(f"Saved: {output_dir / '3D_Plot.png'}")


def main():
    """Main entry point for trajectory analysis."""
    print("=" * 50)
    print("Trajectory Analysis")
    print("=" * 50)
    
    # Set input directory here
    input_dir = Path(r"C:\Users\plwin\scBONITA2_accelerated_PL\2025PastRuns\scBONITA_output\trajectories\tutorial_dataset_hsaRWNetworknew.graphml\text_files\cell_trajectories")
    
    if not input_dir.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return
    
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        return
    
    # Set output directory (same as input)
    output_dir = input_dir
    
    # Read trajectory files
    print("\nReading trajectory files...")
    columns_dict, trajectory_dict = read_trajectory_files(input_dir)
    
    if not trajectory_dict:
        print("Error: No trajectory files found!")
        print(f"Looking for 'cell_*_trajectory.csv' files in: {input_dir}")
        return
    
    print(f"Found {len(trajectory_dict)} trajectory files")
    print(f"Identified {len(columns_dict)} unique states")
    
    # Compute statistics
    print("\nComputing state statistics...")
    state_occurrence, state_probability = compute_state_statistics(trajectory_dict)
    
    # Save results
    print("\nSaving results...")
    save_results(output_dir, columns_dict, trajectory_dict, 
                 state_occurrence, state_probability)
    
    # Create visualizations
    print("\nGenerating t-SNE visualizations...")
    create_tsne_plots(columns_dict, state_probability, output_dir)
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
