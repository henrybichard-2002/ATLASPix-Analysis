# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 18:47:32 2025

@author: henry
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_atlaspix_geometry(num_pixels=372, pitch_x=1.0, layer_y_sep=0.5):
    """
    Generates the 3D wire geometry based on the ATLASPix3 folding scheme.

    Each wire is a list of segments, where each segment is ((x1,y1,z1), (x2,y2,z2)).
    The coordinate system is:
    - x: horizontal position across the column
    - y: metal layer (0 for M4, 1 for M5, etc.)
    - z: vertical position along the column length

    Args:
        num_pixels (int): Total number of pixels/wires in the column.
        pitch_x (float): Horizontal distance between adjacent wires.
        layer_y_sep (float): Vertical distance between metal layers.

    Returns:
        list: A list of wires, where each wire is a list of its segments.
    """
    wires = []
    # These parameters are based on the diagram's layout
    fold_point1 = 124
    fold_point2 = 248
    column_height = 100.0 # Arbitrary physical height for the pixel section
    buffer_height = 50.0  # Arbitrary height for the buffer section
    
    # Layer assignments (simplified)
    # Wires going "up" are on one layer, "down" are on another
    layer_up = 0 * layer_y_sep   # e.g., Metal4
    layer_down = 1 * layer_y_sep # e.g., Metal5

    for i in range(num_pixels):
        wire_segments = []
        
        # Determine the horizontal position based on the pixel index
        # This models the left/right grouping and folding
        if 0 <= i < fold_point1:
            # Starts on the left, goes up, comes down on the right
            x_start = i * pitch_x
            x_return = (fold_point2 - 1 - i) * pitch_x
        elif fold_point1 <= i < fold_point2:
            # Starts on the right, goes up, comes down on the left
            x_start = i * pitch_x
            x_return = (fold_point2 - 1 - i) * pitch_x
        else: # 248 <= i < 372
            # This is a second, mirrored group. For simplicity, we model it
            # as a continuation, but a real model might have a separate fold.
            j = i - fold_point2
            x_start = (fold_point1 + j) * pitch_x
            x_return = (fold_point1 - 1 - j) * pitch_x

        # 1. Path up through the pixels
        start_point_up = (x_start, layer_up, 0)
        end_point_up = (x_start, layer_up, column_height)
        wire_segments.append((start_point_up, end_point_up))

        # 2. Path down through the hit buffers (return path)
        start_point_down = (x_return, layer_down, column_height)
        end_point_down = (x_return, layer_down, column_height - buffer_height)
        wire_segments.append((start_point_down, end_point_down))
        
        wires.append(wire_segments)
        
    return wires


def calculate_coupling_matrix(wires, k_side=1.0, k_vert=0.5):
    """
    Calculates the N x N coupling capacitance matrix.

    Args:
        wires (list): The list of wire geometries.
        k_side (float): Proportionality constant for side-by-side coupling.
        k_vert (float): Proportionality constant for vertical (layer-to-layer) coupling.

    Returns:
        np.ndarray: The symmetric N x N coupling matrix.
    """
    num_wires = len(wires)
    C_matrix = np.zeros((num_wires, num_wires))

    for i in range(num_wires):
        for j in range(i + 1, num_wires):
            total_coupling = 0
            # Compare every segment of wire i with every segment of wire j
            for seg_i in wires[i]:
                for seg_j in wires[j]:
                    p1_i, p2_i = seg_i
                    p1_j, p2_j = seg_j

                    # Check for parallel segments along Z
                    # This is the dominant coupling factor
                    is_parallel_z = (p1_i[0] != p1_j[0] or p1_i[1] != p1_j[1]) and \
                                    (p1_i[1] == p1_j[1] or p1_i[0] == p1_j[0]) # Same layer or vertically aligned
                    
                    if is_parallel_z:
                        # Find overlapping z-range
                        z_overlap_min = max(min(p1_i[2], p2_i[2]), min(p1_j[2], p2_j[2]))
                        z_overlap_max = min(max(p1_i[2], p2_i[2]), max(p1_j[2], p2_j[2]))
                        
                        parallel_length = max(0, z_overlap_max - z_overlap_min)

                        if parallel_length > 0:
                            dist_x = abs(p1_i[0] - p1_j[0])
                            dist_y = abs(p1_i[1] - p1_j[1])
                            
                            distance_sq = dist_x**2 + dist_y**2
                            
                            # Different k-factor depending on side-by-side or vertical
                            k = k_side if dist_y == 0 else k_vert
                            
                            # Simplified model: C is proportional to L/d^2
                            # A 1/d model is also common. 1/d^2 emphasizes closer wires.
                            coupling = k * parallel_length / distance_sq
                            total_coupling += coupling
            
            C_matrix[i, j] = total_coupling
            C_matrix[j, i] = total_coupling # Matrix is symmetric
            
    return C_matrix


def simulate_crosstalk(C_matrix, aggressor_idx, V_swing=1.0, C_gnd=5.0):
    """
    Calculates the noise voltage on victim lines using a capacitive divider model.

    Args:
        C_matrix (np.ndarray): The coupling matrix.
        aggressor_idx (int): The index of the switching wire.
        V_swing (float): The voltage swing on the aggressor line.
        C_gnd (float): Assumed capacitance of each line to ground (arbitrary units).
                       This is a critical parameter for the magnitude of the crosstalk.

    Returns:
        np.ndarray: An array of induced voltages on all other lines.
    """
    num_wires = C_matrix.shape[0]
    victim_voltages = np.zeros(num_wires)
    
    # Get the coupling capacitances between the aggressor and all victims
    C_coupling = C_matrix[aggressor_idx, :]
    
    # Using the voltage divider formula: V_victim = V_swing * C_ij / (C_ij + C_j_gnd)
    # We assume C_j_gnd is constant for all j
    victim_voltages = V_swing * C_coupling / (C_coupling + C_gnd)
    
    # Noise on the aggressor itself is 0 in this model
    victim_voltages[aggressor_idx] = 0
    
    return victim_voltages

# --- Main Execution ---

# 1. Define Model Parameters
NUM_PIXELS = 372
WIRE_PITCH = 1.0          # Arbitrary units, e.g., micrometers
LAYER_SEPARATION = 0.8    # Vertical distance between metal layers
CAP_TO_GROUND = 10.0      # Capacitance to GND in arbitrary units (e.g., fF)
AGGRESSOR_PIXEL = 60      # Let's see the effect of pixel 60 switching

# 2. Generate the wire geometry based on the diagram
print("Generating wire geometry...")
wires = generate_atlaspix_geometry(NUM_PIXELS, WIRE_PITCH, LAYER_SEPARATION)

# 3. Calculate the coupling capacitance matrix
print("Calculating coupling matrix...")
# We use a larger constant for side-by-side coupling than for vertical coupling
C_matrix = calculate_coupling_matrix(wires, k_side=1.0, k_vert=0.5)

# 4. Simulate crosstalk from a single aggressor
print(f"Simulating crosstalk from aggressor pixel {AGGRESSOR_PIXEL}...")
crosstalk_voltages = simulate_crosstalk(C_matrix, AGGRESSOR_PIXEL, C_gnd=CAP_TO_GROUND)

# 5. Visualize the results
print("Plotting results...")
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle(f'Crosstalk Analysis for a {NUM_PIXELS}-pixel Column', fontsize=16)

# Plot 1: The Coupling Capacitance Matrix
im = axes[0].imshow(C_matrix, cmap='inferno')
axes[0].set_title(r'Coupling Capacitance Matrix ($C_{ij}$)')
axes[0].set_xlabel('Victim Pixel Index')
axes[0].set_ylabel('Aggressor Pixel Index')
fig.colorbar(im, ax=axes[0], label='Coupling Strength (arbitrary units)')

# Plot 2: Crosstalk from the specific aggressor
axes[1].bar(range(NUM_PIXELS), crosstalk_voltages * 100) # Show as percentage
axes[1].set_title(f'Induced Noise Voltage on Victims from Aggressor Pixel {AGGRESSOR_PIXEL}')
axes[1].set_xlabel('Victim Pixel Index')
axes[1].set_ylabel('Crosstalk Noise (% of V_swing)')
axes[1].axvline(AGGRESSOR_PIXEL, color='r', linestyle='--', label=f'Aggressor {AGGRESSOR_PIXEL}')
axes[1].legend()
axes[1].set_xlim(0, NUM_PIXELS)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



