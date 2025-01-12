"""
This file visualizes the results from the NSGA-II

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
import torch


def count_channels(gene):
    """
    Count the number of actual channels in the given array.

    Parameters:
    - channels: numpy array of shape (n_channels, n_points, 2),
                where each channel consists of its control points.

    Returns:
    - count: Number of actual channels (channels where the first two points are not identical).
    """
    count = 0
    for channel in gene:
        # Check if the first two points of the channel are identical
        if not np.array_equal(channel[1], channel[2]):
            count += 1
    return count

def catmull_rom_spline(channel, num_cam_points=1000):
    """
    Evaluates a 2D Catmull-Rom spline given control points.
    The spline is computed between the first permanent point and the second permanent point.
    Straight lines are used from the start point to the first permanent point,
    and from the second permanent point to the end point.
    
    :param channel: Array of control points of shape (N, 2).
    :param num_cam_points: Total number of points to generate along the entire path.
    :return: Array of points along the path of shape (num_points, 2).
    """
    
    # Define permanent control points
    start_point = np.array([-17, 50])
    first_permanent_point = np.array([8, 50])
    second_permanent_point = np.array([172, 50])
    end_point = np.array([197, 50])
    
    y_min = 5
    y_max = 96
    x_min = 5
    x_max = 175
    # --- Generate the spline between the first and second permanent points ---
    
    # Combine the control points for the spline
    spline_control_points = np.vstack([first_permanent_point, channel, second_permanent_point])
    
    # Number of points for the spline segment
    num_spline_points = num_cam_points - 40  # Adjust as needed
    
    # Create parameter values for control points (uniform spacing)
    t = np.linspace(0, 1, len(spline_control_points))
    
    # Extract x and y coordinates
    x = spline_control_points[:, 0]
    y = spline_control_points[:, 1]
    
    # Generate the Catmull-Rom spline
    spline_x = CubicSpline(t, x, bc_type='clamped')
    spline_y = CubicSpline(t, y, bc_type='clamped')
    
    # Generate points along the spline
    t_values = np.linspace(0, 1, num_spline_points)
    spline_curve_x = spline_x(t_values)
    spline_curve_y = spline_y(t_values)
    
    # Format the spline points
    spline_curve_points = np.vstack([spline_curve_x, spline_curve_y]).T
    
    # --- Generate the straight line from start point to first permanent point ---
    
    num_start_line_points = 20  # Adjust as needed
    start_line_x = np.linspace(start_point[0], first_permanent_point[0], num_start_line_points)
    start_line_y = np.linspace(start_point[1], first_permanent_point[1], num_start_line_points)
    start_line_points = np.vstack([start_line_x, start_line_y]).T
    
    # --- Generate the straight line from second permanent point to end point ---
    
    num_end_line_points = 20  # Adjust as needed
    end_line_x = np.linspace(second_permanent_point[0], end_point[0], num_end_line_points)
    end_line_y = np.linspace(second_permanent_point[1], end_point[1], num_end_line_points)
    end_line_points = np.vstack([end_line_x, end_line_y]).T
    
    # --- Clip the y-values to be within the specified range ---

    spline_curve_points[:, 1] = np.clip(spline_curve_points[:, 1], y_min, y_max)
    spline_curve_points[:, 0] = np.clip(spline_curve_points[:, 0], x_min, x_max)

    # --- Combine all segments ---
    
    curve_points = np.vstack([start_line_points, spline_curve_points, end_line_points])
    
    return curve_points

def visualize_cam_curve(gene, heat_source='h2', save=False, title=None, ax=None):
    """
    Visualizes cooling channels on a plate with inlet and outlet rectangles
    and an additional set of colored 2D patches.

    If ax is provided, all the plots will be drawn on that Axes object.
    Otherwise, a new figure will be created.

    """
    num_channels = count_channels(gene)
    start_point = (-17, 50)
    first_permanent_point = (8, 50)
    second_permanent_point = (172, 50)
    end_point = (197, 50)

    # If no ax is passed, create a new figure and axis
    if ax is None:
        plt.figure(figsize=(10, 5))
        ax = plt.gca()

    # Draw the plate
    plate_rect = patches.Rectangle((0, 0), 181, 101, color='lightgray', alpha=0.7)
    ax.add_patch(plate_rect)

    # Add inlet
    inlet_rect = patches.Rectangle((-20, 45), 20, 10, color='lightgray', alpha=0.7)
    ax.add_patch(inlet_rect)

    # Add outlet
    outlet_rect = patches.Rectangle((181, 45), 20, 10, color='lightgray', alpha=0.7)
    ax.add_patch(outlet_rect)

    # Plot the cooling channels
    for i in range(num_channels):
        channel = gene[i]
        control_points = np.vstack([start_point, *channel, second_permanent_point, end_point])
        curve_points = catmull_rom_spline(channel)

        control_x, control_y = control_points.T
        curve_x, curve_y = curve_points.T

        ax.plot(curve_x, curve_y, color='blue', linewidth=4,
                label='Cooling channel path' if i == 0 else "")
        ax.scatter(control_x, control_y, color='red', zorder=5,
                   label='Random points' if i == 0 else "")

    # Calculate the size of each grid cell
    plate_x = [0, 181]
    plate_y = [0, 101]

    x_start = plate_x[0]
    x_end =  plate_x[1]
    y_start = plate_y[0]
    y_end = plate_y[1]
    cell_width = (x_end - x_start) / 4
    cell_height = (y_end - y_start) / 3

    # Spacing factor to reduce square size and create space between them
    spacing_factor = 0.5  # Squares occupy 50% of the grid cell

    # New square sizes
    new_square_width = cell_width * spacing_factor
    new_square_height = cell_height * spacing_factor

    # Now add the colored 2D patches
    yellow = 'red'
    orange = '#FF8C00'
    red = '#FFD700'
    
    if heat_source == 'h1':
        colors_list = [red, orange, yellow, red, red, yellow, red, red, orange, orange, 'red', orange]
    elif heat_source == 'h2':
        colors_list = [yellow, orange, red, yellow, red, red, orange, red, red, orange, red, orange]
    elif heat_source == 'h3':
        colors_list = [red, orange, yellow, red, red, yellow, red, red, orange, orange, 'red', orange]
    elif heat_source == 'h4':
        colors_list = [red, orange, yellow, red, red, yellow, red, red, orange, orange, 'red', orange]
    
    x_centers = [x_start + (i + 0.5) * cell_width for i in range(4)]
    y_centers = [y_start + (j + 0.5) * cell_height for j in range(3)]
    color_idx = 0

    for i in range(4):
        for j in range(3):
            x_center = x_centers[i]
            y_center = y_centers[j]

            # Calculate the corners of the rectangle
            x0 = x_center - new_square_width / 2
            y0 = y_center - new_square_height / 2

            color = colors_list[color_idx % len(colors_list)]
            color_idx += 1

            rect = patches.Rectangle((x0, y0), new_square_width, new_square_height,
                                     color=color, alpha=0.35, edgecolor='none')
            ax.add_patch(rect)

    # Adjust plot limits
    ax.set_xlim((-20, 200))
    ax.set_ylim((0, 101))

    if title is not None:
        ax.set_title(title, fontsize=14)

    # Optionally hide the axes for a clean look
    ax.axis('off')

    # If we're not saving and no external ax is given, show the plot directly
    if save and ax is None:
        plt.savefig('Cooling_Channel_Geometry.png', dpi=300, bbox_inches='tight', pad_inches=0)
    elif ax is None:
        plt.show()


def create_curve_grid(gene, thickness=3, grid_shape=(181, 101)):

    """
    Create a binary grid representing the curves for multiple channels.

    Inputs:
    - gene: Array of control points for multiple channels.
    - thickness: Thickness of the curve on the grid.
    - grid_shape: Dimensions of the binary grid.

    Output:
    - grid: Binary grid with curves marked (1) and background (0).
    """

    grid = np.zeros(grid_shape, dtype=int)
    num_channels = gene.shape[0]

    for i in range(num_channels):
        channel = gene[i]
        curve_points = catmull_rom_spline(channel)

        if curve_points is not None:
        # Round curve points to the nearest integer and ensure they are within bounds
            for x, y in np.round(curve_points).astype(int):
                if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1]:
                    # Mark a small area around the curve point as "material" (value 1) using logical OR
                    x_min, x_max = max(0, x - thickness), min(grid_shape[0], x + thickness + 1)
                    y_min, y_max = max(0, y - thickness), min(grid_shape[1], y + thickness + 1)
                    grid[x_min:x_max, y_min:y_max] |= 1
        else:
            return None
    return grid

def visualize_curve_grid(grid):

    """
    Visualize the binary grid representation of the curve.

    Input:
    - grid: Binary grid of shape (width, height).

    Output:
    - A plot displaying the binary grid.
    """

    plt.figure(figsize=(10, 5))
    plt.imshow(grid.T, cmap='gray', origin='lower')
    plt.title('Binary Grid Representation of Bezier Curve')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.colorbar(label='Material (1) / No Material (0)')
    plt.show()





# Load the data
with open('pareto_front_objectives.pkl', 'rb') as f:
    results = pickle.load(f)
with open('pareto_front_geometries.pkl', 'rb') as f:
    geometries = pickle.load(f)

geometries = [geometries[i].reshape(8, 8, 2) for i in range(len(geometries))]

# Extract objective values
objective1 = results[:, 0]  # e.g., first objective
objective2 = results[:, 1]  # e.g., second objective

best_idx_obj1 = np.argmin(results[:, 0])
best_idx_obj2 = np.argmin(results[:, 1])
best_idx_tradeoff = 29

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M1 = torch.load('M1_performance_predictor.pt', map_location=torch.device('cpu'))
M1.to(device)
M1.eval()

inlet_velocity = torch.load('Data/v1.pt').to(device)        #0.1 m/s
heat_source = torch.load('Data/h2.pt').to(device)
heat_source = heat_source.unsqueeze(0).unsqueeze(0)
inlet_velocity = inlet_velocity.unsqueeze(0).unsqueeze(0)


for i in range(len(objective1)):
    print("Objective Values for :", i, results[i])
    visualize_cam_curve(geometries[i], title=f'Geometry #{i}: {results[i]}')


# Find the solution with the best (minimum) value for the first objective
best_idx_obj1 = np.argmin(results[:, 0])
print("Best Solution for Objective 1:")
visualize_cam_curve(geometries[best_idx_obj1], title=f'Maximum Temperature: {results[best_idx_obj1][0]:.2f} K, Pressure Drop: {results[best_idx_obj1][1]:.2f} Pa')
print("Objective Values:", results[best_idx_obj1])

# Find the solution with the best (minimum) value for the second objective
best_idx_obj2 = np.argmin(results[:, 1])
print("Best Solution for Objective 2:")
visualize_cam_curve(geometries[best_idx_obj2], title=f'Maximum Temperature: {results[best_idx_obj2][0]:.2f} K, Pressure Drop: {results[best_idx_obj2][1]:.2f} Pa')
print("Objective Values:", results[best_idx_obj2])

# Find a balanced trade-off (e.g., minimum sum of objectives)
best_idx_tradeoff = np.argmin(results.sum(axis=1))
b_index = 29
print("Best Trade-Off Solution:")
visualize_cam_curve(geometries[b_index], title=f'Maximum Temperature: {results[b_index][0]:.2f} K, Pressure Drop: {results[b_index][1]:.2f} Pa')
print("Objective Values:", results[b_index])


