import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
import random
import torch 
from scipy.interpolate import CubicSpline
import pickle
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Custom Problem Definition
# ---------------------------

class ChannelGeometryProblem(Problem):
    def __init__(self, M1, heat_source, inlet_velocity, n_channels=8, num_points=8, plate_width=180, plate_height=100):
        self.n_channels = n_channels
        self.points_per_channel = num_points

        self.M1 = M1
        self.heat_source = heat_source
        self.inlet_velocity = inlet_velocity
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Each channel has (points_per_channel) points, each point has (x,y).
        # Total variables = n_channels * points_per_channel * 2
        n_var = self.n_channels * self.points_per_channel * 2
        
        # Bounds: for example, x in [0, plate_width], y in [0, plate_height]
        xl = np.zeros(n_var)
        xu = np.zeros(n_var)
        for i in range(n_var):
            # Even indices -> x-coordinate
            if i % 2 == 0:
                xl[i] = 0
                xu[i] = plate_width
            # Odd indices -> y-coordinate
            else:
                xl[i] = 0
                xu[i] = plate_height
        
        # Two objectives: for example:
        # f1 = (negative of efficiency) we want to minimize negative efficiency → maximize efficiency
        # f2 = pressure_drop (want to minimize)
        # Suppose no explicit constraints for now.
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu)

    def count_channels(self, channels):
        """
        Count the number of actual channels in the given array.

        Parameters:
        - channels: numpy array of shape (n_channels, n_points, 2),
                    where each channel consists of its control points.

        Returns:
        - count: Number of actual channels (channels where the first two points are not identical).
        """
        count = 0
        for channel in channels:
            # Check if the first two points of the channel are identical
            if not np.array_equal(channel[0], channel[1]):
                count += 1
        return count


    def catmull_rom_spline(self, channel, num_cam_points=1000):
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

    def create_curve_grid(self, gene, thickness=3, grid_shape=(181, 101)):
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
        n = self.count_channels(gene)

        for i in range(n):
            channel = gene[i]
            curve_points = self.catmull_rom_spline(channel)

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

    def fitness_score(self, population, batch_size=64):
        
        """
        Evaluate the fitness score of each gene in the population using manual mini-batching.

        Inputs:
        - population: List of genes (channels).
        - heat_source: Heat source tensor.
        - inlet_velocity: Velocity of fluid inlet tensor.
        - batch_size: Size of each mini-batch for evaluation.
        - w1, w2: Weights for temperature and pressure drop in fitness calculation.

        Output:
        - fitness_score: Tensor of fitness scores for the population.
        """

        # Ensure the model is in evaluation mode
        self.M1.eval()

        pressure_drop_min = 22.117146658070848
        pressure_drop_max = 648.5903317508491
        temperature_min = 298
        temperature_max = 486.4246826171875


        # Prepare channel geometry using list comprehension for efficiency
        channel_geometry_list = [
            self.create_curve_grid(gene).T for gene in population if self.create_curve_grid(gene) is not None
        ]

        # Convert list to tensor and move to device
        channel_geometry = torch.tensor(channel_geometry_list, dtype=torch.float32, device=self.device)
        channel_geometry = channel_geometry.unsqueeze(1)

        num_samples = channel_geometry.size(0)

        # Expand heat source and inlet velocity matrices once
        heat_source_expanded = self.heat_source.unsqueeze(0).unsqueeze(0).expand(num_samples, -1, -1, -1)
        inlet_velocity_expanded = self.inlet_velocity.unsqueeze(0).unsqueeze(0).expand(num_samples, -1, -1, -1)

        all_pressure_drops = []
        all_temperature_preds = []

        # Process in mini-batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            batch_heat_source = heat_source_expanded[start_idx:end_idx]
            batch_channel_geometry = channel_geometry[start_idx:end_idx]
            batch_velocity_matrix = inlet_velocity_expanded[start_idx:end_idx]

            with torch.no_grad():
                # Run the model on the mini-batch
                pressure_drop_pred, temperature_pred = self.M1(
                    batch_heat_source, batch_channel_geometry, batch_velocity_matrix
                )

            all_pressure_drops.append(pressure_drop_pred)
            all_temperature_preds.append(temperature_pred)

        # Concatenate all mini-batch results
        pressure_drop_pred = torch.cat(all_pressure_drops, dim=0)
        temperature_pred = torch.cat(all_temperature_preds, dim=0)

        # Unnormalize predictions
        pressure_drop_pred_actual = pressure_drop_pred * (pressure_drop_max - pressure_drop_min) + pressure_drop_min
        temperature_pred_actual = temperature_pred * (temperature_max - temperature_min) + temperature_min

        # Calculate temperature metrics using PyTorch operations
        temperature_pred = temperature_pred.squeeze(1)
        avg_T = temperature_pred.mean(dim=(1, 2))
        Tmax = temperature_pred.amax(dim=(1, 2))
        Tmin = temperature_pred.amin(dim=(1, 2))

        max_temperature = Tmax * (temperature_max - temperature_min) + temperature_min


        return pressure_drop_pred_actual.flatten() , max_temperature


    def _evaluate(self, X, out, *args, **kwargs):

        population = [X[i].reshape(self.n_channels, self.points_per_channel, 2) for i in range(X.shape[0])]

        # Call your integrated fitness_score method
        pressure_drop, max_temperature = self.fitness_score(population)

        # Convert to numpy if needed
        if torch.is_tensor(pressure_drop):
            pressure_drop = pressure_drop.cpu().numpy()
        if torch.is_tensor(max_temperature):
            max_temperature = max_temperature.cpu().numpy()

        # Objectives
        f1 = max_temperature
        f2 = pressure_drop
        F = np.column_stack([f1, f2])
        out["F"] = F

class ChannelMutation(Mutation):
    def __init__(self, n_channels=8, n_points=8, mutation_probabilities=[0.5, 0.75, 1.0], overall_mutation_prob=0.1, sigma=5.0):
        super().__init__()
        self.sigma = sigma  # standard deviation for random mutation
        self.n_channels = n_channels
        self.n_points = n_points
        self.mutationprobabilities = mutation_probabilities
        self.overall_mutation_prob = overall_mutation_prob

    def count_channels(self, gene):
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
            if not np.array_equal(channel[0], channel[1]):
                count += 1
        return count

    def generate_random_points(self, num_points, sigma, y_min=10, y_max=90, x_max=170):
    
        random_points = []
        first_permanent_point = (8, 50)
        random_points.append(first_permanent_point)
        last_height = 50

        for i in range(num_points-1):
            x = (i+1)*x_max/num_points
            if i==0:
                y = random.randint(y_min, y_max)
            else:
                y = np.random.normal(loc=last_height, scale=sigma)

            if y>y_max:
                y = y_max
            if y<y_min:
                y = y_min
            
            last_height = y
            new_point = (x, y)
            random_points.append(new_point)

        random_points = np.array(random_points)
        random_points = random_points.reshape(1,num_points,2)
        
        return random_points

    def add_noise(self, gene, sigma_noise=10.0):
        """
        Apply Gaussian mutation to a gene.

        Inputs:
        - gene: Array of control points for the gene.
        - mutationProbability: Probability of mutating each control point.
        - sigma: Standard deviation of the Gaussian noise.

        Output:
        - mutated_gene: Mutated version of the input gene.
        """

        y_min = 8
        y_max = 92

        n = self.count_channels(gene)

        mutated_gene = gene.copy()
        for i in range(n):
            for j in range(1, self.n_points):
                # Apply Gaussian noise to the control point
                #mutated_gene[i, j, 0] += np.random.normal(0, sigma_noise)
                mutated_gene[i, j, 1] += np.random.normal(0, sigma_noise)

                if mutated_gene[i, j, 0]>y_max:
                    mutated_gene[i, j, 0] = y_max
                if mutated_gene[i, j, 1] < y_min:
                    mutated_gene[i, j, 1] = y_min
        return mutated_gene

    def channel_split(self, gene, sigma=5, sigma_noise=10.0):

        n = self.count_channels(gene)
        if n == 0:
            print("DEBUG: Invalid gene in channel_split input!")

        if n < self.n_channels:
            if n == 0:
                return gene
            else:
                second_channel = self.generate_random_points(self.n_points, sigma)
                r = random.randint(0,self.n_points-1)
                second_channel[0][:r] = gene[0][:r]
                mutated_gene = gene.copy()
                mutated_gene[n] = second_channel
        else:
            mutated_gene = gene

        n = self.count_channels(mutated_gene)
        if n == 0:
            print("DEBUG: Invalid gene in channel_split output!")

        return mutated_gene

    def channel_combine(self, gene, sigma=5, sigma_noise=10):
        
        num_channels = self.count_channels(gene)

        n = self.count_channels(gene)
        if n == 0:
            print("DEBUG: Invalid gene in channel_combine input!")

        mutated_gene = gene

        r = random.randint(0,self.n_points-1)
        r1 = random.randint(0,num_channels-1)

        if num_channels>1:
            mutated_gene[r1][:r] = mutated_gene[r1-1][:r]
        else:
            mutated_gene = self.channel_split(gene, sigma, sigma_noise)
        
        n = self.count_channels(mutated_gene)
        if n == 0:
            print("DEBUG: Invalid gene in channel_combine output!")
        return mutated_gene
    
    def _do(self, problem, X, **kwargs):
        # X is (n_individuals, n_var)
        n_individuals, n_var = X.shape

        for i in range(n_individuals):
            if np.random.rand() < self.overall_mutation_prob: 
                # Reshape into (n_channels, n_points, 2)
                gene = X[i].reshape(self.n_channels, self.n_points, 2)

                r = random.random()

                if r < self.mutationprobabilities[0]:
                    mutated_gene = self.add_noise(gene, sigma_noise=10.0)
                elif r < self.mutationprobabilities[1]:
                    mutated_gene = self.channel_split(gene, self.sigma)
                else:
                    mutated_gene = self.channel_combine(gene, self.sigma)

                # Flatten back to original shape
                X[i] = mutated_gene.reshape(n_var)

        return X

class MyCustomSampling(Sampling):
    def __init__(self, n_channels=8, num_points=8, sigma=5, y_min=10, y_max=90, x_max=170):
        """
        Initialize the custom sampling parameters.
        - sigma: Std dev for vertical positioning
        - num_points: Number of control points (excluding the fixed start point)
        - y_min, y_max, x_max: Bounds for point placement
        """
        super().__init__()
        self.sigma = sigma
        self.num_points = num_points
        self.y_min = y_min
        self.y_max = y_max
        self.x_max = x_max
        self.n_channels = n_channels

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate n_samples solutions. Each solution is defined by
        (num_points+1)*2 decision variables (x,y coordinates for each control point).
        """
        X = []
        for _ in range(n_samples):
            random_individual = self.generate_random_individual(
                self.num_points, self.sigma, 
                y_min=self.y_min, y_max=self.y_max, x_max=self.x_max
            )
            # random_points shape: (1, num_points+1, 2)
            # Flatten into a single vector of decision variables
            individual = random_individual.reshape(-1)
            X.append(individual)
        X = np.array(X)
        return X

    def count_channels(self, channels):
        """
        Count the number of actual channels in the given array.

        Parameters:
        - channels: numpy array of shape (n_channels, n_points, 2),
                    where each channel consists of its control points.

        Returns:
        - count: Number of actual channels (channels where the first two points are not identical).
        """
        count = 0
        for channel in channels:
            # Check if the first two points of the channel are identical
            if not np.array_equal(channel[0], channel[1]):
                count += 1
        return count

    def generate_random_individual(self, num_points, sigma, y_min=10, y_max=90, x_max=170):

        random_points = []
        first_permanent_point = (8, 50)
        random_points.append(first_permanent_point)
        last_height = 50

        for i in range(self.num_points-1):
            x = (i+1)*x_max/self.num_points
            if i == 0:
                y = random.randint(y_min, y_max)
            else:
                y = np.random.normal(loc=last_height, scale=self.sigma)

            y = max(min(y, y_max), y_min)
            last_height = y
            random_points.append((x, y))

        # Shape the first channel
        first_channel = np.array(random_points).reshape(1, self.num_points, 2)
        
        # Create the remaining channels as repeats of the first permanent point
        # These will be dummy channels
        dummy_channel = np.tile(np.array(first_permanent_point), (self.num_points, 1)).reshape(1, self.num_points, 2)
        #dummy_channel = first_channel.copy()

        # Concatenate the first channel with the dummy channels
        individual = np.concatenate([first_channel] + [dummy_channel]*(self.n_channels-1), axis=0)

        n=self.count_channels(individual)
        if n == 0:
            print("DEBUG: Invalid gene sampled right here!")

        return individual

# class NullCrossover(Crossover):
#     def __init__(self, num_channels, num_points):
#         super().__init__(n_parents=2, n_offsprings=2)
#         self.num_channels = num_channels
#         self.num_points = num_points

#     def count_channels(self, channels):
#         """
#         Count the number of actual channels in the given array.

#         Parameters:
#         - channels: numpy array of shape (n_channels, n_points, 2),
#                     where each channel consists of its control points.

#         Returns:
#         - count: Number of actual channels (channels where the first two points are not identical).
#         """
#         count = 0
#         for channel in channels:
#             # Check if the first two points of the channel are identical
#             if not np.array_equal(channel[0], channel[1]):
#                 count += 1
#         return count
    
#     def _do(self, problem, X, **kwargs):
#         """
#         Perform single-channel crossover on two parents.

#         Parameters:
#         - X: (n_matings, n_parents, n_var), where n_parents=2 for NSGA-II.

#         Returns:
#         - offspring: (n_matings, n_offsprings, n_var).
#         """
#         n_matings = X.shape[0]
#         offspring = np.empty_like(X)

#         for i in range(n_matings):
#             if X.shape[1] == 2:
#                 # Extract parents
#                 parent1 = X[i, 0].reshape(self.num_channels, self.num_points, 2)
#                 parent2 = X[i, 1].reshape(self.num_channels, self.num_points, 2)

#                 n=self.count_channels(parent1)
#                 if n == 0:
#                     print("DEBUG: Invalid gene in Parent1!")
#                 n=self.count_channels(parent2)
#                 if n == 0:
#                     print("DEBUG: Invalid gene in Parent2!")

#                 # Randomly select a channel
#                 channel_idx = np.random.randint(1, self.num_channels)

#                 # Select random crossover point within the channel
#                 crossover_point = np.random.randint(1, self.num_points)

#                 # Create offspring
#                 child1 = parent1.copy()
#                 child2 = parent2.copy()

#                 # Swap points after the crossover point in the selected channel
#                 child1[channel_idx, crossover_point:] = parent2[channel_idx, crossover_point:]
#                 child2[channel_idx, crossover_point:] = parent1[channel_idx, crossover_point:]

#                 n=self.count_channels(child1)
#                 if n == 0:
#                     print("DEBUG: Invalid gene in Child1!")
#                 n=self.count_channels(child2)
#                 if n == 0:
#                     print("DEBUG: Invalid gene in Child2!")

#                 # Flatten back to original dimensions
#                 offspring[i, 0] = child1.flatten()
#                 offspring[i, 1] = child2.flatten()
#             elif X.shape[1] != 2:
#                 offspring[i, 0] = X[i, 0]
#                 continue

#         return offspring
    
class NullCrossover(Crossover):
    def __init__(self, n_channels, n_points):
        super().__init__(n_parents=2, n_offsprings=2)
        self.num_channels = n_channels
        self.num_points = n_points
    def _do(self, problem, X, **kwargs):
        return X

# ---------------------------
# Running NSGA-II
# ---------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M1 = torch.load('M1_performance_predictor.pt', map_location=torch.device('cpu'))
    M1.to(device)
    M1.eval()

    inlet_velocity = torch.load('Data/v1.pt').to(device)
    heat_source = torch.load('Data/h2.pt').to(device)

    alpha = 0.5
    mu = 0.1
    n_channels = 8
    n_points = 8

    N = 50
    n_gen = 3

    prob1 = 1-2*alpha
    prob2 = 1-alpha
    prob3 = 1
    mutation_probabilities = [prob1, prob2, prob3]

    # Define the problem
    problem = ChannelGeometryProblem(
    M1=M1, 
    heat_source=heat_source, 
    inlet_velocity=inlet_velocity,
    n_channels=n_channels, 
    num_points=n_points
    )

    # Create the NSGA-II algorithm instance
    algorithm = NSGA2(
        pop_size=N,
        sampling = MyCustomSampling(n_channels, n_points),
        crossover = NullCrossover(n_channels, n_points),
        mutation = ChannelMutation(n_channels, n_points, mutation_probabilities, sigma=5.0)
    )

    # Define termination criterion, for example 50 generations
    termination = DefaultSingleObjectiveTermination(n_max_gen=n_gen)

    # Run optimization
    res = minimize(problem, algorithm, termination, seed=1, verbose=True)

    # Extract results
    print("Pareto-front objectives:\n", res.F)
    #print("Pareto-front decision variables:\n", res.X)

    with open('pareto_front_objectives.pkl', 'wb') as f:
        pickle.dump(res.F, f)
    with open('pareto_front_geometries.pkl', 'wb') as f:
        pickle.dump(res.X, f)
    print("Pareto front saved successfully")

    print("Pareto :", res.F.shape)


