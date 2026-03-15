import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from model import DynamicNN
from train_test_loop import train_model, test_model
import time
import random
import os
import csv

import numpy as np
from scipy.stats import norm


# Hyperparameter search spaces
LR_SPACE = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2] 
NUM_HIDDEN_LAYERS_SPACE = [1, 2, 3, 4]
HIDDEN_DIM_SPACE = [16, 32, 64, 128, 256]
BATCH_SIZE_SPACE = [32, 64, 128, 256]

# Loading the MNIST dataset
train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

img, label = train_data[0]
INPUT_SIZE = img.shape[0] * img.shape[1] * img.shape[2]   # 784
OUTPUT_SIZE = 10


class TestHpo:

    def __init__(self):
        self.current_lr_index = 0
        self.current_num_hl_index = 0
        self.current_hl_dim_index = 0
        self.current_batch_size_index = 0

    def generate_hidden_layer_configs(self, num_hidden_layers, hidden_dim):
        """
        Build one architecture using:
        - input size
        - selected number of hidden layers
        - selected hidden dimension
        - output size
        """
        configs = [INPUT_SIZE]

        for _ in range(num_hidden_layers):
            configs.append(hidden_dim)

        configs.append(OUTPUT_SIZE)
        return configs

    def get_model_performance(self):

        batch_size = BATCH_SIZE_SPACE[self.current_batch_size_index]

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        hidden_layers = self.generate_hidden_layer_configs(
            NUM_HIDDEN_LAYERS_SPACE[self.current_num_hl_index],
            HIDDEN_DIM_SPACE[self.current_hl_dim_index]
        )

        model = DynamicNN(hidden_layers)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR_SPACE[self.current_lr_index])

        # keep epochs to 3 so that model training does not take too long...
        model = train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, 3)
        test_loss, test_acc = test_model(model, test_dataloader, loss_fn)

        return test_acc

    
    def print_method_results(self, method_name, total_elapsed_time, best_test_acc, configuration = None):

        print(f"HPO using {method_name} complete...")
        print(f"Total time taken: {total_elapsed_time:.4f} seconds")
        print(f"The best test accuracy found was {best_test_acc:.2f}%")

        if configuration is not None:
            print("The configuration that performed the best was : ")
            print(f"lr = {configuration[0]}")
            print(f"Number of hidden layers = {configuration[1]}")
            print(f"Dimension of hidden layers = {configuration[2]}")
            print(f"Batch Size = {configuration[3]}")

    def write_results(self, f, writer, acc_, elapsed_time_step):
        writer.writerow([
            self.current_lr_index,
            self.current_num_hl_index,
            self.current_hl_dim_index,
            self.current_batch_size_index,
            f"{acc_:.4f}",
            f"{elapsed_time_step:.4f}"
        ])
        f.flush()  # flush to disk immediately


    def test_grid_search_performance(self, csv_file='gridsearchresults.csv'):
        start_time_total = time.perf_counter()
        best_test_acc = 0
        configuration = None

        # Initialize start indices from last line of CSV
        start_indices = [0, 0, 0, 0]

        # Ensure CSV exists
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                pass  # create empty file

        # Read last line if file is not empty
        with open(csv_file, 'r', newline='') as f:
            reader = list(csv.reader(f))
            if reader:
                last_line = reader[-1]
                if len(last_line) >= 4:
                    start_indices = [int(x) for x in last_line[:4]]

        # Open CSV in append mode
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # Initialize while loop indices with start_indices
            # since there are more than 400 combinations, we want to save any progress
            # and progress from that point onwards
            lr_index = start_indices[0]
            while lr_index < len(LR_SPACE):
                hl_num_space_index = start_indices[1] if lr_index == start_indices[0] else 0
                while hl_num_space_index < len(NUM_HIDDEN_LAYERS_SPACE):
                    hl_dim_space_index = start_indices[2] if (lr_index == start_indices[0] and hl_num_space_index == start_indices[1]) else 0
                    while hl_dim_space_index < len(HIDDEN_DIM_SPACE):
                        batch_size_space_index = start_indices[3] if (lr_index == start_indices[0] and
                                                                    hl_num_space_index == start_indices[1] and
                                                                    hl_dim_space_index == start_indices[2]) else 0
                        while batch_size_space_index < len(BATCH_SIZE_SPACE):

                            # Set current hyperparameters
                            self.current_lr_index = lr_index
                            self.current_num_hl_index = hl_num_space_index
                            self.current_hl_dim_index = hl_dim_space_index
                            self.current_batch_size_index = batch_size_space_index

                            # Measure time for this combination
                            step_start_time = time.perf_counter()
                            test_acc = self.get_model_performance()
                            step_end_time = time.perf_counter()
                            elapsed_time_step = step_end_time - step_start_time

                            # Update best
                            if test_acc > best_test_acc:
                                best_test_acc = test_acc
                                configuration = [
                                    LR_SPACE[lr_index],
                                    NUM_HIDDEN_LAYERS_SPACE[hl_num_space_index],
                                    HIDDEN_DIM_SPACE[hl_dim_space_index],
                                    BATCH_SIZE_SPACE[batch_size_space_index]
                                ]

                            self.write_results(f, writer, test_acc, elapsed_time_step)

                            batch_size_space_index += 1
                        hl_dim_space_index += 1
                    hl_num_space_index += 1
                lr_index += 1

        # Total time
        end_time_total = time.perf_counter()
        total_elapsed_time = end_time_total - start_time_total

        self.print_method_results("Grid Search", total_elapsed_time, best_test_acc, configuration)

        return total_elapsed_time, best_test_acc, configuration


    # randomly generate a new combination 
    def get_random_combination(self):
        configuration_indices = [
            random.randint(0, len(LR_SPACE) - 1),
            random.randint(0, len(NUM_HIDDEN_LAYERS_SPACE) - 1),
            random.randint(0, len(HIDDEN_DIM_SPACE) - 1),
            random.randint(0, len(BATCH_SIZE_SPACE) - 1)
        ]
        return np.array(configuration_indices)
    
    def test_random_combination(self, bounds=None):
        if bounds:
            # bounds = [lr_lower, lr_upper, num_hl_lower, num_hl_upper,
            #           dim_hl_lower, dim_hl_upper, batch_lower, batch_upper]

            # bounds are used to restrict the search space during successive halving
            lr_lower, lr_upper = bounds[0], bounds[1]
            num_hl_lower, num_hl_upper = bounds[2], bounds[3]
            dim_hl_lower, dim_hl_upper = bounds[4], bounds[5]
            batch_lower, batch_upper = bounds[6], bounds[7]

            self.current_lr_index = random.randint(lr_lower, lr_upper - 1)
            self.current_num_hl_index = random.randint(num_hl_lower, num_hl_upper - 1)
            self.current_hl_dim_index = random.randint(dim_hl_lower, dim_hl_upper - 1)
            self.current_batch_size_index = random.randint(batch_lower, batch_upper - 1)
        else:
            # default full range
            self.current_lr_index = random.randint(0, len(LR_SPACE) - 1)
            self.current_num_hl_index = random.randint(0, len(NUM_HIDDEN_LAYERS_SPACE) - 1)
            self.current_hl_dim_index = random.randint(0, len(HIDDEN_DIM_SPACE) - 1)
            self.current_batch_size_index = random.randint(0, len(BATCH_SIZE_SPACE) - 1)

        # Evaluate performance
        test_acc = self.get_model_performance()

        # Build configuration
        configuration = [
            LR_SPACE[self.current_lr_index],
            NUM_HIDDEN_LAYERS_SPACE[self.current_num_hl_index],
            HIDDEN_DIM_SPACE[self.current_hl_dim_index],
            BATCH_SIZE_SPACE[self.current_batch_size_index]
        ]

        return test_acc, configuration

    def test_random_search_performance(self, csv_file='randomsearchresults.csv'):
        start_time_total = time.perf_counter()
        configuration = None
        best_test_acc = 0

        # Ensure CSV exists
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                pass  # create empty file

        # Open CSV in append mode
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            for iteration in range(10):
                # measure time for this iteration
                step_start_time = time.perf_counter()
                test_acc, curr_configuration = self.test_random_combination()
                step_end_time = time.perf_counter()
                elapsed_time_step = step_end_time - step_start_time

                # update best
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    configuration = curr_configuration

                # Write to CSV: current indices, test_acc, elapsed_time
                self.write_results(f, writer, test_acc, elapsed_time_step)

        # Total time for the whole random search
        end_time_total = time.perf_counter()
        total_elapsed_time = end_time_total - start_time_total
        self.print_method_results("Random Search", total_elapsed_time, best_test_acc, configuration)

        return total_elapsed_time, best_test_acc, configuration


    def test_successive_halving_performance(self, csv_file='successivehalvingresults.csv'):
        start_time_total = time.perf_counter()
        configuration = None
        best_test_acc = 0

        # Initialize bounds for the full hyperparameter spaces
        bounds = [
            0, len(LR_SPACE),
            0, len(NUM_HIDDEN_LAYERS_SPACE),
            0, len(HIDDEN_DIM_SPACE),
            0, len(BATCH_SIZE_SPACE)
        ]

        # Ensure CSV exists
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                pass

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            for iteration in range(10):
                best_itr_performance = 0

                # Calculate integer step sizes
                lr_step = max(1, (bounds[1] - bounds[0]) // 2)
                hl_step = max(1, (bounds[3] - bounds[2]) // 2)
                dim_step = max(1, (bounds[5] - bounds[4]) // 2)
                batch_step = max(1, (bounds[7] - bounds[6]) // 2)

                best_bounds_itr = None

                for lr_start in range(bounds[0], bounds[1], lr_step):
                    for hl_start in range(bounds[2], bounds[3], hl_step):
                        for dim_start in range(bounds[4], bounds[5], dim_step):
                            for batch_start in range(bounds[6], bounds[7], batch_step):

                                # Define current subspace bounds
                                test_bounds = [
                                    lr_start, min(lr_start + lr_step, bounds[1]),
                                    hl_start, min(hl_start + hl_step, bounds[3]),
                                    dim_start, min(dim_start + dim_step, bounds[5]),
                                    batch_start, min(batch_start + batch_step, bounds[7])
                                ]

                                step_start_time = time.perf_counter()
                                # Pass bounds to random combination
                                test_acc, curr_configuration = self.test_random_combination(bounds=test_bounds)
                                step_end_time = time.perf_counter()
                                elapsed_time_step = step_end_time - step_start_time

                                # update global best
                                if test_acc > best_test_acc:
                                    best_test_acc = test_acc
                                    configuration = curr_configuration

                                # Track best for this iteration to shrink bounds
                                if test_acc > best_itr_performance:
                                    best_itr_performance = test_acc
                                    best_bounds_itr = test_bounds

                                # Write results to CSV
                                writer.writerow([
                                    self.current_lr_index,
                                    self.current_num_hl_index,
                                    self.current_hl_dim_index,
                                    self.current_batch_size_index,
                                    f"{test_acc:.4f}",
                                    f"{elapsed_time_step:.4f}",
                                    test_bounds[0], test_bounds[1],
                                    test_bounds[2], test_bounds[3],
                                    test_bounds[4], test_bounds[5],
                                    test_bounds[6], test_bounds[7]
                                ])
                                f.flush()
                
                bounds = best_bounds_itr

        total_elapsed_time = time.perf_counter() - start_time_total
        # print out results 
        self.print_method_results("Successive Halving", total_elapsed_time, best_test_acc, configuration)

        return total_elapsed_time, best_test_acc, configuration

    ##helper functions for 
    def initialize_length_scale_vec(self):

        length_scale_vec = np.zeros(4)

        length_scale_vec[0] = 0.1 * (max(LR_SPACE) - min(LR_SPACE))
        length_scale_vec[1] = 0.1 * (max(NUM_HIDDEN_LAYERS_SPACE) - min(NUM_HIDDEN_LAYERS_SPACE))
        length_scale_vec[2] = 0.1 * (max(HIDDEN_DIM_SPACE) - min(HIDDEN_DIM_SPACE))
        length_scale_vec[3] = 0.1 * (max(BATCH_SIZE_SPACE) - min(BATCH_SIZE_SPACE))

        return np.maximum(length_scale_vec, 1e-12)

    def compute_kernel_function(self, length_scale_vec, signal_variance, config1, config2):
        
        d_div_length_scale = (config1 - config2) / length_scale_vec
        # compute kernel function
        return signal_variance * np.exp( -0.5 * np.sum(d_div_length_scale**2))

    ## gram matrix K 
    def get_matrix_K(self, length_scale_vec, signal_variance, candidates):

        length = len(candidates)
        K = np.zeros((length, length))

        for i in range(length):
            for j in range(length):
                K[i,j] = self.compute_kernel_function(
                    length_scale_vec, signal_variance, candidates[i], candidates[j]
                )

        K += 1e-8 * np.eye(length)
        return K

    def get_kernel_vector(self, length_scale_vec, signal_variance, candidates, new_candidate):

        length = len(candidates)
        k_ = np.zeros(length)

        for i in range(length):
            k_[i] = self.compute_kernel_function(length_scale_vec, signal_variance,candidates[i], new_candidate)

        return k_

    def get_posterior_predictive_mean(self, k_, K, Y):
        alpha = np.linalg.solve(K, Y)
        return k_.T @ alpha 

    def get_posterior_variance_at_val(self, length_scale_vec, signal_variance, k_, K, val):
        v = np.linalg.solve(K, k_)
        variance = self.compute_kernel_function(length_scale_vec, signal_variance, val, val) - k_.T @ v
        #avoid division by zero error
        variance = max(variance, 1e-12)

        return variance
    # compute EI
    # Expected Improvement for minimization
    def get_EI(self, mean, variance, Y):
        min_ = np.min(Y)
        sigma_ = np.sqrt(max(variance, 1e-12)) 
        v = (min_ - mean) / sigma_
        return sigma_ * (v * norm.cdf(v) + norm.pdf(v))

    def test_bayesian_optimization(self, csv_file='bayesianoptimizationresults.csv'):

        length_scale_vec = self.initialize_length_scale_vec()
        signal_variance  = 1.0 

        candidates = []
        Y = []

        start_time_total = time.perf_counter()
        configuration = None
        best_test_acc = 0

        # Ensure CSV exists
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                pass  # create empty file

        # Open CSV in append mode
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # build baseline candidates from 10 results 
            for iteration in range(10):
                # measure time for this iteration
                step_start_time = time.perf_counter()
                test_acc, curr_configuration = self.test_random_combination()
                step_end_time = time.perf_counter()
                elapsed_time_step = step_end_time - step_start_time

                candidates.append(curr_configuration)
                Y.append(-test_acc)

                # update best
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    configuration = curr_configuration

                # Write to CSV: current indices, test_acc, elapsed_time
                self.write_results(f, writer, test_acc, elapsed_time_step)
            
            Y = np.array(Y, dtype=float)
            candidates = np.array(candidates, dtype=float)

            for bayesian_optmization_iterations in range(20):

                K = self.get_matrix_K(length_scale_vec, signal_variance, candidates)
                highest_EI = -float('inf')
                itr_best_candidate = None
                itr_best_candidate_indices = [0,0,0,0]

                for random_candidiate in range(20):

                    current_candidate_indices = self.get_random_combination()
                    
                    current_candidate = np.zeros(4)
                    current_candidate[0] = LR_SPACE[current_candidate_indices[0]]
                    current_candidate[1] = NUM_HIDDEN_LAYERS_SPACE[current_candidate_indices[1]]
                    current_candidate[2] = HIDDEN_DIM_SPACE[current_candidate_indices[2]]
                    current_candidate[3] = BATCH_SIZE_SPACE[current_candidate_indices[3]]

                    k_ = self.get_kernel_vector(length_scale_vec, signal_variance, candidates, current_candidate)
                    mean = self.get_posterior_predictive_mean(k_, K, Y)
                    variance = self.get_posterior_variance_at_val(length_scale_vec, signal_variance, k_, K, current_candidate)

                    curr_EI = self.get_EI(mean, variance, Y)

                    if curr_EI >= highest_EI:
                        highest_EI = curr_EI
                        itr_best_candidate = current_candidate
                        itr_best_candidate_indices = current_candidate_indices

                self.current_lr_index = itr_best_candidate_indices[0]
                self.current_num_hl_index = itr_best_candidate_indices[1]
                self.current_hl_dim_index = itr_best_candidate_indices[2]
                self.current_batch_size_index = itr_best_candidate_indices[3]
                
                step_start_time = time.perf_counter()
                candidate_acc = self.get_model_performance()
                step_end_time = time.perf_counter()
                elapsed_time_step = step_end_time - step_start_time

                # Write to CSV: current indices, test_acc, elapsed_time
                self.write_results(f, writer, candidate_acc, elapsed_time_step)

                if candidate_acc > best_test_acc:
                    best_test_acc = candidate_acc
                    configuration = itr_best_candidate.tolist()
                
                Y = np.append(Y, -candidate_acc)
                candidates = np.vstack([candidates, itr_best_candidate])

        # Total time for the whole Bayesian search
        end_time_total = time.perf_counter()
        total_elapsed_time = end_time_total - start_time_total

        self.print_method_results("Bayesian Optimization", total_elapsed_time, best_test_acc, configuration)

        return total_elapsed_time, best_test_acc, configuration

