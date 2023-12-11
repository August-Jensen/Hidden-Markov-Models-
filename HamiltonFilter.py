import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from scipy.optimize import minimize


"""
When calling, it should be given data and n_states.
    Additionally, it should be possible to specify the following 
        parameters, 
        distribution function,
        generate randomly,
        transition probabilities, 
        initial state probabilits, 
        parameter bounds 
"""

class BaseFilter():
    """docstring for BaseFilter
    This version is for a 2 state SV model.
    Missing
        - Use Dictionary or lists? How to pass transition_matrix, and parameter values into the model.
        - Could allow for drawing random_values, or else setting them by default, if they are not provided when calling
        - Distribution of emissions, and MLE. Normal or t distributed with nu degrees of freedom.
        
    """

    def __init__(self, data, n_states, initial_state_probs=None, initial_transition_matrix=None, initial_parameters=None, initial_bounds=None,):
        """
        
        """
        # Set basis parameters
        self.data = data
        self.n_states = n_states
        self.num_obs = len(data)
        self.iteration = 0
        # self.distribution for normal or t distributed errors

        # set up parameters, and parameter bounds
        self.sigmas = initial_parameters if initial_parameters is not None else self.default_params() 
        # print(f'Sigma Parameters:  {self.sigmas}')
        self.parameter_bounds = np.full((n_states, 2), (0.001, None))  # Filling with (0.001, None)
        self.parameter_bounds = [(low, high) for low, high in self.parameter_bounds]

        # print(f'Parameter Bounds:  {self.parameter_bounds}')

        # set up state and transition probabilities.
        self.initial_state_probs = initial_state_probs if initial_state_probs is not None else self.default_state_probs()
        # print(f'Initial State Probability:  {self.initial_state_probs}')

        self.transition_matrix = initial_transition_matrix if initial_transition_matrix is not None else self.initialize_transition_matrix(n_states)
        # print(f'Transition Matrix: \n  {self.transition_matrix}')
        self.probability_parameters = self.extract_transition_parameters(self.transition_matrix)
        # print(f'Probabilitity parameters, length:{len(self.probability_parameters)}, parameters: {self.probability_parameters}')

        self.probability_bounds =  self.probability_bounds = [(0.001, 0.9999) for _ in range(len(self.probability_parameters))] # Filling with (0.001, 0.9999)
        # print(f'Bounds on Probabilities, length:{len(self.probability_bounds)}, parameters: {self.probability_bounds}')

        



    def initialize_transition_matrix(self, n_states):
        """
        Sets a transition matrix with n_states x n_states dimensions.
        Diagonal (probability of staying in a state) has probability 0.95,
        rest are adjusted to sum to 1
        """
        
        # Create a matrix of zeros
        transition_matrix = np.zeros((n_states, n_states))

        # Fill the diagonal with 0.95
        np.fill_diagonal(transition_matrix, 0.90)

        # Calculate the off-diagonal value
        off_diagonal_value = (1 - 0.9) / (n_states - 1)

        # Fill the off-diagonal elements
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    transition_matrix[i, j] = off_diagonal_value

        return transition_matrix

    
    def extract_transition_parameters(self, transition_matrix):
        """
        Extracts the parameters (first n_states - 1 elements of each column) from the transition matrix.
        """
        n_states = transition_matrix.shape[0]
        # Extract all but the last element from each column
        params = transition_matrix[:-1, :].flatten()
        return params

    def update_transition_matrix(self,n_states, probability_parameters):
        """
        Update the transition matrix based on a list of parameters.
        Each column's last element is set to 1 minus the sum of the other elements in the column.
        """
        # Ensure the correct number of parameters are provided
        assert len(probability_parameters) == n_states * (n_states - 1), "Incorrect number of parameters"

        # Initialize a matrix to hold the updated values
        updated_matrix = np.zeros((n_states, n_states))

        # Fill in the matrix column by column
        for col in range(n_states):
            # Extract parameters for this column (excluding the last state)
            start_idx = col * (n_states - 1)
            end_idx = start_idx + (n_states - 1)
            column_params = probability_parameters[start_idx:end_idx]

            # Set the values for this column
            updated_matrix[:-1, col] = column_params

            # Calculate and set the last element of the column
            updated_matrix[-1, col] = 1 - np.sum(column_params)

        return updated_matrix

    def default_params(self):
        """
        Set the initial parameters to a list if random=False
        Else, set to random
        """

        if self.n_states == 2:
            params = [3, 2]
            return params
        elif self.n_states == 3:
            params = [0.5, 1, 2]
            return params
        elif self.n_states == 4:
            params = [0.5, 1, 1.5, 2]
            return params
        elif self.n_states == 5:
            params = [0.5, 1, 1.5, 2, 2.5]
            return params

    def default_state_probs(self):
        initial_state_probs = np.full(self.n_states, 1.0 / self.n_states)
        return initial_state_probs

    def emission_probability(self, state, t):
        """
        Calculate the emission probability for a given state and time t.
        """
        # print(self.mu[state],self.phi[state])
        if t == 0:
            previous_x = 0  # Handle the case for the first observation
        else:
            previous_x = self.data[t-1]

        mean = 0 # self.mu[state] + self.phi[state] * previous_x
        variance = self.sigmas[state] ** 2
        emissions = norm.pdf(self.data[t], mean,variance)
        #print(emissions)
        return  emissions


    def filtering_step(self, t):
        """
        Calculate the emission probabiliteis, 
        Then multiply emission probability with pedicted probability from prior period
        Sum these, 
        return the filtered probabiliy as the numerator divided by the denominator. 
        """
        emission_probs = np.array([self.emission_probability(state, t) for state in range(self.n_states)])

        # Set predicted probability of prior period
        if t == 0:
            predicted_prob = self.initial_state_probs
        else:
            predicted_prob = self.predicted_probability[:, t-1]

        # Calculate the filtered probability
        numerator = emission_probs * predicted_prob
        denominator = np.sum(numerator)
        filtered_prob = numerator / denominator
        #print(filtered_prob)

        # Store results
        self.filtered_probability[:, t] = filtered_prob
        self.emission_probabilities[:, t] = emission_probs


    def prediction_step(self, t):
        """
        Calculate the predicted probability for t+1 based on the 
        filtered probabilities at time t, and the transition matrix

        """
        if t == 0:
            # For the first step, we use the initial state probabilities,
            self.predicted_probability[:,1] = self.initial_state_probs
        else:
            for state in range(self.n_states):
                # Calculate the predicted probability for each state at t+1
                self.predicted_probability[state, t+1] = np.sum(
                    self.transition_matrix[:, state] * self.filtered_probability[:, t]
                )

        

    def calculate_log_likelihood_contributions(self, t):
        """
        Calculate the log-likelihood contribution at time t.
        """
        x = self.data[t]

        # Compute log-likelihood for each state. 
        log_likelihoods = np.zeros(self.n_states)
        for state in range(self.n_states):
            mean = 0 # self.mu[state]
            variance = self.sigmas[state] 
            log_likelihoods[state] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(variance) - ((x - mean) ** 2) / (2 * variance)

        # Multiply each state's log-likelihood by its emission probability and sum
        log_likelihood_contribution = np.sum(log_likelihoods * self.emission_probabilities[:, t])
        # print(f'Log-Likelihood Contribution: {log_likelihood_contribution}')
        return log_likelihood_contribution

    def filter_algorithm(self, estimation_parameters): #params)
        """
        Initialize data Arrays.
        Setup Parameters.
        Set first predicted_probability
        Then run a for loop of:
            filtering_step
            prediciton_step
            calculate likelihood.

        """
        n_states = self.n_states  # Assuming n_states is defined in your class

        # Split estimation_parameters into sigmas and transition matrix parameters
        self.sigmas = estimation_parameters[:n_states]
        probability_parameters = estimation_parameters[n_states:]

        # Update the transition matrix
        self.transition_matrix = self.update_transition_matrix(n_states, probability_parameters)
        print(self.transition_matrix)
        # print(f'Sigma Parameters:  {self.sigmas}')
        # print(f'Transition Matrix: \n  {self.transition_matrix}')
        # print(f'Probabilitity parameters, length:{len(self.probability_parameters)}, parameters: {self.probability_parameters}')
        # setup numpy arrays 
        # Initialize Arrays for storing values
        self.predicted_probability = np.zeros([self.n_states, self.num_obs + 1])
        self.filtered_probability = np.zeros([self.n_states,self.num_obs])
        self.smoothed_probabilities = np.zeros([self.n_states,self.num_obs])
        self.likelihood_contributions = np.zeros(self.num_obs)
        self.emission_probabilities = np.zeros([self.n_states, self.num_obs])

        # Set the first predicted_probability to be initial_state_probs
        # This is also done in prediction step and filtering step
        self.predicted_probability[:, 0] = self.initial_state_probs

        # Set up Transition Matrix probabilities.

        #print('Now running for loop in filter_algorithm')
        for t in range(self.num_obs):
            #print(f'Iteration:  {t}')
            # Perform filtering_step at time t
            self.filtering_step(t)

            # Perform the prediction step for the next time point
            self.prediction_step(t)

            # Calculate and store the log likelihood contribution for time t
            self.likelihood_contributions[t] = self.calculate_log_likelihood_contributions(t)
        total_likelihood = -np.sum(self.likelihood_contributions)
        print(f'Total log-likelihood:  {total_likelihood}')
        print(f'Iteration: {self.iteration}')
        self.iteration += 1

        return total_likelihood

    def fit(self, ): # initial_guess, bounds):
        """
        set residuals as the minimize function, that minimizes the filter_algorithm, for initial parameters, and bounds.

        """
        # # Create Transition Matrix
        # combined_bounds = self.parameter_bounds + self.probability_bounds
        # print(combined_bounds)

        # estimation_parameters = np.concatenate([self.sigmas, self.probability_parameters])
        # print(estimation_parameters)
        # Combine parameters
        estimation_parameters = np.concatenate([self.sigmas, self.probability_parameters])
        print("Estimation Parameters:", estimation_parameters)
        # Combine bounds
        combined_bounds = self.parameter_bounds + self.probability_bounds
        print("Combined Bounds:", combined_bounds)
        # Ensure the number of bounds matches the number of parameters
        if len(combined_bounds) != len(estimation_parameters):
            raise ValueError("The number of bounds does not match the number of parameters.")

        res = minimize(self.filter_algorithm, estimation_parameters, method='L-BFGS-B', bounds=combined_bounds)
        print(res)



        Gamma_hat = res.x
        v_hessian = res.hess_inv.todense()
        se_hessian = np.sqrt(np.diagonal(v_hessian))

        # Print results
        for i in range(len(Gamma_hat)):
            print(f'Parameter {i}: {Gamma_hat[i]}, standard error: {se_hessian[i]}')

        return Gamma_hat, se_hessian

    def smoothing_step(self):
        """
        
        """
        
        pass

    def predict(self):
        """
        
        """
        
        pass



    def statistics(self):
        """

        """
        pass

    def plot_results(self):
        """
        
        """
        
        pass

    def plot_smoothed_probabilities(self):
        """
        
        """
        
        pass

    def plot_convergence(self):
        """
        
        """
        
        pass