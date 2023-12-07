import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


class HamiltonFilter:
    def __init__(self, data, n_states,): # parameters):
        # Initialize Data
        self.data = data
        # Set Number of Regime States
        self.n_states = n_states
        
        # Normalize each column to sum to 1
        self.transition_matrix = self.initialize_transition_matrix(n_states)




    def initialize_transition_matrix(self, n_states):
        """
        Initializes the transition matrix with specified properties.
        """
        # Create an empty matrix
        if n_states > 2:
            matrix = np.zeros((n_states, n_states))

            # Fill the diagonal with values between 0.95 and 1
            pii_values = np.random.uniform(0.95, 1, size=n_states)
            np.fill_diagonal(matrix, pii_values)

            # Set the off-diagonal values in each column
            for i in range(n_states):
                # Calculate the value to fill for off-diagonal elements
                off_diagonal_value = (1 - pii_values[i]) / (n_states - 2)

                # Fill off-diagonal elements except the last one
                for j in range(n_states):
                    if j != i:
                        matrix[j, i] = off_diagonal_value

                # Adjust the last off-diagonal element in each column
                matrix[-1, i] = 1 - matrix[:, i].sum() + matrix[-1, i]
        elif n_states == 2:
            matrix = np.zeros((n_states, n_states))

            # Fill the diagonal with values between 0.95 and 1
            pii_values = np.random.uniform(0.95, 1, size=n_states)
            np.fill_diagonal(matrix, pii_values)

            # Set the off-diagonal values in each column
            for i in range(n_states):
                # Calculate the value to fill for off-diagonal elements
                off_diagonal_value = (1 - pii_values[i]) 
                # Fill off-diagonal elements except the last one
                for j in range(n_states):
                    if j != i:
                        matrix[j, i] = off_diagonal_value

                # Adjust the last off-diagonal element in each column
                matrix[-1, i] = 1 - matrix[:, i].sum() + matrix[-1, i]

        # Print the transition matrix and the sum of each column
        # print("Transition Matrix:\n", matrix)
        # print("Sum of each column:", matrix.sum(axis=0))

        return matrix

    
















    def predict_step(self, xi_11):
        return self.transition_matrix.dot(xi_11)

    def filter_step(self, y_t, xi_10):
        eta = np.array([self.GaussianDensity(y_t, 0, self.sigma2[i]) for i in range(self.n_states)])
        xi_11 = eta * xi_10 / (np.dot(eta, xi_10))
        return xi_11

    def initialize_state_probabilities(self):
        # Set up initial state probabilities, if needed
        # Here, it's assumed to be uniform, but this can be changed based on the model specifics
        return np.full(self.n_states, 1.0 / self.n_states)

    def GaussianDensity(self, x, mean, variance):
        # Gaussian density function
        return np.exp(-0.5 * ((x - mean) ** 2) / variance) / np.sqrt(2 * np.pi * variance)
    
    def run_filter(self, initial_params, y):
        T = len(y)
        xi_10 = np.zeros((self.n_states, T + 1))
        xi_11 = np.zeros((self.n_states, T))
        likelihoods = np.zeros(T)

        # Initial state probabilities
        xi_10[:, 0] = self.initialize_state_probabilities()

        for t in range(T):
            # State densities
            eta = np.array([self.GaussianDensity(y[t], 0, self.sigma2[i]) for i in range(self.n_states)])
            
            # Likelihood
            likelihoods[t] = np.log(np.dot(xi_10[:, t], eta))

            # Filtering
            xi_11[:, t] = self.filter_step(y[t], xi_10[:, t])

            # Prediction
            xi_10[:, t + 1] = self.predict_step(xi_11[:, t])

        return -np.sum(likelihoods)  # Negative log-likelihood

    def optimize_parameters(self, y, initial_params, bounds):
        result = minimize(self.run_filter, initial_params, args=(y,), method='L-BFGS-B', bounds=bounds)
        estimated_params = result.x
        variance_hessian = result.hess_inv.todense()
        standard_errors = np.sqrt(np.diag(variance_hessian))
        return estimated_params, standard_errors


    def forward_backward_filter(self, y, P, sigma2):
        T = len(y)
        n_states = self.n_states

        # Forward filtering
        xi_10 = np.zeros((n_states, T + 1))
        xi_11 = np.zeros((n_states, T))
        xi_10[:, 0] = self.initialize_state_probabilities()

        for t in range(T):
            eta = np.array([self.GaussianDensity(y[t], 0, sigma2[i]) for i in range(n_states)])
            xi_11[:, t] = eta * xi_10[:, t] / (np.dot(eta, xi_10[:, t]))
            xi_10[:, t + 1] = P.dot(xi_11[:, t])

        # Backward smoothing
        xi_1T = np.zeros_like(xi_11)
        xi_1T[:, T - 1] = xi_11[:, T - 1]

        for t in range(T - 2, -1, -1):
            for i in range(n_states):
                backward_message = 0
                for j in range(n_states):
                    backward_message += xi_1T[j, t + 1] * P[i, j]
                xi_1T[i, t] = xi_11[i, t] * backward_message

        return xi_10, xi_11, xi_1T
    def analyze_and_plot(self, y, estimated_params, standard_errors):
        # Unpack estimated parameters for transition probabilities and volatilities
        transition_probs = estimated_params[:self.n_states]
        sigma2 = estimated_params[self.n_states:]**2

        # Forward filter and backward smoother
        xi_10, xi_11, xi_1T = self.forward_backward_filter(y, transition_probs, sigma2)

        # Compute filtered volatility for each state
        vol = np.sum(xi_11 * sigma2[:, np.newaxis], axis=0)

        # Plotting
        self.plot_results(y, vol)
        self.plot_probabilities(xi_10, xi_11, xi_1T, len(y))

        # Print estimated parameters and standard errors
        for i in range(self.n_states):
            print(f'State {i+1} transition probability = {transition_probs[i]}, std.error = {standard_errors[i]}')
            print(f'State {i+1} volatility = {sigma2[i]}, std.error = {standard_errors[i + self.n_states]}')

    def plot_results(self, y, vol):
        fig, ax = plt.subplots(2, figsize=(14, 7))
        fig.suptitle('Log-return and Filtered Volatility')

        # Plot log-return
        ax[0].plot(y, color='r')
        ax[0].title.set_text('Log-return, $x_t$')

        # Plot filtered volatility
        ax[1].plot(np.sqrt(vol))
        ax[1].title.set_text('Filtered volatility, $E[\sigma_t|x_t,x_{t-1},...,x_1]$')

        plt.show()
        

    def plot_probabilities(self, xi_10, xi_11, xi_1T, T):
        n_states = xi_10.shape[0]  # Number of states
        fig, axs = plt.subplots(n_states, 3, figsize=(18, 4 * n_states))  # Adjust size as needed

        for state in range(n_states):
            # Predicted probabilities for this state
            axs[state, 0].plot(xi_10[state, :])
            axs[state, 0].set_title(f'State {state + 1} - Predicted Probabilities')
            axs[state, 0].set_xlim(0, T)
            axs[state, 0].axhline(0, color='black', linestyle="--")
            axs[state, 0].axhline(1, color='black', linestyle="--")

            # Filtered probabilities for this state
            axs[state, 1].plot(xi_11[state, :])
            axs[state, 1].set_title(f'State {state + 1} - Filtered Probabilities')
            axs[state, 1].set_xlim(0, T)
            axs[state, 1].axhline(0, color='black', linestyle="--")
            axs[state, 1].axhline(1, color='black', linestyle="--")

            # Smoothed probabilities for this state
            axs[state, 2].plot(xi_1T[state, :])
            axs[state, 2].set_title(f'State {state + 1} - Smoothed Probabilities')
            axs[state, 2].set_xlim(0, T)
            axs[state, 2].axhline(0, color='black', linestyle="--")
            axs[state, 2].axhline(1, color='black', linestyle="--")

        plt.tight_layout()
        plt.show()


