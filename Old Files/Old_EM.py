import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
# Load and preprocess your data
# data = pd.read_csv('your_timeseries_data.csv')
# Define your MS-VAR model

class ExpectationMaximization(object):
    """
    docstring for ExpectationMaximization
    For The AR(1) process

    """

    def __init__(self, n_states, data, tolerance=1e-6, max_iterations=100):
        """
        We need to implement the max_iterations, and tolerance
        """
        # Setup of Input
        self.n_states = n_states
        self.data = data
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        # Setup of Probabilities
        self.transition_matrix = self.initialize_transition_matrix(n_states)
        self.initial_state_probs = np.full(n_states, 1.0 / n_states)


        #Setup of Parameters
        self.mu = [-0.4,0,0.4] # (np.random.rand(n_states)* 2)-1  # or set to a specific starting value [-0.4,0,0.4] # 
        self.phi =  [-0.4,0,0.4] # (np.random.rand(n_states) * 2)-1 # or set to a specific starting value [-0.1,0,0.1]
        self.sigma =  [2,3,4] # np.random.rand(n_states)*3   #[0.1,0.1,0.1]
        print(self.mu, self.phi, self.sigma)

        # Initialize matrix for storing responsibilities (E-step)
        self.responsibilities = np.zeros((len(data), n_states))



        # Initialize Histories for Plotting Parameters
        self.mu_history = []
        self.phi_history = []
        self.sigma_history = []
        self.smoothed_probabilities = None

    def initialize_transition_matrix(self, n_states):
        """
        Initializes the transition matrix with specified properties.
        """
        # Create an empty matrix
        if n_states == 2:
            matrix = np.array([[0.975,0.025],[0.025,0.975]])

        elif n_states == 3:
            matrix = np.array([[0.95,0.025,0.025],[0.025,0.95,0.025],[0.025,0.025,0.95]])

        return matrix
        # if n_states > 2:
        #     matrix = np.zeros((n_states, n_states))

        #     # Fill the diagonal with values between 0.95 and 1
        #     pii_values = np.random.uniform(0.95, 1, size=n_states)

        #     np.fill_diagonal(matrix, pii_values)

        #     # Set the off-diagonal values in each column
        #     for i in range(n_states):
        #         # Calculate the value to fill for off-diagonal elements
        #         off_diagonal_value = (1 - pii_values[i]) / (n_states - 2)

        #         # Fill off-diagonal elements except the last one
        #         for j in range(n_states):
        #             if j != i:
        #                 matrix[j, i] = off_diagonal_value

        #         # Adjust the last off-diagonal element in each column
        #         matrix[-1, i] = 1 - matrix[:, i].sum() + matrix[-1, i]
        # elif n_states == 2:
        #     matrix = np.zeros((n_states, n_states))

        #     # Fill the diagonal with values between 0.95 and 1
        #     pii_values = np.random.uniform(0.95, 1, size=n_states)
        #     np.fill_diagonal(matrix, pii_values)

        #     # Set the off-diagonal values in each column
        #     for i in range(n_states):
        #         # Calculate the value to fill for off-diagonal elements
        #         off_diagonal_value = (1 - pii_values[i]) 
        #         # Fill off-diagonal elements except the last one
        #         for j in range(n_states):
        #             if j != i:
        #                 matrix[j, i] = off_diagonal_value

        #         # Adjust the last off-diagonal element in each column
                # matrix[-1, i] = 1 - matrix[:, i].sum() + matrix[-1, i]

        # Print the transition matrix and the sum of each column
        # print("Transition Matrix:\n", matrix)
        # print("Sum of each column:", matrix.sum(axis=0))



    def emission_probability(self, state, t):
        """
        Calculate the emission probability for a given state and time t.
        """
        # print(self.mu[state],self.phi[state])
        if t == 0:
            previous_x = 0  # Handle the case for the first observation
        else:
            previous_x = self.data[t-1]

        mean = self.mu[state] + self.phi[state] * previous_x
        variance = self.sigma[state] ** 2
        emissions = norm.pdf(self.data[t], mean,variance)
        #print(emissions)
        return  emissions

    def calculate_likelihood(self, state):
        total_log_likelihood = 0

        # Calculate the log likelihood for each state

        for t in range(1, len(self.data)):
            for state in range(self.n_states):
                print(state)
                if t == 0:
                    previous_x = 0 # Handle the case for the first observation
                else:
                    previous_x = self.data[t-1]
                #print(self.mu[state],self.phi[state],self.sigma[state])

               
                log_likelihood = -0.5 * np.log(2 * np.pi) \
                                 - 0.5 * np.log(self.sigma[state]) \
                                 - ((self.data[t] - self.mu[state] - self.phi[state] * previous_x) ** 2) / (2 * self.sigma[state])
                total_log_likelihood += log_likelihood

            # Update previous_x if necessary for your model
            #previous_x = # some function of self.data[t] or similar


        return -total_log_likelihood  # negative because scipy.optimize minimizes

    # def calculate_likelihood_contribution(self, state, t, previous_x):
    #     '''Calculate the likelihood contribution for a given state and time step.
        
    #     Args:
    #     state (int): The current state.
    #     t (int): The time step.
    #     previous_x (float): The value of x at the previous time step.

    #     Returns:
    #     float: The minimized residual for the given state and time step.
    #     '''
    #     def residuals(x):
    #         res = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(self.sigma[state]) - ((self.data[t] - self.mu[state] - self.phi[state] * x) ** 2) / (2 * self.sigma[state])
    #         return -res  # Minimizing the negative of the residual

    #     # Using scipy minimize function with method 'L-BFGS-B'
    #     result = minimize(residuals, previous_x, method='L-BFGS-B')

    #     if result.success:
    #         return result.fun
    #     else:
    #         raise ValueError("Optimization failed")


    # def calculate_log_likelihood_contributions(self):
    #     """
    #     Calculate the log-likelihood contributions for each observation.
    #     """
    #     log_likelihood_contributions = np.zeros(len(self.data))

    #     for t in range(len(self.data)):
    #         log_likelihood_sum = 0
    #         for state in range(self.n_states):
    #             # Calculate the log likelihood for each state
    #             if t == 0:
    #                 previous_x = 0 # Handle the case for the first observation
    #             else:
    #                 previous_x = self.data[t-1]

    #             # Log likelihood for normal distribution
    #             one = -0.5 * np.log(2 * np.pi) 
    #             two = - 0.5 * np.log(self.sigma[state]) 
    #             three = - (self.data[t] - self.mu[state] - self.phi[state] * previous_x) ** 2
    #             four = (2 * self.sigma[state])
    #             log_likelihood = one + two + three / four



    #             # Sum up the likelihoods from all states
    #             log_likelihood_sum += log_likelihood

    #         # Taking the log of the summed likelihood
    #         log_likelihood_contributions[t] = np.log(log_likelihood_sum) if log_likelihood_sum > 0 else np.log(-np.inf)

            
    #     return log_likelihood_contributions

    def forward_pass(self):
        """
        Perform the forward pass of the Forward-Backward algorithm.
        Calculate the probability of being in each state at each time step,
        given the observations up to that point.
        """
        n_observations = len(self.data)
        forward_probabilities = np.zeros((n_observations, self.n_states))

        # Initialize forward probabilities for the first observation
        for state in range(self.n_states):
            forward_probabilities[0, state] = self.initial_state_probs[state] * self.emission_probability(state, 0)

        # Iterate over each time step
        for t in range(1, n_observations):
            for state in range(self.n_states):
                sum_prob = 0
                for prev_state in range(self.n_states):
                    sum_prob += forward_probabilities[t-1, prev_state] * self.transition_matrix[prev_state, state]
                forward_probabilities[t, state] = sum_prob * self.emission_probability(state, t)

        return forward_probabilities


    def backward_pass(self):
        """
        Perform the backward pass of the Forward-Backward algorithm.
        Calculate the probability of the upcoming observations given each state at each time step.
        """
        n_observations = len(self.data)
        backward_probabilities = np.zeros((n_observations, self.n_states))

        # Initialize backward probabilities for the last observation (usually set to 1)
        backward_probabilities[n_observations - 1, :] = 1

        # Iterate over each time step in reverse order
        for t in range(n_observations - 2, -1, -1):
            for state in range(self.n_states):
                sum_prob = 0
                for next_state in range(self.n_states):
                    sum_prob += backward_probabilities[t + 1, next_state] * \
                                self.transition_matrix[state, next_state] * \
                                self.emission_probability(next_state, t + 1)
                backward_probabilities[t, state] = sum_prob

        return backward_probabilities


    def calculate_state_probability(self, forward_probabilities, backward_probabilities):
        """
        Calculate the state probabilities (posterior probabilities or responsibilities)
        for each state at each time step, given the entire sequence of observations.
        """
        n_observations = len(self.data)
        state_probabilities = np.zeros((n_observations, self.n_states))

        # Calculate the normalization factor at each time step
        for t in range(n_observations):
            normalization_factor = np.sum([forward_probabilities[t, state] * backward_probabilities[t, state]
                                           for state in range(self.n_states)])
            
            for state in range(self.n_states):
                if normalization_factor > 0:
                    state_probabilities[t, state] = (forward_probabilities[t, state] *
                                                     backward_probabilities[t, state]) / normalization_factor
                else:
                    state_probabilities[t, state] = 0

        return state_probabilities

    def e_step(self,forward_probabilities, backward_probabilities):
        """
        Perform the E-step of the EM algorithm.
        Calculate the expected state responsibilities for each observation.
        """
        # Perform forward and backward passes
        # forward_probabilities = self.forward_pass()
        # backward_probabilities = self.backward_pass()

        # Calculate state probabilities (responsibilities)
        state_probabilities = self.calculate_state_probability(forward_probabilities, backward_probabilities)

        return state_probabilities

    
    def m_step(self, state_probabilities, forward_probabilities, backward_probabilities):
        """
        Perform the M-step of the EM algorithm.
        Update the model parameters based on the responsibilities calculated in the E-step.
        """
        n_observations = len(self.data)
        
        # Update transition matrix
        
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = 0
                denominator = 0
                for t in range(n_observations - 1):
                    # Prob(i at t, j at t+1 | observed data)
                    prob_transition = forward_probabilities[t, i] * \
                                      self.transition_matrix[i, j] * \
                                      self.emission_probability(j, t + 1) * \
                                      backward_probabilities[t + 1, j]

                    numerator += prob_transition
                    denominator += forward_probabilities[t, i] * backward_probabilities[t, i]

                self.transition_matrix[i, j] = numerator / denominator if denominator > 0 else 0

        # Update mu, phi, and sigma for each state
        for state in range(self.n_states):
            mu_numerator = 0
            mu_denominator = 0
            phi_numerator = 0
            phi_denominator = 0
            sigma_numerator = 0
            sigma_denominator = 0

            for t in range(1, n_observations):
                responsibility = state_probabilities[t, state]
                mu_numerator += responsibility * self.data[t]
                mu_denominator += responsibility
                phi_numerator += responsibility * self.data[t] * self.data[t - 1]
                phi_denominator += responsibility * self.data[t - 1] ** 2

                residual = self.data[t] - self.mu[state] - self.phi[state] * self.data[t - 1]
                sigma_numerator += responsibility * residual ** 2
                sigma_denominator += responsibility

            self.mu[state] = mu_numerator / mu_denominator if mu_denominator > 0 else 0
            self.phi[state] = phi_numerator / phi_denominator if phi_denominator > 0 else 0
            self.sigma[state] = (sigma_numerator / sigma_denominator if sigma_denominator > 0 else 0) ** 0.5
    
    def check_convergence(self, previous_log_likelihood, current_log_likelihood):
        """
        Check if the convergence criterion is met.
        """
        return abs(current_log_likelihood - previous_log_likelihood) < self.tolerance

    def fit(self, ): # max_iterations=100, tolerance=1e-6):
        """
        Run the EM algorithm, iterating through E-step and M-step until convergence.
        """
        previous_log_likelihood = 0

        for iteration in range(self.max_iterations):
            # Perform forward and backward passes
            forward_probabilities = self.forward_pass()
            backward_probabilities = self.backward_pass()
            # E-step: Calculate state probabilities
            state_probabilities = self.e_step(forward_probabilities, backward_probabilities)

            # Store the smoothed probabilities
            if iteration == self.max_iterations - 1:
                self.smoothed_probabilities = state_probabilities


            # M-step: Update model parameters
            self.m_step(state_probabilities,forward_probabilities, backward_probabilities)
            
            # Store parameter values in their histories
            self.mu_history.append(self.mu.copy())
            self.phi_history.append(self.phi.copy())
            self.sigma_history.append(self.sigma.copy())


            # Calculate current log-likelihood
            current_log_likelihood =  np.sum(self.calculate_likelihood())
            print(current_log_likelihood)
            # Check for convergence
            if self.check_convergence(previous_log_likelihood, current_log_likelihood, ):
                print(f"Convergence reached at iteration {iteration}.")
                break

            previous_log_likelihood = current_log_likelihood
            print(f' Current Iteration: {iteration},   Mu:  {self.mu},  phi:   {self.phi},    Sigma:  {self.sigma} ')

        print(f"EM algorithm completed. Total iterations: {iteration}.")

    def plot_convergence(self):
        # Ensure histories are available
        if not hasattr(self, 'mu_history') or not self.mu_history:
            print("No parameter history available. Run the fit method first.")
            return

        regimes = self.n_states

        # Plotting mu, phi, and sigma values
        for param_history, param_name in zip([self.mu_history, self.phi_history, self.sigma_history], 
                                             ['Mu', 'Phi', 'Sigma']):
            plt.figure(figsize=(12, 4))
            for state in range(regimes):
                param_values = [param[state] for param in param_history]
                plt.plot(param_values, label=f'{param_name} State {state}')
            plt.xlabel('Iteration')
            plt.ylabel(f'{param_name} Value')
            plt.title(f'{param_name} Values over Iterations')
            plt.legend()
            plt.show()

    def plot_smoothed_probabilities(self, true_states=None):
        # Ensure smoothed probabilities are available
        if not hasattr(self, 'smoothed_probabilities'):
            print("No smoothed probabilities available. Run the fit method first.")
            return

        num_states = self.n_states
        fig, axes = plt.subplots(num_states, 1, figsize=(12, 3 * num_states))

        for state in range(num_states):
            ax = axes[state] if num_states > 1 else axes
            ax.plot(self.smoothed_probabilities[:, state], label=f'State {state} Probability')
            ax.set_title(f'Smoothed Probability of being in State {state}')
            ax.set_xlabel('Observation')
            ax.set_ylabel('Probability')

            if true_states is not None:
                true_state_indicator = (true_states == state).astype(int)
                ax.scatter(range(len(true_states)), true_state_indicator, label=f'True State {state}', alpha=0.5, marker='o')

            ax.legend()

        plt.tight_layout()
        plt.show()
    # def plot_convergence(self.mu_history, self.phi_history, self.sigma_history):
    #     regimes = len(mu_history[0])

    #     # Plotting mu values
    #     plt.figure(figsize=(12, 4))
    #     for state in range(regimes):
    #         mus = [mu_history[i][state] for i in range(len(mu_history))]
    #         plt.plot(mus, label=f'Mu State {state}')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Mu Value')
    #     plt.title('Mu Values over Iterations')
    #     plt.legend()
    #     plt.show()

    #     # Plotting phi values
    #     plt.figure(figsize=(12, 4))
    #     for state in range(regimes):
    #         phis = [phi_history[i][state] for i in range(len(phi_history))]
    #         plt.plot(phis, label=f'Phi State {state}')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Phi Value')
    #     plt.title('Phi Values over Iterations')
    #     plt.legend()
    #     plt.show()

    #     # Plotting sigma values
    #     plt.figure(figsize=(12, 4))
    #     for state in range(regimes):
    #         sigmas = [sigma_history[i][state] for i in range(len(sigma_history))]
    #         plt.plot(sigmas, label=f'Sigma State {state}')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Sigma Value')
    #     plt.title('Sigma Values over Iterations')
    #     plt.legend()
    #     plt.show()

    # def plot_smoothed_probabilities(self.smoothed_probabilities, true_states=None):
    #     num_states = smoothed_probabilities.shape[1]
    #     fig, axes = plt.subplots(num_states, 1, figsize=(12, 3 * num_states))

    #     for state in range(num_states):
    #         ax = axes[state] if num_states > 1 else axes
    #         # Plot the smoothed probabilities for each state
    #         ax.plot(smoothed_probabilities[:, state], label=f'State {state} Probability')
    #         ax.set_title(f'Smoothed Probability of being in State {state}')
    #         ax.set_xlabel('Observation')
    #         ax.set_ylabel('Probability')

    #         if true_states is not None:
    #             # Overlay the true states if provided
    #             true_state_indicator = (true_states == state).astype(int)
    #             ax.scatter(range(len(true_states)), true_state_indicator, label=f'True State {state}', alpha=0.5, marker='o')

    #         ax.legend()

    #     plt.tight_layout()
    #     plt.show()










