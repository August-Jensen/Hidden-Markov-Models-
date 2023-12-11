import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from scipy.optimize import minimize



class Hamilton():
    def __init__(self, n_states, Gamma0, data):
        """
        We need to implement the max_iterations, and tolerance
        """
        # Setup of Input
        self.n_states = n_states
        self.data = data
        self.num_obs = len(data)

        # Setup of Probabilities
        self.transition_matrix = self.initialize_transition_matrix(n_states)
        self.initial_state_probs = np.full(n_states, 1.0 / n_states)

        #Setup of Parameters
        self.mu = [0,0] # (np.random.rand(n_states)* 2)-1  # or set to a specific starting value [-0.4,0,0.4] # 
        self.phi = [0,0] # (np.random.rand(n_states) * 2)-1 # or set to a specific starting value [-0.1,0,0.1]
        self.sigma =  Gamma0[3:4]#np.random.rand(n_states)*3   #[0.1,0.1,0.1]
        print(self.mu, self.phi, self.sigma)

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
        variance = self.sigma[state] ** 2
        emissions = norm.pdf(self.data[t], mean,variance)
        #print(emissions)
        return  emissions


    def filtering_step(self, t):
        # Calculate emission probabilities for each state
        emission_probs = np.array([self.emission_probability(state, t) for state in range(self.n_states)])

        if t == 0:
            predicted_prob = self.initial_state_probs
        else:
            predicted_prob = self.predicted_probability[:, t-1]

        # Calculate filtered probabilities
        numerator = emission_probs * predicted_prob
        denominator = np.sum(numerator)
        filtered_prob = numerator / denominator

        # Store results
        self.filtered_probability[:, t] = filtered_prob
        self.emission_probabilities[:, t] = emission_probs

    def prediction_step(self, t):
        """
        Calculate the predicted probabilities for time t+1 based on the 
        filtered probabilities at time t and the transition matrix.
        """
        if t == 0:
            # For the first step, use the initial state probabilities
            self.predicted_probability[:, 1] = self.initial_state_probs
        else:
            for state in range(self.n_states):
                # Calculate the predicted probability for each state at t+1
                self.predicted_probability[state, t+1] = np.sum(
                    self.transition_matrix[:, state] * self.filtered_probability[:, t]
                )

    def calculate_log_likelihood_contribution(self, t):
        """
        Calculate the log likelihood contribution for time t.
        """
        x = self.data[t]

        # Compute log likelihood for each state
        log_likelihoods = np.zeros(self.n_states)
        for state in range(self.n_states):
            mean = self.mu[state]
            variance = self.sigma[state] ** 2
            log_likelihoods[state] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(variance) - ((x - mean) ** 2) / (2 * variance)

        # Multiply each state's log likelihood by its emission probability and sum
        log_likelihood_contribution = np.sum(log_likelihoods * self.emission_probabilities[:, t])
        return log_likelihood_contribution

    def filter_algorithm(self, params):
        """
        Run the filter algorithm and calculate the negative sum of log likelihood contributions.
        """
        self.mu, self.phi, self.sigma = params[:self.n_states], params[self.n_states:2*self.n_states], params[2*self.n_states:]

        # setup numpy arrays 
        # Initialize Arrays for storing values
        self.predicted_probability = np.zeros([self.n_states, self.num_obs + 1])
        self.filtered_probability = np.zeros([self.n_states,self.num_obs])
        self.smoothed_probabilities = np.zeros([self.n_states,self.num_obs])
        self.likelihood_contributions = np.zeros(self.num_obs)
        self.emission_probabilities = np.zeros([self.n_states, self.num_obs])

        # Set the first predicted_probability to be initial_state_probs
        self.predicted_probability[:, 0] = self.initial_state_probs

        for t in range(self.num_obs):
            # Perform filtering_step at time t
            self.filtering_step(t)

            # Perform the prediction step for the next time point
            self.prediction_step(t)

            # Calculate and store the log likelihood contribution for time t
            self.likelihood_contributions[t] = self.calculate_log_likelihood_contribution(t)
    
    def fit(self, initial_guess, bounds):
        """
        Fit the model to the data using scipy.optimize.minimize.
        """
        res = minimize(self.filter_algorithm, initial_guess, method='L-BFGS-B', bounds=bounds)
        Gamma_hat = res.x
        v_hessian = res.hess_inv.todense()
        se_hessian = np.sqrt(np.diagonal(v_hessian))

        # Print results
        for i in range(len(Gamma_hat)):
            print(f'Parameter {i}: {Gamma_hat[i]}, standard error: {se_hessian[i]}')

        return Gamma_hat, se_hessian