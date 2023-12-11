import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
# Load and preprocess your data
# data = pd.read_csv('your_timeseries_data.csv')
# Define your MS-VAR model

class NsSvHamiltonFilter:
    def __init__(self, n_states, sigma2):
        self.n_states = n_states
        self.sigma2 = sigma2  # Variance in each state
        transition_matrix = np.random.rand(n_states, n_states)

        # Normalize each column to sum to 1
        self.transition_matrix = transition_matrix / transition_matrix.sum(axis=0, keepdims=True)

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














































class SvHamiltonFilter:
    def __init__(self, n_states, sigma2, p00, p11):
        self.n_states = n_states
        self.sigma2 = sigma2  # Variance in each state
        self.transition_matrix = np.array([[p00, 1 - p11], [1 - p00, p11]])

    def predict_step(self, xi_11):
        return self.transition_matrix.dot(xi_11)

    def filter_step(self, y_t, xi_10):
        eta = np.array([self.GaussianDensity(y_t, 0, self.sigma2[0]),
                        self.GaussianDensity(y_t, 0, self.sigma2[1])])

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
            eta = np.array([self.GaussianDensity(y[t], 0, self.sigma2[0]),
                            self.GaussianDensity(y[t], 0, self.sigma2[1])])

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

    def analyze_and_plot(self, y, estimated_params, standard_errors):
        # Unpack estimated parameters
        p00, p11, h1, h2 = estimated_params
        sigma2 = np.array([h1**2, h2**2])
        T = len(y)

        # Transition matrix
        P = np.array([[p00, 1 - p11], [1 - p00, p11]])

        # Forward filter and backward smoother
        xi_10, xi_11, xi_1T = self.forward_backward_filter(y, P, sigma2)

        # Compute filtered volatility
        vol = xi_11[0, :] * sigma2[0] + (1 - xi_11[0, :]) * sigma2[1]

        # Plotting
        self.plot_results(y, vol, xi_10, xi_11, xi_1T, T)

        # Print estimated parameters and standard errors
        print('P11 = {}, std.errors = {}'.format(estimated_params[0], standard_errors[0]))
        print('P22 = {}, std.errors = {}'.format(estimated_params[1], standard_errors[1]))
        print('h1 = {}, std.errors = {}'.format(estimated_params[2], standard_errors[2]))
        print('h2 = {}, std.errors = {}'.format(estimated_params[3], standard_errors[3]))
    
    def forward_backward_filter(self, y, P, sigma2):
        T = len(y)

        xi_10 = np.zeros((2, T + 1))
        xi_11 = np.zeros((2, T))
        xi_1T = np.zeros((2, T))

        # Initialize the state probabilities
        xi_10[:, 0] = self.initialize_state_probabilities()

        # Forward filtering
        for t in range(T):
            eta = np.array([self.GaussianDensity(y[t], 0, sigma2[0]),
                            self.GaussianDensity(y[t], 0, sigma2[1])])

            xi_11[:, t] = eta * xi_10[:, t] / (np.dot(eta, xi_10[:, t]))
            xi_10[:, t + 1] = P.dot(xi_11[:, t])

        # Backward smoothing
        xi_1T[:, T - 1] = xi_11[:, T - 1]
        for t in range(T - 2, -1, -1):
            xi_1T[:, t] = xi_11[:, t] * (P.T.dot(xi_1T[:, t + 1] / xi_10[:, t + 1]))

        return xi_10, xi_11, xi_1T

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
        fig, ax = plt.subplots(3, figsize=(16, 9))
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle('State Probabilities')

        # Plot predicted state probability
        ax[0].plot(1 - xi_10[0, :], label='P(s_t=1|...)')
        ax[0].set_xlim(0, T)
        ax[0].title.set_text('Predicted State Probability')
        ax[0].axhline(0, color='black', linestyle="--")
        ax[0].axhline(1, color='black', linestyle="--")

        # Plot filtered state probability
        ax[1].plot(1 - xi_11[0, :], label='P(s_t=1|...)')
        ax[1].set_xlim(0, T)
        ax[1].title.set_text('Filtered State Probability')
        ax[1].axhline(0, color='black', linestyle="--")
        ax[1].axhline(1, color='black', linestyle="--")

        # Plot smoothed state probability
        ax[2].plot(1 - xi_1T[0, :], label='P(s_t=1|...)')
        ax[2].set_xlim(0, T)
        ax[2].title.set_text('Smoothed State Probability')
        ax[2].axhline(0, color='black', linestyle="--")
        ax[2].axhline(1, color='black', linestyle="--")

        plt.show()

















# class HamiltonFilter:
#     """
#     Takes a dataset, a log likelihood function and the number of states to estimate.
#     :init - Initialize class.
#     :initialize_transition_matrix - Create Transition Matrix Given n_states.
#     :initialize_parameters - Create Parameters to estimate, based on log Likelihood Function.
#     """
#     def __init__(self, data, n_states):
#         """
#         Initialize the Hamilton Estimation Model.

#         :param log_likelihood_func: A function to compute the log-likelihood.
#         :param data: Time series data.
#         :param n_states: Number of states in the model.
#         """
#         self.log_likelihood_func = log_likelihood_func
#         self.data = data
#         self.num_obs = len(data)
#         self.n_states = n_states

#         # Initialize the transition matrix and parameters
#         self.transition_matrix = self.initialize_transition_matrix(n_states)
#         self.state_probabilities = np.ones(n_states) / n_states  # Uniform initial state probabilities
#         self.parameters = self.initialize_parameters()

#     def initialize_transition_matrix(self, n_states):
#         """
#         Initialize the state transition matrix with columns summing to 1.

#         :param n_states: Number of states.
#         :return: Initialized transition matrix.
#         """
#         transition_matrix = np.random.rand(n_states, n_states)
#         # Normalize each column to sum to 1
#         transition_matrix = transition_matrix / transition_matrix.sum(axis=0, keepdims=True)
#         return transition_matrix

#     def initialize_parameters(self, n_states):
#         """
#         Initialize parameters for each state.

#         :param n_states: Number of states.
#         :return: Dictionary of parameters for each state.
#         """
#         parameters = {}
#         for state in range(n_states):
#             parameters[state] = {
#                 'mu': np.random.rand(),   # Mean parameter for state
#                 'phi': np.random.rand(),  # Autoregressive parameter for state
#                 'sigma': np.random.rand() # Standard deviation for state
#             }
#         return parameters

#     def log_likelihood_contribution(self, t, state):
#         """
#         Calculate the log likelihood contribution for a given time and state.

#         :param t: Time index.
#         :param state: Current state.
#         :return: Log likelihood contribution.
#         """
#         if t == 0:
#             return 0  # No contribution at t = 0

#         mu = self.parameters[state]['mu']
#         phi = self.parameters[state]['phi']
#         sigma = self.parameters[state]['sigma']
#         data = self.data

#         residual = data[t] - mu - phi * data[t - 1]
#         return -0.5 * np.log(2 * np.pi * sigma) - 0.5 * (residual ** 2) / sigma

#     def predict(self):
#         """
#         Prediction step: Update state probabilities based on the transition matrix.
#         """
#         self.state_probabilities = np.dot(self.transition_matrix, self.state_probabilities)

#     def filter(self, t):
#         """
#         Filtering step: Update state probabilities based on the new observation.

#         :param t: Time index of the new observation.
#         """
#         if t >= len(self.data):
#             raise IndexError("Time index out of bounds")

#         likelihoods = np.array([self.log_likelihood_contribution(t, state) for state in range(self.n_states)])
#         likelihoods = np.exp(likelihoods - np.max(likelihoods))  # Prevent numerical underflow
#         self.state_probabilities *= likelihoods
#         self.state_probabilities /= self.state_probabilities.sum()  # Normalize probabilities

#     def fit(self):
#         """
#         Fit the model to the data by iterating over filtering and prediction steps.
#         """
#         T = len(self.data)
#         self.state_probabilities_history = np.zeros((self.n_states, T))  # To store state probabilities over time

#         for t in range(T):
#             # Filtering step
#             self.filter(t)

#             # Store the state probabilities at time t
#             self.state_probabilities_history[:, t] = self.state_probabilities

#             # Prediction step for the next time point
#             if t < T - 1:  # No prediction needed at the last time point
#                 self.predict()






# # class MSVAR:

#     def __init__(self, data, lags, regimes):
#         """
#         Initialize the MS-VAR model parameters.
#         :param data: pandas DataFrame, time series data.
#         :param lags: int, number of lags in the VAR model.
#         :param regimes: int, number of regimes in the Markov-switching model.
#         """
#         self.data = data
#         self.lags = lags
#         self.regimes = regimes
#         self.parameters = None  # Placeholder for model parameters

#     def initialize_msvar_params(num_vars, num_lags, num_regimes):
#         """
#         Initialize parameters for the MS-VAR model.
        
#         :param num_vars: int, number of variables (i.e., number of columns in the DataFrame).
#         :param num_lags: int, number of lags in the VAR model.
#         :param num_regimes: int, number of regimes in the Markov-switching model.
#         :return: dict, containing initialized parameters.


#         :Get columns of the Dataframe
#         :Get Lags of Autoregressive Element
#         :Get Lags of GARCH Element
#         :Get number of Regimes
#         Return Dictionary of the parameters with random values.
#         """
#         # Vector of constants (intercepts) for each variable, for each regime
#         mu = np.random.randn(num_regimes, num_vars)  # Example initialization

#         # Autocorrelation matrices for each lag, for each regime
#         autocorrelation_matrices = np.random.randn(num_regimes, num_lags, num_vars, num_vars)  # Example initialization

#         # Transition probability matrix for the regimes
#         transition_matrix = np.full((num_regimes, num_regimes), 1.0 / num_regimes)  # Uniform probabilities as a starting point

#         return {
#             "mu": mu,
#             "autocorrelation_matrices": autocorrelation_matrices,
#             "transition_matrix": transition_matrix
#         }

#     def e_step(data, params):
#         """
#         Perform the E-step of the EM algorithm for the MS-VAR model.
#         Estimate the state sequence and thie expected value of the latent variables.

#         Create a function for the expectation of the log-likelihood evalueated using the current estimate of the parameters. 
#         Given the underlying latent variables


#         :param data: The observed time series data.
#         :param params: The current estimates of the model parameters.
#         :return: Expected values of latent variables (regime states).
#         """
#         # Initialize the expected values of the latent variables
#         # For simplicity, this could be a matrix with dimensions [num_time_points x num_regimes]
#         expected_latent_vars = np.zeros((len(data), params['transition_matrix'].shape[0]))

#         # Implement the logic to update expected_latent_vars based on current parameters
#         # This will involve computing probabilities of being in each regime at each time point

#         return expected_latent_vars

#     def m_step(data, expected_latent_vars):
#         """
#         Perform the M-step of the EM algorithm for the MS-VAR model.
#         Calculate the local maximum Log-likelihood. Compute the parameters that maximizes the expected log-likelihood found in the E-step 


#         :param data: The observed time series data.
#         :param expected_latent_vars: The expected values of the latent variables from the E-step.
#         :return: Updated model parameters.
#         """
#         new_params = {
#             'mu': None,  # Placeholder for updated mu
#             'autocorrelation_matrices': None,  # Placeholder for updated autocorrelation matrices
#             'transition_matrix': None  # Placeholder for updated transition matrix
#         }

#         # Implement the logic to update the parameters based on the expected latent variables
#         # This will involve re-estimating mu, autocorrelation_matrices, and the transition_matrix

#         return new_params

#     def fit(self, data, initial_params, max_iterations=100, tolerance=1e-6):
#         """
#         Fit an MS-VAR model using the EM algorithm.

#         :param data: The observed time series data.
#         :param initial_params: Initial parameter estimates.
#         :param max_iterations: Maximum number of iterations.
#         :param tolerance: Convergence tolerance.
#         :return: Fitted model parameters.
#         """
#         params = initial_params
#         for iteration in range(max_iterations):
#             # E-step
#             expected_latent_vars = e_step(data, params)

#             # M-step
#             new_params = m_step(data, expected_latent_vars)

#             # Check for convergence (this can be based on the change in parameters)
#             if np.allclose(params, new_params, atol=tolerance):
#                 break

#             params = new_params

#         return params

#     def summarize(self):
#         """
#         print results for paramter values, test statistics, log-likelihood value
#         and regime values 
#         """
#         pass

#     def predict(self, steps):
#         """
#         Make predictions with the fitted model.
#         :param steps: int, number of steps to predict.
#         :return: numpy array, predicted values.
#         """
#         # Placeholder for prediction method
#         pass

#     def forecast(self, steps):
#         pass





# import numpy as np
# import pandas as pd
# from scipy.optimize import minimize

# class MS_ME:

#     def __init__(self, data, lags, regimes):
#         """
#         Initialize the MS-AR model parameters.
#         :param data: pandas DataFrame, time series data.
#         :param lags: int, number of lags in the AR model.
#         :param regimes: int, number of regimes in the Markov-switching model.
#         """
#         self.data = data.values if isinstance(data, pd.DataFrame) else data
#         self.lags = lags
#         self.regimes = regimes
#         self.parameters = self.initialize_params(self.lags, self.regimes)
#         #print(len(self.data))

#     def initialize_params(self, lags, regimes):
#         """
#         Initialize model parameters randomly.
#         :param lags: int, number of lags in the AR model.
#         :param regimes: int, number of regimes.
#         :return: dict, initialized parameters.
#         """
#         alpha = np.random.randn(regimes, lags)  # Alpha parameters for each regime
#         phi = np.random.randn(regimes, lags)  # Phi parameters for each regime

#         # Transition probability matrix for the regimes
#         transition_matrix = np.full((regimes, regimes), 1.0 / regimes)

#         return {
#             "alpha": alpha,
#             "phi": phi,
#             "transition_matrix": transition_matrix
#         }
#     def forward_pass(self, data, params):
#         # Print the type and length of data for debugging
#         # print("Type of data in forward_pass:", type(data))
#         # print("Length of data in forward_pass:", len(data))
#         num_data_points = len(data)
#         alpha = params['alpha']
#         phi = params['phi']
#         transition_matrix = params['transition_matrix']

#         # Initializing filtered probabilities array
#         filtered_probs = np.zeros((num_data_points, self.regimes))

#         # Initial state probabilities (can be uniform or based on prior knowledge)
#         filtered_probs[0, :] = 1.0 / self.regimes

#         for t in range(1, num_data_points):
#             for state in range(self.regimes):
#                 # Calculate the probability of the data point given the state
#                 # This requires a likelihood function for your AR model
#                 likelihood = self.calculate_likelihood(data[t], data[t-1], alpha[state], phi[state])

#                 # Update filtered probability for each state
#                 filtered_probs[t, state] = likelihood * np.sum(filtered_probs[t-1, :] * transition_matrix[:, state])

#             # Normalize probabilities to sum to 1
#             filtered_probs[t, :] /= np.sum(filtered_probs[t, :])

#         return filtered_probs

#     def backward_pass(self, filtered_probs, params):
#         num_data_points = len(filtered_probs)
#         transition_matrix = params['transition_matrix']

#         # Initializing smoothed probabilities array
#         smoothed_probs = np.copy(filtered_probs)

#         for t in range(num_data_points - 2, -1, -1):
#             for state in range(self.regimes):
#                 # Compute the backward probability
#                 backward_prob = np.sum(transition_matrix[state, :] * smoothed_probs[t + 1, :])

#                 # Update smoothed probability
#                 smoothed_probs[t, state] *= backward_prob

#             # Normalize probabilities to sum to 1
#             smoothed_probs[t, :] /= np.sum(smoothed_probs[t, :])

#         return smoothed_probs

#     def calculate_state_probability(self, data, state, params, t):
#         """
#         Calculate smoothed state probabilities at time t using forward-backward algorithm.
#         :param data: pandas DataFrame, time series data.
#         :param params: dict, current model parameters.
#         :param t: int, time step.
#         :return: numpy array, smoothed state probabilities at time t.
#         """
#         filtered_probs = self.forward_pass(data, params)
#         smoothed_probs = self.backward_pass(filtered_probs, params)
#         return smoothed_probs[t,state]# :]

#     def calculate_likelihood(self, current_obs, previous_obs, alpha, phi, sigma=1.0):
#         """
#         Calculate the likelihood of observing the current data point given the state-specific AR parameters.
#         :param current_obs: float, current observation.
#         :param previous_obs: float, previous observation.
#         :param alpha: float, state-specific alpha parameter.
#         :param phi: float, state-specific phi parameter.
#         :param sigma: float, standard deviation of the error term (default is 1.0 for standard normal distribution).
#         :return: float, likelihood of the current observation.
#         """
#         predicted_obs = alpha + phi * previous_obs
#         likelihood = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((current_obs - predicted_obs) / sigma)**2)
#         return likelihood


#     def e_step(self, data, params):
#         """
#         E-step - estimate expected values of latent variables.
#         :param data: pandas DataFrame, time series data.
#         :param params: dict, current model parameters.
#         :return: numpy array, expected latent states.
#         """
#         num_data_points = len(data)
#         expected_states = np.zeros((num_data_points, self.regimes))

#         # Example: Calculating the expected state probabilities
#         for t in range(1, num_data_points):
#             for state in range(self.regimes):
#                 # Probability of being in 'state' at time t given data and parameters
#                 # Placeholder for actual logic:
#                 # You need to implement how to calculate these probabilities
#                 # based on the current parameters and the observed data.
#                 prob_state_given_data = self.calculate_state_probability(data, state, params, t)
#                 expected_states[t, state] = prob_state_given_data

#         self.smoothed_probabilities = expected_states
#         return expected_states

   

#     def m_step(self, data, expected_states):
#         """
#         M-step - maximize the expected log-likelihood with respect to the parameters.
#         :param data: pandas DataFrame or numpy array, time series data.
#         :param expected_states: numpy array, expected state probabilities from E-step.
#         """
#         num_data_points = len(data)
#         # new_params = {}
#         new_alpha = np.zeros((self.regimes, self.lags))
#         new_phi = np.zeros((self.regimes, self.lags))
#         # Updating parameters for each state
#         for state in range(self.regimes):
#             # Defining the objective function to be minimized (negative log-likelihood)
#             def objective(params):
#                 alpha, phi = params
#                 log_likelihood = 0
#                 for t in range(1, num_data_points):
#                     likelihood = self.calculate_likelihood(data[t], data[t-1], alpha, phi)
#                     # Weighted log-likelihood for this state
#                     log_likelihood += expected_states[t, state] * np.log(likelihood)
#                 return -log_likelihood

#             # Initial guesses for alpha and phi
#             # print(self.parameters['alpha'][state,0], self.parameters['phi'][state,0])
#             initial_guess = np.array([self.parameters['alpha'][state,0], self.parameters['phi'][state,0]])

#             # Numerical optimization
#             result = minimize(objective, initial_guess, method='L-BFGS-B')
#             new_alpha[state, 0], new_phi[state, 0] = result.x

#             # Update parameters for this state
#             #new_params[state] = {'alpha': new_alpha, 'phi': new_phi}
#             # Update transition probabilities
#         num_transitions = np.zeros((self.regimes, self.regimes))

#         for t in range(1, num_data_points):
#             for i in range(self.regimes):
#                 for j in range(self.regimes):
#                     num_transitions[i, j] += expected_states[t-1, i] * expected_states[t, j]

#         # Normalizing to get probabilities
#         new_transition_matrix = num_transitions / num_transitions.sum(axis=1, keepdims=True)

#         # Update parameters
#         self.parameters = {
#             'alpha': new_alpha,
#             'phi': new_phi,
#             'transition_matrix': new_transition_matrix  # Use the new transition matrix
#         }


#     def fit(self, max_iterations=100, tolerance=1e-6):
#         """
#         Fit the MS-AR model using the EM algorithm.
#         :param max_iterations: int, maximum number of iterations for the EM algorithm.
#         :param tolerance: float, convergence tolerance for the EM algorithm.
#         """
#         self.alpha_history = []
#         self.phi_history = []

#         log_likelihood_old = -np.inf

#         for iteration in range(max_iterations):
#             # E-step: Estimate expected state probabilities
#             expected_states = self.e_step(self.data, self.parameters)

#             # M-step: Update model parameters
#             self.m_step(self.data, expected_states)


#             self.alpha_history.append(self.parameters['alpha'].copy())
#             self.phi_history.append(self.parameters['phi'].copy())

#             # Calculating the new log-likelihood
#             log_likelihood_new = self.calculate_total_log_likelihood(self.data, self.parameters, expected_states)

#             # Check for convergence
#             if np.abs(log_likelihood_new - log_likelihood_old) < tolerance:
#                 print(f"EM algorithm converged at iteration {iteration}")
#                 break

#             log_likelihood_old = log_likelihood_new
#             print(f"Current step: {iteration} out of {max_iterations}... With Estimated Parameters Currently at {self.parameters}" )
#         else:
#             print("EM algorithm did not converge")

#     def calculate_total_log_likelihood(self, data, params, expected_states):
#         """
#         Calculate the total log-likelihood of the data given the model parameters and expected states.
#         :param data: pandas DataFrame or numpy array, time series data.
#         :param params: dict, current model parameters.
#         :param expected_states: numpy array, expected state probabilities from E-step.
#         :return: float, total log-likelihood.
#         """
#         total_log_likelihood = 0
#         num_data_points = len(data)

#         for t in range(1, num_data_points):
#             for state in range(self.regimes):
#                 # print(params['alpha'])
#                 # print(params['phi'])
#                 alpha = params['alpha'][state, 0]  # Access alpha for the current state
#                 phi = params['phi'][state, 0]      # Access phi for the current state
#                 likelihood = self.calculate_likelihood(data[t], data[t-1], alpha, phi)
#                 total_log_likelihood += expected_states[t, state] * np.log(likelihood)
                
                
#         return total_log_likelihood


#     def predict(self):
#         if self.parameters is None:
#             print("Model is not yet fitted.")
#             return None

#         num_data_points = len(self.data)
#         predictions = np.zeros(num_data_points)
        
#         # Initialize the first state (this could be based on initial probabilities or set to a default)
#         current_state = 0  # or some initialization logic

#         for t in range(1, num_data_points):
#             # Get the parameters for the current state
#             alpha = self.parameters['alpha'][current_state, 0]
#             phi = self.parameters['phi'][current_state, 0]
            
#             # Predict the next value
#             predictions[t] = alpha + phi * self.data[t-1]
            
#             # Update the state (this is a simplification; in practice, you'd use the transition probabilities)
#             # Example: current_state = np.random.choice(self.regimes, p=transition_probabilities[current_state])
#             # For simplicity, we keep the state constant in this example.

#         return predictions

#     def plot_results(self):
#         """
#         Plot Results of the estimation.
#         - Plot the convergence of parameters
#         - Plot residuals, and distribution
#         - Plot the Smoothed probabilities. 
#         """
#         # Plotting alpha values
#         plt.figure(figsize=(12, 6))
#         for state in range(self.regimes):
#             alphas = [alpha_history[state, 0] for alpha_history in self.alpha_history]
#             plt.plot(alphas, label=f'Alpha State {state}')
#         plt.xlabel('Iteration')
#         plt.ylabel('Alpha Value')
#         plt.title('Alpha Values over Iterations')
#         plt.legend()
#         plt.show()

#         # Plotting phi values
#         plt.figure(figsize=(12, 6))
#         for state in range(self.regimes):
#             phis = [phi_history[state, 0] for phi_history in self.phi_history]
#             plt.plot(phis, label=f'Phi State {state}')
#         plt.xlabel('Iteration')
#         plt.ylabel('Phi Value')
#         plt.title('Phi Values over Iterations')
#         plt.legend()
#         plt.show()

#     def plot_convergence(self, true_phi=None, true_alpha=None):
#         # Plotting alpha values
#         plt.figure(figsize=(12, 3))
#         for state in range(self.regimes):
#             alphas = [alpha_history[state, 0] for alpha_history in self.alpha_history]
#             plt.plot(alphas, label=f'Alpha State {state}')
#             if true_alpha is not None:
#                 plt.axhline(y=true_alpha[state], color='r', linestyle='--', label=f'True Alpha State {state}')
#         plt.xlabel('Iteration')
#         plt.ylabel('Alpha Value')
#         plt.title('Alpha Values over Iterations')
#         plt.legend()
#         plt.show()

#         # Plotting phi values
#         plt.figure(figsize=(12, 3))
#         for state in range(self.regimes):
#             phis = [phi_history[state, 0] for phi_history in self.phi_history]
#             plt.plot(phis, label=f'Phi State {state}')
#             if true_phi is not None:
#                 plt.axhline(y=true_phi[state], color='g', linestyle='--', label=f'True Phi State {state}')
#         plt.xlabel('Iteration')
#         plt.ylabel('Phi Value')
#         plt.title('Phi Values over Iterations')
#         plt.legend()
#         plt.show()

#     def plot_smoothed_probabilities(self, true_states=None):
#         if not hasattr(self, 'smoothed_probabilities'):
#             print("No smoothed probabilities available. Run fit method first.")
#             return

#         num_states = self.regimes
#         fig, axes = plt.subplots(num_states, 1, figsize=(12, 3 * num_states))

#         for state in range(num_states):
#             ax = axes[state] if num_states > 1 else axes
#             # Plot the smoothed probabilities for each state
#             ax.plot(self.smoothed_probabilities[:, state], label=f'State {state} Probability')
#             ax.set_title(f'Smoothed Probability of being in State {state}')
#             ax.set_xlabel('Observation')
#             ax.set_ylabel('Probability')

#             if true_states is not None:
#                 # Overlay the true states if provided
#                 true_state_indicator = (true_states == state).astype(int)
#                 ax.scatter(range(len(true_states)), true_state_indicator, label=f'True State {state}', alpha=0.5, marker='o')

#             ax.legend()

#         plt.tight_layout()
#         plt.show()


#     def plot_model_diagnostics(self):
#         if not hasattr(self, 'data') or self.parameters is None:
#             print("Model or data is not available.")
#             return

#         estimated_values = self.predict()  # Assuming 'predict' method exists and returns estimated values
#         residuals = self.data - estimated_values

#         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
#         # Colors
#         observed_color = 'skyblue'
#         estimated_color = 'salmon'
#         residual_color = 'lightgreen'
#         histogram_color = 'plum'
#         normal_dist_color = 'darkred'

#         # Plot 1: Estimation vs. Data
#         axes[0, 0].plot(self.data, label='Observed Data', color=observed_color)
#         axes[0, 0].plot(estimated_values, label='Estimated Data', color=estimated_color, linestyle='--')
#         axes[0, 0].set_title('Observed vs. Estimated Data')
#         axes[0, 0].set_xlabel('Time')
#         axes[0, 0].set_ylabel('Value')
#         axes[0, 0].legend()

#         # Plot 2: Residuals
#         axes[0, 1].plot(residuals, label='Residuals', color=residual_color)
#         axes[0, 1].set_title('Residuals over Time')
#         axes[0, 1].set_xlabel('Time')
#         axes[0, 1].set_ylabel('Residual')
#         axes[0, 1].legend()

#         # Plot 3: Histogram of Residuals
#         sns.histplot(residuals, kde=False, ax=axes[1, 0], color='blue', label='Residuals')
#         xmin, xmax = axes[1, 0].get_xlim()
#         x = np.linspace(xmin, xmax, 100)
#         p = np.exp(-x**2/2) / np.sqrt(2 * np.pi)
#         scale_factor = len(residuals) * (xmax - xmin) / (np.max(np.histogram(residuals, bins=100)[0]))
#         axes[1, 0].plot(x, p * scale_factor, label='Normal Distribution', color=normal_dist_color)
#         axes[1, 0].set_title('Histogram of Residuals')
#         axes[1, 0].legend()
        
#         # Plot 4: Autocorrelation of Residuals
#         plot_acf(residuals, ax=axes[1, 1], alpha=0.05, color='steelblue')
#         axes[1, 1].set_title('Autocorrelation of Residuals')

#         plt.tight_layout()
#         plt.show()



#     def summary(self):
#         """
#         Create a summary of the model estimation results.
#         """
#         if self.parameters is None:
#             print("Model is not yet fitted.")
#             return

#         # Displaying Model Parameters
#         print("Model Parameters:")
#         for state in range(self.regimes):
#             print(f"State {state}:")
#             print(f"  Alpha: {self.parameters['alpha'][state,0]}")
#             print(f"  Phi: {self.parameters['phi'][state,0]}")
#         print("\n")

#         # Final Log-Likelihood
#         expected_states = self.e_step(self.data, self.parameters)
#         final_log_likelihood = self.calculate_total_log_likelihood(self.data, self.parameters, expected_states)
#         print(f"Final Log-Likelihood: {final_log_likelihood}")
#         print("\n")

#         # Test Statistics (e.g., AIC, BIC)
#         num_params = self.regimes * 2  # Number of parameters (alpha and phi for each state)
#         aic = -2 * final_log_likelihood + 2 * num_params
#         bic = -2 * final_log_likelihood + np.log(len(self.data)) * num_params
#         print("Test Statistics:")
#         print(f"  AIC: {aic}")
#         print(f"  BIC: {bic}")
#         print("\n")

#         # State Probabilities
#         print("Estimated State Probabilities:")
#         for t, probs in enumerate(expected_states):
#             print(f"Time {t}: {probs}")


       
# # Example Usage
# # data = pd.read_csv('your_data.csv')  # Load your time series data here
# # model = MS_ME(data, lags=2, regimes=2)
# # model.fit(max_iterations=100, tolerance=1e-6)
# # predictions = model.predict(steps=10)
