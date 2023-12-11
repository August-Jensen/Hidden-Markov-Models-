import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class DataGeneratingProcess(object):
    """docstring for DataGeneratingProcess"""
    def __init__(self, n_states=3, n_obs=100, process_type="Constant", transition_matrix=None):
        self.n_states = n_states
        self.n_obs = n_obs
        self.transition_matrix = transition_matrix if transition_matrix is not None else self.initialize_transition_matrix(n_states)
        if isinstance(process_type, dict):
            self.parameters = self.trim_parameters_to_n_states(process_type)
        elif isinstance(process_type, str):
            if process_type == "Constant":
                self.parameters = self.trim_parameters_to_n_states({
                    'dist': 'normal',
                    'nu': [10, 15, 25, 65, 100],
                    'omega': [0.01, 0.03, 0.02, 0.01, 0.01],
                    'alpha': [0.0, 0, 0, 0, 0],
                    'beta': [0, 0, 0, 0, 0],
                    'mu': [-0.3, 0.3, 0, 0.5, -0.5],
                    'phi': [0, 0, -0, 0, -0],
                    'phi_2': [0,-0, -0, 0,-0],
                    'phi_3': [-0, 0,-0,0,-0]
                })
            elif process_type == "AR(1)":
                # Define and trim parameters for "AR(1)"
                self.parameters = self.trim_parameters_to_n_states({
                    # Define your parameters for "AR(1)" here
                })
            elif process_type == "Random":
                # Generate random parameters using internal bounds
                self.parameters = self.generate_random_parameters()
            else:
                raise ValueError(f"Unknown process type: {process_type}")
        else:
            raise ValueError("process_type must be a string or a dictionary")


    def initialize_transition_matrix(self, n_states,):
        """
        Initializes the transition matrix with specified properties.
        """
        # Create an empty matrix
        if n_states > 2:
            matrix = np.zeros((n_states, n_states))

            # Fill the diagonal with values between 0.95 and 1
            pii_values = np.random.uniform(0.98, 1, size=n_states)

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
            pii_values = np.random.uniform(0.98, 1, size=n_states)
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


    def generate_random_parameters(self):
        # Default bounds if none are provided
        bounds_01 = {key: (0, 1) for key in ['omega', 'alpha', 'beta']}
        bounds_1 = {key: (-1, 1) for key in ['mu', 'phi', 'phi_2', 'phi_3']}
        bounds = {**bounds_01, **bounds_1}

        random_params = {}
        for key in bounds.keys():
            if key == 'nu':
                random_params[key] = np.random.randint(2, 101, size=5)
            else:
                lower, upper = bounds[key]
                random_params[key] = np.random.uniform(lower, upper, size=5)

        # Set 'dist' to a random choice between 'normal' and 't-distribution'
        random_params['dist'] = np.random.choice(['normal', 't-distribution'])

        return self.trim_parameters_to_n_states(random_params)

        
    def trim_parameters_to_n_states(self, params):
        trimmed_params = {}
        for key, value in params.items():
            if isinstance(value, list) and len(value) > self.n_states:
                trimmed_params[key] = value[:self.n_states]
            else:
                trimmed_params[key] = value
        return trimmed_params

    # def initialize_transition_matrix(self, n_states):
    #     """
    #     Initializes the transition matrix with specified properties.
    #     """
    #     # Create an empty matrix
    #     if n_states > 2:
    #         matrix = np.zeros((n_states, n_states))

    #         # Fill the diagonal with values between 0.95 and 1
    #         pii_values = np.random.uniform(0.95, 1, size=n_states)
    #         np.fill_diagonal(matrix, pii_values)

    #         # Set the off-diagonal values in each column
    #         for i in range(n_states):
    #             # Calculate the value to fill for off-diagonal elements
    #             off_diagonal_value = (1 - pii_values[i]) / (n_states - 2)

    #             # Fill off-diagonal elements except the last one
    #             for j in range(n_states):
    #                 if j != i:
    #                     matrix[j, i] = off_diagonal_value

    #             # Adjust the last off-diagonal element in each column
    #             matrix[-1, i] = 1 - matrix[:, i].sum() + matrix[-1, i]
    #     elif n_states == 2:
    #         matrix = np.zeros((n_states, n_states))

    #         # Fill the diagonal with values between 0.95 and 1
    #         pii_values = np.random.uniform(0.95, 1, size=n_states)
    #         np.fill_diagonal(matrix, pii_values)

    #         # Set the off-diagonal values in each column
    #         for i in range(n_states):
    #             # Calculate the value to fill for off-diagonal elements
    #             off_diagonal_value = (1 - pii_values[i]) 
    #             # Fill off-diagonal elements except the last one
    #             for j in range(n_states):
    #                 if j != i:
    #                     matrix[j, i] = off_diagonal_value

    #             # Adjust the last off-diagonal element in each column
    #             matrix[-1, i] = 1 - matrix[:, i].sum() + matrix[-1, i]

    #     # Print the transition matrix and the sum of each column
    #     # print("Transition Matrix:\n", matrix)
    #     # print("Sum of each column:", matrix.sum(axis=0))

    #     return matrix


    def generate_data(self):
        # Initialize arrays
        states = np.zeros(self.n_obs, dtype=int)
        x = np.zeros(self.n_obs)
        sigma = np.zeros(self.n_obs)
        z = np.zeros(self.n_obs)

        # Set initial values
        sigma[0] = 0
        x[0] = 0

        # Choose initial state randomly
        states[0] = np.random.choice(self.n_states)

        for t in range(1, self.n_obs):
            # Determine the state in period t based on the transition matrix
            states[t] = np.random.choice(self.n_states, p=self.transition_matrix[:, states[t-1]])

            # Generate noise term z
            if self.parameters['dist'] == 'normal':
                z[t] = np.random.normal(0, 1)
            elif self.parameters['dist'] == 't-distribution':
                z[t] = np.random.standard_t(self.parameters['nu'][states[t]])

            # Calculate sigma[t]
            omega, alpha, beta = self.parameters['omega'], self.parameters['alpha'], self.parameters['beta']
            sigma[t] = (omega[states[t]] + alpha[states[t]] * (x[t-1] ** 2) + beta[states[t]] * (sigma[t-1] ** 2)) ** 0.5

            # Calculate x[t]
            mu, phi, phi_2, phi_3 = self.parameters['mu'], self.parameters['phi'], self.parameters['phi_2'], self.parameters['phi_3']
            x[t] = mu[states[t]] + phi[states[t]] * x[t-1] + phi_2[states[t]] * x[t-2] + phi_3[states[t]] * x[t-3] + z[t] * sigma[t]

        # Creating a DataFrame for the generated data
        data = pd.DataFrame({
            'State': states,
            'X': x,
            'Noise': z,
            'Sigma': sigma
        })

        return data


    def plot_data(self, data):
        fig, ax1 = plt.subplots()

        # Plot x data as a line plot
        ax1.plot(data['X'], color='b', label='X data')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('X data', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create a twin axis to plot states
        ax2 = ax1.twinx()
        ax2.scatter(range(self.n_obs), data['State'], color='r', label='States', alpha=0.6)
        ax2.set_ylabel('States', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Title and layout
        plt.title('States and X data over Time')
        fig.tight_layout()
        plt.show()

    def generate(self):
        # Generate the data
        data = self.generate_data()

        # Plot the data
        self.plot_data(data)
        print(self.transition_matrix)
        # Print the data and its descriptive statistics
        print(data)
        print(data.describe())

        return data
