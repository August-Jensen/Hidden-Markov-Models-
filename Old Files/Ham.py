# Imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from scipy.optimize import minimize



# Filter Class
class HamiltonFilter():
 	"""docstring for HamiltonFilter"""
 	def __init__(self, n_states, data):
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
        self.mu = (np.random.rand(n_states)* 2)-1  # or set to a specific starting value [-0.4,0,0.4] # 
        self.phi = (np.random.rand(n_states) * 2)-1 # or set to a specific starting value [-0.1,0,0.1]
        self.sigma = np.random.rand(n_states)*3   #[0.1,0.1,0.1]
        print(self.mu, self.phi, self.sigma)

        # Initialize matrix for storing responsibilities (E-step)
        self.responsibilities = np.zeros((len(data), n_states))


        # Initialize Arrays for storing values
        self.predicted_probability = np.zeros([n_states, num_obs + 1])
        self.filtered_probability = np.zeros([n_states,num_obs])
        self.smoothed_probabilities = np.zeros([n_states,num_obs])
        self.likelihood_contributions = np.zeros(num_obs)

        # Initialize Histories for Plotting Parameters
        self.mu_history = []
        self.phi_history = []
        self.sigma_history = []

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

        mean = self.mu[state] + self.phi[state] * previous_x
        variance = self.sigma[state] ** 2
        emissions = norm.pdf(self.data[t], mean,variance)
        #print(emissions)
        return  emissions

    def likelihood_contributions(self, t):
    	"""
    	Calculate the likelihood contribution for each state in period t
    	"""
    	likelihood = 0
    	for state in range(self.n_states):
    		likelihood += self.predicted_probability[state,t] * self.emission_probability(state, t)
    	return likelihood

 	def prediction_step(self):
 		"""
 		Calculate probability of being in state i at time t, given observed data to time t-1 
 		P(s_t = i | X_0:t-1 ; theta) = sum_j=1^n_states p_ji P(s_t-1=j | X_0:t-1 ; theta)
 		1. Prediction_prob list
 		2. for state in range(n_states): column of transition matrix * Probability of last state being 'state', given data to last period.
        """
        self.predicted_probability[:,t+1] = self.transition_matrix.dot(self.filtered_probability[:,t])
 		
 	def filter_step(self,t):
  		"""
        
        """
		for state in range(self.n_states):
			numerator = self.emission_probability(state, t) * self.predicted_probability[state, t]
        	self.filtered_probability[state, t] = numerator / self.likelihood_contribution(t)

 	def smoothed_probabilities(self):
 		"""
        Calculate the emission probability for a given state and time t.
        """
        pass

 	def optimize_parameters(self):
 		"""
 		Set up transition probability and predicted, & filtered probabilities.
 		Set first predicted probabilities, as the intitial state probabilities.
        Run the for loop t in data, 
        	Filter_sum = []
        	Run the for loop state in n_states
        		Emission probability
        		Filtering[state] = Emission * predicted_probability[state, t]
        		Filter_sum.append Filtering[state]
			
			sum filter_sum

        	Run the for loop, state in n_states
        		Filtering_probability[[state],t] = 	fil
				

        """
        pass


        self.initialize_arrays()  # Make sure all arrays are initialized appropriately
	    for t in range(self.num_obs):
	        self.filtering_step(t)
	        if t < self.num_obs - 1:  # Check to avoid index error on the last prediction
	            self.prediction_step(t)
	        self.likelihood_contributions(t)


 	def fit(self):
 		"""
        
        """

        for t in range(num_obs):
	        # Vectorized computation of emission probabilities for each state
	        emission_probs = np.array([self.emission_probability(observation, state) for state in range(n_states)])
	        
	        # Efficient computation of filtered probabilities
	        numerator = emission_probs * self.predicted_probability
	        denominator = np.sum(numerator)
	        self.filtered_probabilities = numerator / denominator

        	for state in n_states:
        		emissions[state] = emission_probability() # Insert data
        		total_filter += emissions[state]*predicted_probability[state,t]
"""
Fit should run optimize_parameters.
	Optimize Parameters should use scipy.opt opt.minimize(likelihood, Gamma0, method='L-BFGS-B',bounds=((0.001,0.9999),(0.001,0.9999),(0.01,None),(0.01,None)))
	<likelihood, is the regression, gamma0 is the initial guesses, and bounds are the parameter values. first 2 are probabilities, second are parameters.
		likelihood should take gamma0 parameters, and turn them into transition probabilities, create the transition matrix, and a regression that generates the residuals.
		    xi_10      = np.zeros([2,T+1]) - <1-xi_10> Predicted state probability, $P(s_t=1|x_{t-1},x_{t-2},...,x_{1})$
		    xi_11      = np.zeros([2,T]) - <1-xi_11> Filtered state probability, $P(s_t=1|x_{t},x_{t-1},...,x_{1})$
		    xi_1T      = np.zeros([2,T]) - <1-xi_1T> Smoothed state probability, $P(s_t=1|x_{T},x_{T-1},...,x_{1})$
		    lik        = np.zeros(T)
			Forward filter - for t in range(len(self.data)): 
				for state in n_states
					Find emission_probability
					Calculate likelihood contribution
					Filtering step
						num0=eta[0]*xi_10[0,t]/(eta[0]*xi_10[0,t]+eta[1]*xi_10[1,t])
				        num1=eta[1]*xi_10[1,t]/(eta[0]*xi_10[0,t]+eta[1]*xi_10[1,t])
				        xi_11[[0,1],t] = num0,num1

					Prediction Step  
						xi_10[[0,1],t+1] = P.dot(xi_11[[0,1],t])
				
				print log-likelihood
				return - sum of likelihood contribution
"""



"""
sigma2 = np.var(y) #used for initial guesses for sigma2 vals
Gamma0  =np.array([0.95,0.95,np.sqrt(2*sigma2),np.sqrt(0.5*sigma2)]) #initial guesses
res=opt.minimize(likelihood, Gamma0, method='L-BFGS-B',bounds=((0.001,0.9999),(0.001,0.9999),(0.01,None),(0.01,None))) #optimizing. We use L-BFGS-B as it allows for bounds and can compute the standard errors (from the inverse hessian) right away
res.x
v_hessian=res.hess_inv.todense() #retrieves the negative inverse hessian matrix (note we have minimized the negative log likelihood function)
se_hessian=np.sqrt(np.diagonal(v_hessian))

#result of optimization
Gamma_hat=res.x
se=se_hessian
print('P11='+str(Gamma_hat[0])+', std.errors='+str(se[0]))
print('P22='+str(Gamma_hat[1])+', std.errors='+str(se[1]))
print('h1='+str(Gamma_hat[2])+', std.errors='+str(se[2]))
print('h2='+str(Gamma_hat[3])+', std.errors='+str(se[3]))


#Plotting returns and filtered probabilities
gamma=Gamma_hat
#parameters
p00    = gamma[0]
p11    = gamma[1]
sigma2 = gamma[2:4]**2;
T      = len(y)
#//transition matrix
P = np.zeros([2,2])
P[0]=p00, 1-p11
P[1]=1-p00, p11
    
#//bookkeeping
xi_10      = np.zeros([2,T+1])
xi_11      = np.zeros([2,T])
xi_1T      = np.zeros([2,T])
lik        = np.zeros(T)

#//regression:
A  = np.vstack(((np.identity(2)-P),np.ones([1,2])))
pi_first = np.linalg.inv(A.T.dot(A)).dot(A.T)
pi_second=np.vstack((np.zeros([2,1]),np.ones([1,1])))
pi=pi_first.dot(pi_second)
xi_10[[0,1],0] = pi.T
#//forward filter recursion
eta=np.zeros(2)
for t in range(T):
    #//state densities
    eta[0]=GaussianDensity(y[t],0,sigma2[0])
    eta[1]=GaussianDensity(y[t],0,sigma2[1])
        
    #likelihood
    #print(np.log(xi_10[[0,1],t]))
    lik[t]   = np.log(xi_10[0,t]*eta[0]+xi_10[1,t]*eta[1])
        
    #filtering
    num0=eta[0]*xi_10[0,t]/(eta[0]*xi_10[0,t]+eta[1]*xi_10[1,t])
    num1=eta[1]*xi_10[1,t]/(eta[0]*xi_10[0,t]+eta[1]*xi_10[1,t])
    xi_11[[0,1],t] = num0,num1

    #prediction
    xi_10[[0,1],t+1] = P.dot(xi_11[[0,1],t])
    
    #Backward smoother (not needed for likelihood)
    xi_1T[:,T-1]=xi_11[:,T-1]
    for t in range(T-2,0,-1):
        xi_1T[:,t]=xi_11[:,t]*(P.T.dot(xi_1T[:,t+1]/xi_10[:,t+1]))
        
vol=np.zeros(len(y))
for i in range(T):
    vol[i]=xi_11[[0],i]*sigma2[0]+ (1-xi_11[[0],i])*sigma2[1]



fig, ax=plt.subplots(2, figsize=(14,7))
#fig.set_figheight=(9)
#fig.set_figwidth=(16)
fig.suptitle('log-return and filtered volatility')
ax[0].plot(y,color='r')
ax[1].plot(np.sqrt(vol))

#Setting titles
ax[0].title.set_text('Log-return, $x_t$')
ax[1].title.set_text('Filtered volatility, $E[\sigma_t|x_t,x_{t-1},...,x_1]$')


#Predicted state probability, Filtered state probability and smoothed state probability
fig, ax=plt.subplots(3, figsize=(16,9))
#fig.tight_layout() 

#Adjusting size between subplots
fig.subplots_adjust(left=None, bottom=0.025, right=None, top=None, wspace=None, hspace=None)
#default
#left  = 0.125  # the left side of the subplots of the figure
#right = 0.9    # the right side of the subplots of the figure
#bottom = 0.1   # the bottom of the subplots of the figure
#top = 0.9      # the top of the subplots of the figure
#wspace = 0.2   # the amount of width reserved for blank space between subplots
#hspace = 0.2   # the amount of height reserved for white space between subplots


ax[0].plot(1-xi_10[0,:])
ax[1].plot(1-xi_11[0,:])
ax[2].plot(1-xi_1T[0,:])

#Setting limits on x axis
ax[0].set_xlim(0, T)
ax[1].set_xlim(0, T)
ax[2].set_xlim(0, T)

#Setting titles
ax[0].title.set_text('Predicted state probability, $P(s_t=1|x_{t-1},x_{t-2},...,x_{1})$')
ax[1].title.set_text('Filtered state probability, $P(s_t=1|x_{t},x_{t-1},...,x_{1})$')
ax[2].title.set_text('Smoothed state probability, $P(s_t=1|x_{T},x_{T-1},...,x_{1})$')

#Setting lines at 0 and 1
ax[0].axhline(0,color='black', linestyle="--")
ax[0].axhline(1,color='black', linestyle="--")

ax[1].axhline(0,color='black', linestyle="--")
ax[1].axhline(1,color='black', linestyle="--")

ax[2].axhline(0,color='black', linestyle="--")
ax[2].axhline(1,color='black', linestyle="--")

"""