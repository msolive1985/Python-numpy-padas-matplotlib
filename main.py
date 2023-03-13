import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define prior probability distribution
def prior_prob(theta):
    if theta < 0 or theta > 1:
        return 0
    else:
        return 1

# Define likelihood function
def likelihood(x, theta):
    if x == 1:
        return theta
    else:
        return 1 - theta

# Define posterior probability distribution
def posterior_prob(theta, x):
    return prior_prob(theta) * likelihood(x, theta)

# Define function to compute posterior distribution
def compute_posterior(theta_values, x):
    posterior = []
    for theta in theta_values:
        posterior.append(posterior_prob(theta, x))
    posterior = np.array(posterior)
    posterior /= posterior.sum()
    return posterior

# Generate data
data = np.random.binomial(1, 0.3, size=100)

# Define range of theta values to consider
theta_values = np.linspace(0, 1, num=1000)

# Compute posterior distributions for the data
posterior_0 = compute_posterior(theta_values, data[0])
posterior_1 = compute_posterior(theta_values, data[1])

# Plot posterior distributions
fig, ax = plt.subplots(2, 1, figsize=(8, 6))

ax[0].plot(theta_values, posterior_0, label='Posterior after 1 observation')
ax[0].set_xlabel('Theta')
ax[0].set_ylabel('Density')
ax[0].legend()

ax[1].plot(theta_values, posterior_1, label='Posterior after 2 observations')
ax[1].set_xlabel('Theta')
ax[1].set_ylabel('Density')
ax[1].legend()

plt.show()
