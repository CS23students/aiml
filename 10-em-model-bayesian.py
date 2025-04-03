import numpy as np
import math
# Define the Bayesian network structure
nodes = ['A', 'B', 'C', 'D']
parents = {
'B': ['A'],
'C': ['A'],
'D': ['B', 'C']
}

probabilities = {
'A': np.array([0.6, 0.4]),
'B': np.array([[0.3, 0.7], [0.9, 0.1]]),
'C': np.array([[0.2, 0.8], [0.7, 0.3]]),
'D': np.array([[[0.1, 0.9], [0.6, 0.4]], [[0.7, 0.3], [0.8, 0.2]]])
}
Define the observed data
data = {'A': 1, 'D': 0}
# Define the EM algorithm
def em_algorithm(nodes, parents, probabilities, data, max_iterations=100):
# Initialize the parameters
for node in nodes:
if node not in data:
probabilities[node] = np.ones(probabilities[node].shape) /
probabilities[node].shape[0]
# Run the EM algorithm
for iteration in range(max_iterations):
# E-step: Compute the expected sufficient statistics
expected_counts = {}
for node in nodes:
if node not in data:
expected_counts[node] = np.zeros(probabilities[node].shape)
# Compute the posterior probability distribution over the latent variables given the
# observed data
joint_prob = np.ones(probabilities['A'].shape) for
node in nodes:
if node in data:
joint_prob *= probabilities[node][data[node]]
else:
parent_probs = [probabilities[parent][data[parent]] for parent in

parents[node]]
joint_prob *= probabilities[node][tuple(parent_probs)]
posterior = joint_prob / np.sum(joint_prob)
# Compute the expected sufficient statistics for
node in nodes:
if node not in data:
if node in parents:
parent_probs = [probabilities[parent][data[parent]] for parent in

parents[node]]

expected_counts[node] = np.sum(posterior *
probabilities[node][tuple(parent_probs)], axis=0)

else:
expected_counts[node] = np.sum(posterior * probabilities[node], axis=0)
M-step: Update the parameter estimates for
node in nodes:
if node not in data:
probabilities[node] = expected_counts[node] / np.sum(expected_counts[node])
Check for convergence
if iteration > 0 and np.allclose(prev_probabilities, probabilities):
break
prev_probabilities = np.copy(probabilities)
return probabilities
Run the EM algorithm
probabilities = em_algorithm(nodes, parents, probabilities, data)
# Print the final parameter estimates for
node in nodes:
print(node, probabilities[node])
