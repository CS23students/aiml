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

-------------------------------------------------------------------------------------------------------------------------


import numpy as np

# Define the Bayesian network structure
nodes = ['A', 'B', 'C', 'D']
parents = {
    'B': ['A'],
    'C': ['A'],
    'D': ['B', 'C']
}

# Initial probabilities
probabilities = {
    'A': np.array([0.6, 0.4]),  # P(A=0) = 0.6, P(A=1) = 0.4
    'B': np.array([[0.3, 0.7],   # P(B=0|A=0), P(B=1|A=0)
                   [0.9, 0.1]]), # P(B=0|A=1), P(B=1|A=1)
    'C': np.array([[0.2, 0.8],
                   [0.7, 0.3]]),
    'D': np.array([
        [[0.1, 0.9], [0.6, 0.4]],  # P(D=0|B=0,C=0), P(D=1|B=0,C=0) etc.
        [[0.7, 0.3], [0.8, 0.2]]
    ])
}

# Observed data: A = 1, D = 0
data = {'A': 1, 'D': 0}

def em_algorithm(nodes, parents, probabilities, data, max_iterations=50):
    prev_probabilities = {}

    for iteration in range(max_iterations):
        expected_counts = {}

        # Initialize expected counts
        for node in nodes:
            if node not in data:
                expected_counts[node] = np.zeros(probabilities[node].shape)

        # Loop over all possible values of latent variables (brute force for small networks)
        total_posterior = 0
        weighted_counts = {node: np.zeros_like(probabilities[node]) for node in nodes if node not in data}

        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        assignment = {'A': a, 'B': b, 'C': c, 'D': d}

                        # Check if assignment matches observed data
                        match = True
                        for var in data:
                            if assignment[var] != data[var]:
                                match = False
                                break
                        if not match:
                            continue

                        # Compute joint probability of the assignment
                        p_a = probabilities['A'][a]
                        p_b = probabilities['B'][a][b]
                        p_c = probabilities['C'][a][c]
                        p_d = probabilities['D'][b][c][d]
                        joint_prob = p_a * p_b * p_c * p_d

                        total_posterior += joint_prob

                        # Update expected counts
                        for node in nodes:
                            if node not in data:
                                if node == 'B':
                                    expected_counts['B'][a][b] += joint_prob
                                elif node == 'C':
                                    expected_counts['C'][a][c] += joint_prob
                                elif node == 'D':
                                    expected_counts['D'][b][c][d] += joint_prob

        # Normalize counts and update probabilities
        for node in expected_counts:
            expected = expected_counts[node]
            if expected.ndim == 1:
                probabilities[node] = expected / np.sum(expected)
            elif expected.ndim == 2:
                row_sums = expected.sum(axis=1, keepdims=True)
                probabilities[node] = expected / row_sums
            elif expected.ndim == 3:
                for i in range(expected.shape[0]):
                    for j in range(expected.shape[1]):
                        total = np.sum(expected[i][j])
                        if total != 0:
                            probabilities[node][i][j] = expected[i][j] / total

        # Check for convergence
        if iteration > 0:
            converged = True
            for node in prev_probabilities:
                if not np.allclose(prev_probabilities[node], probabilities[node], atol=1e-4):
                    converged = False
                    break
            if converged:
                break

        prev_probabilities = {k: np.copy(v) for k, v in probabilities.items()}

    return probabilities

# Run EM
final_probs = em_algorithm(nodes, parents, probabilities, data)

# Print final estimated probabilities
for node in final_probs:
    print(f"\n{node} final probability table:\n{final_probs[node]}")


# A final probability table:
# [0.6 0.4]

# B final probability table:
# [[0. 1.]
#  [0. 1.]]

# C final probability table:
# [[0. 1.]
#  [0. 1.]]

# D final probability table:
# [[[1. 0.]
#   [1. 0.]]
#  [[1. 0.]
#   [1. 0.]]]

