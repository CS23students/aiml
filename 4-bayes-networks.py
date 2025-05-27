# Monty Hall Problem Simulation in Python
# Using logic similar to Bayesian Network nodes: guest, prize, monty
import random

# Define the doors
doors = ['A', 'B', 'C']

# Function to simulate Monty's choice based on guest and prize
def monty_choice(guest, prize):
    # Monty must open a door that:
    # - is not the guest's choice
    # - does not have the prize
    possible_doors = [door for door in doors if door != guest and door != prize]
    return random.choice(possible_doors)

# Simulate probabilities for a fixed guest choice
guest = 'A'
results = {'prize=A': 0, 'prize=B': 0, 'prize=C': 0}

# Simulate 10000 rounds
for _ in range(10000):
    prize = random.choice(doors)
    monty = monty_choice(guest, prize)

    # Count how often the prize is behind each door
    results[f'prize={prize}'] += 1

# Show probabilities
print(f"Given guest chose door: {guest}")
print("Estimated probabilities of prize behind each door:")
for key in sorted(results):
    percentage = (results[key] / 10000) * 100
    print(f"{key} => {percentage:.2f}%")

# Given guest chose door: A
# Estimated probabilities of prize behind each door:
# prize=A => 33.10%
# prize=B => 34.01%
# prize=C => 32.89%



----------------------------------------------------------------------






import math
from pomegranate import *
# Initially the door selected by the guest is completely random
guest = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})
# The door containing the prize is also a random process
prize = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})
# The door Monty picks, depends on the choice of the guest and the prize door
monty = ConditionalProbabilityTable(
[['A', 'A', 'A', 0.0],
['A', 'A', 'B', 0.5],
['A', 'A', 'C', 0.5],
['A', 'B', 'A', 0.0],

['A', 'B', 'B', 0.0],
['A', 'B', 'C', 1.0],
['A', 'C', 'A', 0.0],
['A', 'C', 'B', 1.0],
['A', 'C', 'C', 0.0],
['B', 'A', 'A', 0.0],
['B', 'A', 'B', 0.0],
['B', 'A', 'C', 1.0],
['B', 'B', 'A', 0.5],
['B', 'B', 'B', 0.0],
['B', 'B', 'C', 0.5],
['B', 'C', 'A', 1.0],
['B', 'C', 'B', 0.0],
['B', 'C', 'C', 0.0],
['C', 'A', 'A', 0.0],
['C', 'A', 'B', 1.0],
['C', 'A', 'C', 0.0],
['C', 'B', 'A', 1.0],
['C', 'B', 'B', 0.0],
['C', 'B', 'C', 0.0],
['C', 'C', 'A', 0.5],
['C', 'C', 'B', 0.5],
['C', 'C', 'C', 0.0]], [guest, prize])
d1 = State(guest, name="guest")
d2 = State(prize, name="prize")
d3 = State(monty, name="monty")
# Building the Bayesian Network
network = BayesianNetwork("Solving the Monty Hall Problem With Bayesian Networks")
network.add_states(d1, d2, d3)
network.add_edge(d1, d3)
network.add_edge(d2, d3)
network.bake()
# Compute the probabilities for each scenario
beliefs = network.predict_proba({'guest': 'A'})
print("\n".join("{}\t{}".format(state.name, str(belief)) for state, belief in zip(network.states,
beliefs)))
beliefs = network.predict_proba({'guest': 'A', 'monty': 'B'})
print("\n".join("{}\t{}".format(state.name, str(belief)) for state, belief in zip(network.states,
beliefs)))
beliefs = network.predict_proba({'guest': 'A', 'prize': 'B'})
print("\n".join("{}\t{}".format(state.name, str(belief)) for state, belief in zip(network.states,
beliefs)))






