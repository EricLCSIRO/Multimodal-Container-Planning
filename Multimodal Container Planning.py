# https://arxiv.org/abs/2007.01730

import neal
import numpy
from pyqubo import Array, Placeholder, solve_qubo, Constraint
from pyqubo import Model
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
# Initialize parameters
N = 10  # number of containers
M = 12  # number tracks
K = 3  # slack parameter
# Costs
c_b = [2, 7, 1, 6, 2, 4, 8, 7, 7, 10]
c_t = [23, 25, 23, 17, 24, 22, 19, 16, 21, 17]
# Track capacity
v = {}
v = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
routes = {}  # route of containers
routes[0] = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
routes[1] = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
routes[2] = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
routes[3] = [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]
routes[4] = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
routes[5] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
routes[6] = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
routes[7] = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
routes[8] = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
routes[9] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
# Initialize variable vector
size_of_variable_array = N + K*M
var = Array.create('vector', size_of_variable_array, 'BINARY')
# Defining constraints in the Model
minimize_costs = 0
minimize_costs += Constraint(sum(var[i]*(
    c_t[i]-c_b[i])+c_b[i] for i in range(N)), label="minimize_transport_costs")
capacity_constraint = 0
for j in range(M):
    capacity_constraint += Constraint(
        (sum((1-var[i])*routes[i][j] for i in range(N))
         + sum(var[N + j*K + i]*(2**(i)) - v[j] for i in range(K)))**2,
        label="capacity_constraints")

# parameter values
A = 1
B = 6
Cs = 240
useQPU = True
# Define Hamiltonian as a weighted sum of individual constraints
H = A * minimize_costs + B * capacity_constraint
# Compile the model and generate QUBO
model = H.compile()
qubo, offset = model.to_qubo()
# Choose sampler and solve qubo
if useQPU:
    sampler = EmbeddingComposite(DWaveSampler())
    # solver=DWaveSampler()) #, num_reads=50)
    response = sampler.sample_qubo(qubo, chain_strength=Cs, num_reads=1000)
else:
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo, num_sweeps=10000, num_reads=50)
# Postprocess solution
sample = response.first.sample
print(sample)
