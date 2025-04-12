# Variational Quantum Eigensolver.Py

# Description: Practical hybrid quantum-classical algorithm for current quantum hardqare (NISQ)
# Use Cases: Great for chemistry, optimization, and variational simulations
# Functionality: Uses paramaerized quantum circuit (ansatz) and classical optimizer to minimize the energy of a Hamiltonian 

# Example Uses 
# Simulating molecules (Ex: H2)
# Solving eigenvalue problems
# Variational problem solving 

# Import libraries
from qiskit import Aer 
from qiskit.algorithms import VQE 
from qiskit.circuit.library import TwoLocal 
from qiskit.opflow import Z, I, X, StateFn 
from qiskit.algorithms.optimizers import COBYLA

# Define simple Hamiltonian: H = Z * Z + H * I 
H = (Z ^ Z) + (X ^ I)

# Ansatz
ansatz = TwoLocal(rotation_blocks = 'ry', entanglement_blocks='cz', reps=2)

# Optimizer
optimizer = COBYLA(maxiter=100)
vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=Aer.get_backend('aer_simulator'))
result = vqe.compute_minimum_eigenvalue(operator=H)
 
print('Minimum eigenvalue:', result.eigenvalue.real)

