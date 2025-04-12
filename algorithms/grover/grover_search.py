# Grover Search.Py

# Description: algorithm gives quantum speedup for unstructured search problems 
# Use Cases: Optimization and database-related challenges 
# Functionality: Finds 'marked' item in an unsorted list of N items in O(sqrt(N)) classically

# Examples of Use 
# Solving logic puzzles
# Satisfying inputs to Boolean formulas (SAT problems)
# Optimization search spaces 

# import libraries 
from qiskit import Aer, QuantumCircuit, execute 
from qiskit.circuit.library import GroverOperator 
from qiskit.algorithms import Grover
from qiskit.algorithms.amplitude_estimators import IterativeAmplitudeEstimation 
from qiskit.quantum_info import Statevector
from qiskit.circuit.library.phase_oracle import PhaseOracle 

# Define oracke in Boolean logic 
oracle = PhaseOracle("a & b") # mark state where a=1 and b=1 

grover = Grover(oracle=oracle)
result = grover.run() 

print("Grover result:" result.assignment)


