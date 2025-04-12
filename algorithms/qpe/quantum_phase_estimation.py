# Quantum Phase Estimation.Py

# Description: One of the most important quantum subroutines 
# Use Cases: used for VQE, quantum chemistry, and solving eigenvalue problems
# Functionality: Estimates the phase in eigenvalue equation for unitary operator U (search it up)

# Example Uses 
# VQE & Quantum Chemistry 
# Finding eigenvalues
# Running quantum simulations

#import libraries 
from qiskit import QuantumCircuit, Aer, execute 
from qiskit.circuits.library import QFT 
import numpy as np 

n_count = 3 
qc = QuantumCircuit(n_count + 1, n_count)

# Apply Hademard to counting qubits
qc.h(range(n_count))

# Prepare eigenstances (target = |1))
qc.x(n_count)

# Controlled unitary: for demonstration. use controlled-Z rotation 
for q in range(n_count):
    qc.cp(2 * np.pi / 2**(q+1), q, n_count)

# Inverse QFT 
qc.append(QFT(n_count, inverse = True).to_gate(), range(n_count))

# Measure 
qc.measure(range(n_count), range(n_count))

qc.draw('mp1')
