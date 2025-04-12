# Qiskit Circuits.Md

### THIS IS A QUICK REFERENCE GUIDE FOR BUILDING AND MANIPULATING CIRCUITS IN QISKIT ***

# CIRCUIT INSTALLATION 
from qiskit import QuantumCircuit: 

qc = QuantumCircuits(2, 2) # 2 qubits, 2 classical bits 


# BASIC GATES 

# Hadamard: Creates superposition 
qc.h(0)

# Pauli-X: Bit-flip(NOT)
qc.x(0)

# Pauli-Y: Y gate
qc.y(0)

# Pauli-Z: Phase-flip
qc.z(0)

# Identity: No-op 
qc.i(0)

# Phase (S): Square root of Z
qc.s(0)

# T gate: 4th root Z 
qc.t(0)


# MEASUREMENT 
qc.measure(0,0) # Measure qubit 0 into classical bit 0 


# ENTANGLING GATES 

# CNOT - Controlled-X
qc.cx(0, 1)

# CZ - Controlled-Z
qc.cz(0, 1)

# SWAP - Swaps qubit states 
qc.swap(0, 1)


# ROTATION GATES 

# Rx(0) - rotation around x-axis 
qc.rx(theta, q)

# Ry(0) - rotation around y-axis 
qc.ry(theta, q)

# Rz(0) - rotation around z-axis
qc.rz(theta, q)

# U3(theta, phi, lam) - general 1-qubit rotation
qc.u3(theta, phi, lam, q)

# CIRCUIT SIMULATION 
qc.draw('mpl')  # matplotlib drawing
qc.draw('text') # ASCII drawing

# SIMULATE WITH AER 
from qiskit import Aer, execute
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
counts = job.result().get_counts()
print(counts)

# IBMQ INTEGRATION 
from qiskit import IBMQ
IBMQ.load_account()
provider = IBMQ.get_provider()
backend = provider.get_backend('ibmq_qasm_simulator')