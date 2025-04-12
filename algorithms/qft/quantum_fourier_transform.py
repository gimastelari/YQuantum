# Quantum Fourier Transform.Py

# Description: Fundamental subroutine used in Shor's algorithm, quantum phase estimation and many chemistry simulation 
# Use Cases: Shor's Algorithm, Quantum Phase Estimation. Simulating periodically 
# Functionality: transforms computational basis states into Fourier basis like FFT but for qubits

# import libraries 
from qiskit import QuantumCircuit 
import numpy as np 

def qft(n): 
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i+1, n): 
            qc.cp(np.pi/2**(j-1), j, i)
    for i in range(n//2): 
        qc.swap(i, n-i-1)
    qc.name = "QFT"
    return qc

qc = qft(3)
qc.draw('mp1')



