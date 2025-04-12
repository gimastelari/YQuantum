# Circuit Examples.Md

# Quantum Circuit Examples in Qiskit

This file contains working code snippets for some of the most common quantum circuit patterns you might use during this hackathon.

## 1. Bell State (Entanglement)
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
qc.draw('mpl')

## 2. Quantum Teleportation Circuit 
from qiskit import QuantumCircuit
qc = QuantumCircuit(3, 3)
qc.h(1)
qc.cx(1, 2)
qc.cx(0, 1)
qc.h(0)
qc.measure([0, 1], [0, 1])
qc.cx(1, 2)
qc.cz(0, 2)
qc.measure(2, 2)

## 3. Glover Algorithm 
from qiskit.circuit.library import GroverOperator, PhaseOracle
from qiskit.algorithms import Grover
oracle = PhaseOracle('a & b')
grover = Grover(oracle=oracle)
result = grover.run()
print(result.assignment)

## 4. Basic Superposition 
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

## 5. QFT 
from qiskit.circuit.library import QFT
qc = QFT(3)
qc.draw('mpl')






