import numpy as np
from pyquil import Program
from pyquil.gates import RY, RZ, CNOT, MEASURE
from pyquil.api import QuantumComputer, get_qc
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.api import WavefunctionSimulator

from scipy.optimize import minimize

# === Portfolio Data ===
n_assets = 4
mu = np.array([0.1, 0.2, 0.15, 0.05])  # returns
cov = np.array([    
    [0.005, -0.010, 0.004, -0.002],
    [-0.010, 0.040, -0.002, 0.003],
    [0.004, -0.002, 0.023, -0.005],
    [-0.002, 0.003, -0.005, 0.015]
])  # risk
budget = 2  # number of assets allowed

def build_cost_hamiltonian(mu, cov, lambda_penalty, budget):
    hamiltonian = PauliSum()

    # Objective: -return + risk
    for i in range(n_assets):
        hamiltonian += -mu[i] * PauliTerm("Z", i, 0.5)
        for j in range(n_assets):
            hamiltonian += 0.25 * cov[i][j] * PauliTerm("Z", i) * PauliTerm("Z", j)

    # Constraint: (sum(x_i) - budget)^2
    for i in range(n_assets):
        for j in range(n_assets):
            hamiltonian += lambda_penalty * 0.25 * PauliTerm("Z", i) * PauliTerm("Z", j)

    for i in range(n_assets):
        hamiltonian += -lambda_penalty * budget * 0.5 * PauliTerm("Z", i)

    constant = lambda_penalty * (budget ** 2)
    return hamiltonian, constant

def efficient_su2_ansatz(params, n_qubits):
    prog = Program()
    idx = 0

    # Layer 1: RY + RZ rotations
    for q in range(n_qubits):
        prog += RY(params[idx], q)
        idx += 1
        prog += RZ(params[idx], q)
        idx += 1

    # Layer 2: Linear entanglement (CNOTs)
    for q in range(n_qubits - 1):
        prog += CNOT(q, q + 1)

    # Layer 3: Optional second rotation layer
    for q in range(n_qubits):
        prog += RY(params[idx], q)
        idx += 1
        prog += RZ(params[idx], q)
        idx += 1

    return prog

wfn_sim = WavefunctionSimulator()

def get_expectation(params, hamiltonian):
    circuit = efficient_su2_ansatz(params, n_assets)
    wf = wfn_sim.wavefunction(circuit)
    return hamiltonian.expectation(wf).real

def budget_violation(params):
    wf = wfn_sim.wavefunction(efficient_su2_ansatz(params, n_assets))
    probs = wf.get_outcome_probs()
    violation = 0
    for bitstring, prob in probs.items():
        if abs(bitstring.count('1') - budget) > 0:
            violation += prob
    return violation

# === VQE Loop with Adaptive Penalty ===
lambda_penalty = 1.0
eta = 0.5  # learning rate for penalty update

params = np.random.uniform(0, 2 * np.pi, size=2 * 2 * n_assets)  # 2 RY+RZ per layer, 2 layers

for iteration in range(10):
    # Build Hamiltonian
    H, constant = build_cost_hamiltonian(mu, cov, lambda_penalty, budget)

    # Optimize
    res = minimize(lambda p: get_expectation(p, H) + constant,
                   params, method='COBYLA')
    params = res.x

    # Update Penalty
    violation = budget_violation(params)
    if violation > 0.05:
        lambda_penalty *= (1 + eta * violation)

    print(f"Iter {iteration}: Energy = {res.fun:.4f}, Violation = {violation:.4f}, λ = {lambda_penalty:.2f}")


