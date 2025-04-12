"""
Portfolio Optimization using Qiskit Finance

This script demonstrates how to solve a portfolio optimization problem using quantum computing
approaches including VQE (Variational Quantum Eigensolver) and QAOA (Quantum Approximate 
Optimization Algorithm).

The optimization problem aims to find the optimal asset allocation that maximizes returns
while minimizing risk, subject to budget constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import (
    NumPyMinimumEigensolver,
    QAOA,
    VQE,
    SamplingVQE
)
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer

def create_portfolio_problem(num_assets: int, seed: int = 123):
    """
    Creates a portfolio optimization problem instance using random data.
    
    Args:
        num_assets (int): Number of assets to consider
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: Returns (portfolio, qp) where portfolio is the PortfolioOptimization instance
              and qp is the converted quadratic program
    """
    # Generate stock tickers
    stocks = [f"TICKER{i}" for i in range(num_assets)]
    
    # Create random data provider
    data = RandomDataProvider(
        tickers=stocks,
        start=datetime.datetime(2016, 1, 1),
        end=datetime.datetime(2016, 1, 30),
        seed=seed
    )
    data.run()
    
    # Get expected returns and covariance matrix
    mu = data.get_period_return_mean_vector()
    sigma = data.get_period_return_covariance_matrix()
    
    # Set optimization parameters
    q = 0.5  # risk factor
    budget = num_assets // 2  # budget constraint
    
    # Create portfolio optimization instance
    portfolio = PortfolioOptimization(
        expected_returns=mu,
        covariances=sigma,
        risk_factor=q,
        budget=budget
    )
    
    # Convert to quadratic program
    qp = portfolio.to_quadratic_program()
    
    return portfolio, qp

def print_result(result, portfolio):
    """
    Prints the optimization results in a formatted way.
    
    Args:
        result: The optimization result
        portfolio: The PortfolioOptimization instance
    """
    selection = result.x
    value = result.fval
    print(f"Optimal selection: {selection}")
    print(f"Optimal value: {value:.4f}")
    
    if hasattr(result, 'min_eigen_solver_result'):
        eigenstate = result.min_eigen_solver_result.eigenstate
        if hasattr(eigenstate, 'binary_probabilities'):
            probabilities = eigenstate.binary_probabilities()
        else:
            probabilities = {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
        
        print("\n----------------- Full result ---------------------")
        print("selection\tvalue\t\tprobability")
        print("---------------------------------------------------")
        
        for k, v in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            x = np.array([int(i) for i in list(reversed(k))])
            value = portfolio.to_quadratic_program().objective.evaluate(x)
            print(f"{x}\t{value:.4f}\t\t{v:.4f}")

def main():
    """
    Main execution function that runs the portfolio optimization using different methods.
    """
    # Set random seed for reproducibility
    algorithm_globals.random_seed = 1234
    
    # Create problem instance
    num_assets = 4
    portfolio, qp = create_portfolio_problem(num_assets)
    
    print("Solving using classical NumPyMinimumEigensolver...")
    exact_mes = NumPyMinimumEigensolver()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)
    result = exact_eigensolver.solve(qp)
    print_result(result, portfolio)
    
    print("\nSolving using VQE...")
    cobyla = COBYLA()
    cobyla.set_options(maxiter=500)
    ansatz = RealAmplitudes(num_assets, reps=3)
    sampler = Sampler()
    vqe = VQE(sampler=sampler, ansatz=ansatz, optimizer=cobyla)
    vqe_optimizer = MinimumEigenOptimizer(vqe)
    result = vqe_optimizer.solve(qp)
    print_result(result, portfolio)
    
    print("\nSolving using QAOA...")
    qaoa = QAOA(sampler=sampler, optimizer=cobyla, reps=3)
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    result = qaoa_optimizer.solve(qp)
    print_result(result, portfolio)

if __name__ == "__main__":
    main()
