# YQuantum

A quantum computing research and development project focused on implementing and analyzing quantum algorithms using Qiskit.

## Project Overview

YQuantum is a comprehensive quantum computing project that includes:
- Classical implementations of quantum algorithms
- Quantum algorithm implementations using Qiskit
- Experimental notebooks and documentation
- Research tools and utilities
- Team collaboration resources

## Project Structure

- `algorithms/`: Contains implementations of quantum algorithms
- `Classical Implementation/`: Classical computing approaches to quantum problems
- `docs/`: Project documentation and research materials
- `experiments/`: Experimental code and results
  - `rigetti.py`: Portfolio optimization implementation using Rigetti's PyQuil framework
  - `testing_vqe.py`: Portfolio optimization using Qiskit with VQE and QAOA implementations
  - `testing_vqe.ipynb`: Interactive Jupyter notebooks for VQE testing and visualization
  - `final.ipynb`: Comprehensive experimental notebook with analysis and results
  - `experiment_log.md`: Documentation of experimental procedures and findings
- `notebooks/`: Jupyter notebooks for analysis and visualization
- `team/`: Team-related resources and documentation
- `tools/`: Utility scripts and helper functions

## Experiments Overview

The experiments directory contains implementations and analysis of quantum portfolio optimization algorithms using different quantum computing frameworks:

1. **Rigetti Implementation** (`rigetti.py`):
   - Implements portfolio optimization with 4 assets
   - Uses VQE with an efficient SU2 ansatz
   - Features adaptive penalty method for constraint handling
   - Demonstrates quantum circuit construction and optimization

2. **Qiskit Implementation** (`testing_vqe.py`):
   - Implements portfolio optimization using multiple quantum algorithms
   - Compares VQE, QAOA, and classical NumPy solutions
   - Uses Qiskit Finance for problem formulation
   - Includes comprehensive result analysis and visualization

3. **Interactive Analysis**:
   - Jupyter notebooks for interactive experimentation
   - Visualization of quantum circuit performance
   - Step-by-step analysis of optimization results
   - Documentation of experimental procedures and findings

## Using the Experiments

To explore the portfolio optimization experiments:

1. **Interactive Analysis**:
   - Start with `final.ipynb` for an interactive walkthrough
   - Use `final.ipynb` for comprehensive analysis and results
   - The notebooks include visualizations and step-by-step explanations

2. **Running the Implementations**:
   - For Qiskit implementation: `python experiments/final.py`
   - For Rigetti implementation: `python experiments/rigetti.py`

## VQE Portfolio Optimization Explained

The experiments implement portfolio optimization using the Variational Quantum Eigensolver (VQE) algorithm. Here's how it works:

1. **Problem Formulation**:
   - The portfolio optimization problem is converted into a quadratic unconstrained binary optimization (QUBO) problem
   - The objective is to maximize returns while minimizing risk, subject to budget constraints
   - The problem is mapped to a quantum Hamiltonian using Pauli operators

2. **VQE Implementation**:
   - Uses a parameterized quantum circuit (ansatz) to prepare quantum states
   - The circuit consists of rotation gates (RY, RZ) and entangling gates (CNOT)
   - Parameters are optimized classically to minimize the expectation value of the Hamiltonian

3. **Key Components**:
   - **Cost Hamiltonian**: Combines return maximization and risk minimization terms
   - **Ansatz Circuit**: Efficient SU2 ansatz with alternating rotation and entanglement layers
   - **Constraint Handling**: Uses penalty terms to enforce budget constraints
   - **Classical Optimizer**: COBYLA optimizer to find optimal circuit parameters

4. **Results Analysis**:
   - Compares quantum solutions with classical approaches
   - Analyzes convergence behavior and solution quality
   - Evaluates the impact of different ansatz structures and optimization parameters

For detailed implementation and analysis, refer to the Jupyter notebooks in the experiments directory.

## Dependencies

The project uses the following main dependencies:
- Qiskit (0.44.1)
- Qiskit Aer (0.12.2)
- Qiskit Finance (0.3.4)
- Qiskit Algorithms (0.2.1)
- Qiskit Optimization (0.5.0)
- Matplotlib (≥3.7.0)
- NumPy (≥1.22.0)
- Qiskit IBM Runtime
- SciencePlots

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Contributing

Please refer to the documentation in the `docs/` directory for contribution guidelines and project standards.

## License

This project is licensed under the terms specified in the LICENSE file.
