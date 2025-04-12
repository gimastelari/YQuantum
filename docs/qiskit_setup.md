# Qiskit Setup.Md

# Qiskit & IBMQ Setup Guide

## Step 1: Install Qiskit from terminal 
pip install qiskit

## Step 2: Set ip IBMQ Account 

# Get API token at https://quantum-computing.ibm.com/account 
from qiskit import IBMQ
IBMQ.save_account('YOUR_TOKEN_HERE')
IBMQ.load_account()

## Step 3: Run Local Simulations 
from qiskit import Aer, execute
backend = Aer.get_backend('qasm_simulator')

# Step 4: Use real IBMQ Hardware 
provider = IBMQ.get_provider()
backend = provider.get_backend('ibmq_quito')

