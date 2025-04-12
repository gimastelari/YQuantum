# Shors Algorithm.Py

# Desciption: Exponential speedup for integer factorization 
# Use Cases: Demonstrating exponential quantum speedup, breaking RSA cryptography (theoretical)
# Functionality: For real backends, Shor is only practical for toy numbers (like 15) due to hardware limits 

# Import libraries 
from qiskit.algorithms import Shor 

# Use Qiskit's high-level Shor interface 
shor = Shor() 
N = 15 
result = shor.factor(N)

print("Factors of" N, ":", result.factors)

