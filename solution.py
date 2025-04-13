
### GIVEN PROGRAM ### 
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of properties
n_properties = 10

# Generate random expected returns (premiums) for properties
mu = np.random.uniform(0.05, 0.15, n_properties)

# Generate random correlation matrix and ensure it's symmetric and positive semi-definite
random_matrix = np.random.rand(n_properties, n_properties)
correlation_matrix = (random_matrix + random_matrix.T) / 2
np.fill_diagonal(correlation_matrix, 1)

# Generate covariance matrix by scaling the correlation matrix
sigma = correlation_matrix * np.outer(mu, mu)

# Budget (30% of properties)
budget = int(0.3 * n_properties)

# plot sigma as image
import matplotlib.pyplot as plt

plt.imshow(sigma, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.title("Covariance Matrix")
plt.xlabel("Properties")
plt.ylabel("Properties")
plt.show()

plt.bar(range(n_properties), mu)
plt.title("Expected Returns (Premiums)")
plt.xlabel("Properties")
plt.ylabel("Expected Return")
plt.show()


### CLASSICAL OPTIMIZATION FOR INSURANCE PORTFOLIOS ### 

import matplotlib.pyplot as plt 

np.random.seed(42)

n_properties = 10 

mu = np.random.uniform(0.05, 0.15, n_properties)

random_matrix = np.random.rand(n_properties, n_properties)

random_matrix = np.random.rand(n_properties, n_properties)
correlation_matrix = (random_matrix + random_matrix.T) / 2
np.fill_diagonal(correlation_matrix, 1)

sigma = correlation_matrix * np.outer(mu, mu)

budget = int(0.3 * n_properties)
lambdas = [0.01, 0.1, 1, 10, 100]

scores = []

from itertools import combinations

scores = []
property_indices = list(range(n_properties))

for lam in lambdas:
    best_score = float('inf')

    for combo in combinations(property_indices, budget):
        x = np.zeros(n_properties)
        x[list(combo)] = 1

        # Calculate terms
        risk_term = x @ sigma @ x
        return_term = x @ mu
        penalty_term = lam * (np.sum(x) - budget) ** 2
        total_score = risk_term - return_term + penalty_term

        if total_score < best_score:
            best_score = total_score

    scores.append(best_score)


import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt 

fig, ax = plt.subplots()
ax.plot(lambdas, scores, marker='o')
ax.set_xscale("log")
ax.set_title("Effect of λ (Penalty Term) on Portfolio Objective")
ax.set_xlabel("λ (Penalty Coefficient)")
ax.set_ylabel("Minimum Objective Function Value")
ax.grid(True)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.show()


print(pd.DataFrame({'Lambda': lambdas, 'Objective Score': scores}))


### QISKIT BASED CLASSICAL MONTE CARLO FOR 

np.random.seed(42)

num_policies = 25
budget = 10 
risk_factor = 0.5 

premiums = np.random.uniform(1000, 5000, num_policies)

random_matrix = np.random.rand(num_policies, num_policies)
correlation_matrix = (random_matrix + random_matrix.T) / 2 
np.fill_diagonal(correlation_matrix, 1)

covariance_matrix = correlation_matrix * np.outer(premiums, premiums) * 0.00001

best_portfolio = None 
best_return = -np.inf 

returns_list = []
risks_list = [] 

for _ in range(10000): 
    weights = np.random.random(num_policies)
    weights /= np.sum(weights)
    scaled_weights = weights * budget

    portfolio_return = np.dot(scaled_weights, premiums)
    portfolio_risk = np.dot(scaled_weights.T, np.dot(covariance_matrix, scaled_weights))
    adjusted_return = portfolio_return - risk_factor * portfolio_risk 

    returns_list.append(portfolio_return)
    risks_list.append(portfolio_risk)

    if adjusted_return > best_return:
        best_return = adjusted_return 
        best_portfolio = scaled_weights

expected_return_best = np.dot(best_portfolio, premiums)
risk_best = np.dot(best_portfolio.T, np.dot(covariance_matrix, best_portfolio))

returns_list = [(r / np.sum(premiums)) * 100 for r in returns_list]
expected_return_best = (expected_return_best / np.sum(premiums)) * 100

print("\n=== Monte Carlo Optimization: Insurance Portfolio ===")
print("Best Portfolio Allocation (scaled to budget):", best_portfolio)
print(f"Expected Premium Return: ${expected_return_best:,.2f}")
print(f"Portfolio Risk: {risk_best:.4f}")

# Plot risk vs. return
plt.figure(figsize=(10, 6))
plt.scatter(risks_list, returns_list, alpha=0.5, color='purple', label='Portfolios')
plt.scatter(risk_best, expected_return_best,
            color='green', marker='*', s=200, label='Best Portfolio')
plt.title('Monte Carlo Simulation: Insurance Portfolio Risk vs. Return')
plt.xlabel('Portfolio Risk (Covariance)')
plt.ylabel('Expected Premium Return (%)')
plt.legend()
plt.grid()
plt.show()