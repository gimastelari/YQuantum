
### CLASSICALAL COMPUTATION OF OPTIMIAL INSURANCE PORTFOLIOS USING: 
# Penalty Method (Constrained Optimiztion Technique)
# Qiskit Based Classical Monte Carlo 
# Global Minimum-Variance Portfolio 



### GIVEN PROGRAM FOR GENERATING 
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

# vizualization 
plt.imshow(sigma, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.title("Covariance Matrix")
plt.xlabel("Properties")
plt.ylabel("Properties")
plt.show()

# Vizualization
plt.bar(range(n_properties), mu)
plt.title("Expected Returns (Premiums)")
plt.xlabel("Properties")
plt.ylabel("Expected Return")
plt.show()







### CLASSICAL OPTIMIZATION FOR INSURANCE PORTFOLIOS USING PENALTY METHOD### 

import matplotlib.pyplot as plt 

# reinitialized - however this was already declared before 
np.random.seed(42)
n_properties = 10 
mu = np.random.uniform(0.05, 0.15, n_properties)

# Generating correlation matrix - already implemented previously as well 
random_matrix = np.random.rand(n_properties, n_properties)
correlation_matrix = (random_matrix + random_matrix.T) / 2
np.fill_diagonal(correlation_matrix, 1)


# Calculating covariance matrix - already done previously 
sigma = correlation_matrix * np.outer(mu, mu)

# budget is to select 30% of properties 
budget = int(0.3 * n_properties)

# Try different lambda valus (penalty strength for deviating from budget constraint)
lambdas = [0.01, 0.1, 1, 10, 100]
scores = []

from itertools import combinations

# iterate over all possible combinations of properties and final the optimal one
property_indices = list(range(n_properties))
for lam in lambdas:
    best_score = float('inf')

    # try every combination of 'budget' number of properties 
    for combo in combinations(property_indices, budget):
        x = np.zeros(n_properties)
        x[list(combo)] = 1 # select the current bombination 

        # Objective: Risk-Return + Penalty 
        risk_term = x @ sigma @ x
        return_term = x @ mu
        penalty_term = lam * (np.sum(x) - budget) ** 2
        total_score = risk_term - return_term + penalty_term

        if total_score < best_score:
            best_score = total_score

    scores.append(best_score)

# plotting how the lambda parameter affects the objective function value
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










### QISKIT BASED CLASSICAL MONTE CARLO FOR PORTFOLIO OPTIMIZATION 

# implement seed - already done several times 
np.random.seed(42)

# number of policies and budget 
num_policies = 25
budget = 10 
risk_factor = 0.5 # tradeoff parameter between return and risk

# Generate synthetic premiums and covariance matrix
premiums = np.random.uniform(1000, 5000, num_policies)
random_matrix = np.random.rand(num_policies, num_policies)
correlation_matrix = (random_matrix + random_matrix.T) / 2 
np.fill_diagonal(correlation_matrix, 1)

# Covariance scaled appropriately (units)
covariance_matrix = correlation_matrix * np.outer(premiums, premiums) * 0.00001

# Initialize trackers for the best portfolio and performance metrics 
best_portfolio = None 
best_return = -np.inf 
returns_list = []
risks_list = [] 

# Perform random search using Monte Carlo 
for _ in range(10000): 
    weights = np.random.random(num_policies)
    weights /= np.sum(weights) # normalize to sum to 1 
    scaled_weights = weights * budget # adjust to meet the budget

    # Calculate return and risk 
    portfolio_return = np.dot(scaled_weights, premiums)
    portfolio_risk = np.dot(scaled_weights.T, np.dot(covariance_matrix, scaled_weights))
    adjusted_return = portfolio_return - risk_factor * portfolio_risk 

    # Track Results 
    returns_list.append(portfolio_return)
    risks_list.append(portfolio_risk)

    # Keep best performing portfolio 
    if adjusted_return > best_return:
        best_return = adjusted_return 
        best_portfolio = scaled_weights

# Final Performance metrics for the best portfolio 
expected_return_best = np.dot(best_portfolio, premiums)
risk_best = np.dot(best_portfolio.T, np.dot(covariance_matrix, best_portfolio))

# Normalize returns for plotting 
returns_list = [(r / np.sum(premiums)) * 100 for r in returns_list]
expected_return_best = (expected_return_best / np.sum(premiums)) * 100

# Output 
print("\n=== Monte Carlo Optimization: Insurance Portfolio ===")
print("Best Portfolio Allocation (scaled to budget):", best_portfolio)
print(f"Expected Premium Return: ${expected_return_best:,.2f}")
print(f"Portfolio Risk: {risk_best:.4f}")

# Plot risk vs. return for all portfolios and the best one 
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









### GLOBAL MINIMUM-VARIANCE PORTFOLIO USING OPTIMIZATION 
from scipy.optimize import minimize 
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib as mpl

# Function to compute portfolio variance (objective function)
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights 

# Constraint: weights must sum to 1 
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights)-1}

# Bounds: weights between 0 and 1 
bounds = tuple((0,1) for _ in range(num_policies))

# Inital Guess - equal weight distribution 
initial_guess = num_policies * [1.0 / num_policies]

# Minimize portfolio variance subject to constraints and bounds 
result = minimize(
    portfolio_variance, 
    initial_guess,
    args=(covariance_matrix, ),
    method = "SLSQP",
    bounds = bounds, 
    constraints=constraints
)

# Extract optimal weights and calculate associated return and risk 
gmvp_weights = result.x 
gmvp_return = np.dot(gmvp_weights * budget, premiums)
gmvp_return = (gmvp_return / np.sum(premiums)) * 100 
gmvp_variance = portfolio_variance(gmvp_weights * budget, covariance_matrix)
gmvp_std_dev = np.sqrt(gmvp_variance)

# Output 
print("\n=== Global Minimum-Variance Portfolio (GMVP) ===")
print("GMVP Weights (scaled to budget):", gmvp_weights * budget)
print(f"Expected Premium Return of GMVP: {gmvp_return:.2f}")
print(f"Risk (Variance) of GMVP: {gmvp_variance:.4f}")

# Plot set-up and styling 
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
plt.style.use('seaborn-v0_8-darkgrid')
sns.set()
plt.figure(figsize=(10, 6))

# Plot all Monte Carlo portfolios
plt.scatter(risks_list, returns_list, alpha=0.5, color='blue', label='Random Portfolios')

# Plot Best Monte Carlo portfolio
plt.scatter(
    risk_best,
    expected_return_best,
    color='red',
    marker='*',
    s=200,
    label='Best Monte Carlo Portfolio'
)

# Plot GMVP
plt.scatter(
    gmvp_variance,
    gmvp_return,
    color='green',
    marker='D',
    s=100,
    label='Global Minimum-Variance Portfolio (GMVP)'
)

plt.title('Insurance Portfolio Risk vs. Premium Return')
plt.xlabel('Portfolio Risk (Variance)')
plt.ylabel('Expected Premium Return')
plt.legend()
plt.grid(True)
plt.show()