import numpy as np
import cvxpy as cp

# Payoff matrix for game. Player 1 - Row Player.  Note that this a zero-sum game, hence u_1(a) = -u_2(a) for all a.  Note that this a zero-sum game, hence u_1(a) = -u_2(a) for all a
mp_payoff = np.array([
    [[0, 0], [0.5, -0.5]],
    [[1, -1], [0, 0]]
])

# Extract Player 1's payoff matrix (A) for zero-sum assumption
A = mp_payoff[:, :, 0]

# Player 1's strategy (x) and value (v)
x = cp.Variable(A.shape[0])
v = cp.Variable()

# LP for Player 1: Maximize worst-case payoff
objective = cp.Maximize(v)
constraints = [
    A.T @ x >= v,  # Player 1's payoff >= v against any pure strategy of Player 2
    cp.sum(x) == 1,  # Probabilities sum to 1
    x >= 0           # Non-negativity
]

prob = cp.Problem(objective, constraints)
prob.solve(verbose=True, solver=cp.GUROBI)

# Player 2's strategy (y) is obtained from the dual variables
dual_vars = constraints[0].dual_value  
y = dual_vars / np.sum(dual_vars)

print("Zero-Sum Game Solution:")
print(f"Player 1's optimal strategy: {x.value.round(4)}")
print(f"Player 2's optimal strategy: {y.round(4)}")
print(f"Game value (for Player 1): {v.value.round(4)}")