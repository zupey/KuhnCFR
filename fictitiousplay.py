import numpy as np

def fictitious_play(A, iterations=1000):
    m, n = A.shape
    counts_p1 = np.zeros(m)
    counts_p2 = np.zeros(n)
    
    for _ in range(1, iterations + 1):
        # Current empirical strategies
        x = counts_p1 / np.sum(counts_p1) if np.sum(counts_p1) > 0 else np.ones(m) / m
        y = counts_p2 / np.sum(counts_p2) if np.sum(counts_p2) > 0 else np.ones(n) / n
        
        # Best responses (pure strategies)
        br_p1 = np.argmax(A @ y)
        br_p2 = np.argmin(x @ A)
        
        # Update counts
        counts_p1[br_p1] += 1
        counts_p2[br_p2] += 1
    
    # Final empirical strategies
    x_final = counts_p1 / iterations
    y_final = counts_p2 / iterations
    
    return x_final, y_final

A = np.array([[0, 0.5], [1, 0]])
x, y = fictitious_play(A, iterations=10000)
print("Player 1's strategy:", x.round(4))
print("Player 2's strategy:", y.round(4))