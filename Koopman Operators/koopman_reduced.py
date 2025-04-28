import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv

def system_dynamics(x, u):
    return -x + u  # Basic linear dynamics

def generate_data(x0, u, steps):
    X, X_prime = [], []
    x = x0
    for _ in range(steps):
        X.append(x)
        x_next = system_dynamics(x, u)
        X_prime.append(x_next)
        x = x_next
    return np.array(X), np.array(X_prime)

def compute_koopman_operator(X, X_prime):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X_prime.ndim == 1:
        X_prime = X_prime.reshape(-1, 1)
    return pinv(X) @ X_prime


def predict_future_states(K, x_init, steps):
    predictions = [x_init]
    for _ in range(steps):
        predictions.append(K @ predictions[-1])
    return np.array(predictions)

def main():
    x0 = np.array([1.0])
    u = np.array([0.5])
    
    X, X_prime = generate_data(x0, u, 100)
    K = compute_koopman_operator(X[:-1], X_prime[:-1])
    future_predictions = predict_future_states(K, X[-1], 10)
    
    plt.plot(range(len(X)), X, label="Actual")
    plt.plot(range(len(X), len(X) + len(future_predictions)), future_predictions, 'r--', label="Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("State")
    plt.legend()
    plt.title("Koopman Predictions")
    plt.show()
    
    print("Predicted next states:", future_predictions)

if __name__ == "__main__":
    main()
