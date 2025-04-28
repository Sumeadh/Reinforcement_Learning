import numpy as np
from scipy.linalg import eig, pinv
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

# Define known and unknown dynamics
def known_dynamics(x, u):
    return np.zeros_like(x)  # Placeholder for known dynamics

def unknown_dynamics(x, u):
    return np.zeros_like(x)  # Placeholder for unknown dynamics

# Full system dynamics
def system_dynamics(t, x, u_func):
    u = u_func(t)
    return known_dynamics(x, u) + unknown_dynamics(x, u)

# **🚀 Optimized Observable Dictionary** (Reducing size)
def observable_dictionary(x, u):
    return np.concatenate([x[:10], u])  # Use only the first 10 state variables

# Generate trajectory data
def generate_trajectory_data(x0, u, t_span, dt):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    u_func = lambda t: u  # Constant input
    sol = solve_ivp(system_dynamics, t_span, x0, args=(u_func,), t_eval=t_eval)
    return sol.y.T, t_eval

# Generate phase space data (Reduced num_samples)
def generate_phase_space_data(x0, u, num_samples=10000):
    cov_matrix = 0.1 * np.eye(len(x0))
    X2 = np.random.multivariate_normal(x0, cov_matrix, num_samples)
    U2 = np.tile(u, (num_samples, 1))
    return X2, U2

# Compute Jacobian using finite differences
def compute_jacobian(f, x, u, epsilon=1e-5):
    J = np.zeros((len(f(x, u)), len(x)))
    for i in range(len(x)):
        x1, x2 = np.copy(x), np.copy(x)
        x1[i] += epsilon
        x2[i] -= epsilon
        J[:, i] = (f(x1, u) - f(x2, u)) / (2 * epsilon)
    return J

# 🚀 **Optimized PI-EDMDc Algorithm (Using Sparse Matrices)**
def pi_edmdc(D1, D2, dt):
    X2, U2 = D2
    num_features = len(observable_dictionary(X2[0], U2[0]))

    # Use Sparse Matrices ✅
    Theta_X2_U2 = csc_matrix([observable_dictionary(x, u) for x, u in zip(X2, U2)])

    # **Truncated SVD** for Low-Rank Approximation ✅
    k = min(10, Theta_X2_U2.shape[1] - 1)
    U, S, Vt = svds(Theta_X2_U2, k=k)
    Theta_X2_U2_reduced = U @ np.diag(S) @ Vt

    # **Sparse Koopman Operator Lf**
    J_X2_U2 = np.array([
        compute_jacobian(observable_dictionary, x, u) @ known_dynamics(x, u) 
        for x, u in zip(X2, U2)  # Process all samples, not just 500
    ])
    
    # Convert sparse matrix to dense for pinv computation
    Theta_X2_U2_dense = Theta_X2_U2.toarray()
    
    # Print shapes for debugging
    print("Theta_X2_U2_dense shape:", Theta_X2_U2_dense.shape)
    print("J_X2_U2 shape:", J_X2_U2.shape)
    
    # Compute Lf with proper dimensions
    Lf = pinv(Theta_X2_U2_dense) @ (Theta_X2_U2_dense.T @ J_X2_U2)

    # **Eigendecomposition**
    eigenvalues, eigenvectors = eig(Lf)
    S = eigenvectors
    Lambda = np.diag(eigenvalues)
    S_inv = pinv(S)

    # **Construct Koopman Operator**
    Kf_t2 = S @ np.exp(Lambda * dt) @ S_inv

    # **Learn Unknown Dynamics**
    X1, U1 = D1
    Theta_X1_U1 = csc_matrix([observable_dictionary(x, u) for x, u in zip(X1, U1)])
    Theta_X1_prime_U1 = csc_matrix([observable_dictionary(x_prime, u) for x_prime, u in zip(X1[1:], U1[:-1])])

    # **Use Truncated SVD Again ✅**
    U1, S1, Vt1 = svds(Theta_X1_U1, k=min(100, num_features))
    Theta_X1_U1_reduced = U1 @ np.diag(S1) @ Vt1

    # **Final Koopman Operator**
    # Convert sparse matrix to dense for pinv computation
    Theta_X1_U1_reduced_dense = Theta_X1_U1_reduced.toarray()
    Kht = pinv(Kf_t2 @ Theta_X1_U1_reduced_dense) @ (Kf_t2 @ Theta_X1_prime_U1.toarray())
    Kt = Kf_t2 @ Kht @ Kf_t2

    return Kt

# Regularization using LASSO
def regularize_with_lasso(A, X, X_prime, alpha):
    lasso = Lasso(alpha=alpha)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    X_prime = X_prime.reshape(-1, 1) if X_prime.ndim == 1 else X_prime
    lasso.fit(X, X_prime)
    return lasso.coef_

# Main function
def main():
    x0 = np.zeros(144)  # **Reduced to 144 instead of 140,000**
    u = np.zeros(3)  

    t_span = (0, 10)
    dt = 0.03
    D1, t_eval = generate_trajectory_data(x0, u, t_span, dt)

    num_samples = 10000  # **Reduced from 140,000**
    D2 = generate_phase_space_data(x0, u, num_samples)

    Kt = pi_edmdc(D1, D2, dt)

    alpha = 0.01 * len(D1[0])
    Kt_reg = regularize_with_lasso(Kt, D1[0], D1[1], alpha)

    x_pred = Kt @ observable_dictionary(D1[-1], u)

    print("Predicted next state:", x_pred)

if __name__ == "__main__":
    main()