import numpy as np

def loss_function(p_pred, p_true, T_pred, T_true, alpha, beta):
    """
    Computes the weighted loss between predicted and true pressure and temperature values.
    
    Parameters:
    - p_pred: Predicted pressure drop (scalar).
    - p_true: True pressure drop (scalar).
    - T_pred: Predicted temperature matrix.
    - T_true: True temperature matrix.
    - alpha: Weight for pressure loss (scalar between 0 and 1).
    - beta: Weight for temperature loss (scalar between 0 and 1), where alpha + beta = 1.
    
    Returns:
    - L: Combined loss value (scalar).
    """
    # Compute pressure loss (L_p)
    L_p = np.abs(p_pred - p_true) / (np.abs(p_pred) + np.abs(p_true))
    
    # Compute temperature loss (L_T) using Frobenius norm
    numerator = np.linalg.norm(T_pred - T_true, ord='fro')
    denominator = np.linalg.norm(T_pred, ord='fro') + np.linalg.norm(T_true, ord='fro')
    L_T = numerator / denominator
    
    # Combine losses with weights
    L = alpha * L_p + beta * L_T
    return L
