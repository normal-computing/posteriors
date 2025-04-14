import torch
from torch import Tensor


def standardize_ou_process(
    A: Tensor, b: Tensor, E: Tensor, tol=1e-10
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Standardizes the OU process dx = A x dt + b dt + E dw into
    dy = G y dt + V dw, where V is diagonal with 0s and 1s.

    Uses invertible transformation
    y = T x + v   and  x = T_inv (y - v)
    where T is a matrix and v is a vector.

    Args:
        A: Drift matrix (d, d)
        b: Drift vector (d)
        E: Square root of diffusion matrix (d, d)
            Not necessarily invertible.
        tol: Threshold for singular values.

    Returns:
        T: Transformation matrix, (d, d)
        T_inv: Inverse of T, (d, d)
        v: Transformation vector, (d,)
        G: Transformed drift, (d, d)
        V_diag_vector: Vector of diagonal values with only 0s and 1s, (d,)
    """
    # # Invertible E case (easy) would be
    # T = torch.linalg.inv(E)
    # v = T @ torch.linalg.inv(A) @ b
    # G = T @ A @ E
    # V = torch.ones_like(b)
    # return T, E, v, G, V

    # Compute the singular value decomposition of E.
    # Here, E = U @ diag(S) @ Vh, with Vh = V^T.
    U, S, Vh = torch.linalg.svd(E, full_matrices=True)

    # Determine the effective rank: active noise directions have S > tol.
    mask = S > tol

    # Build a diagonal matrix D where for active directions we use 1 / sqrt(S), otherwise 1.
    # This choice cancels nonzero singular values and leaves the zero ones untouched.
    d = torch.where(mask, 1.0 / S, torch.ones_like(S))
    D = torch.diag(d)

    # Define the transformation T = D U^T.
    T = D @ U.t()

    # Choose v so that the constant term in the drift vanishes.
    # Starting from the relation T A T^{-1} v = T b, we set v = T (A^+ b),
    # where A^+ is the pseudoinverse of A.
    A_pinv = torch.linalg.pinv(A)
    v = T @ (A_pinv @ b)

    # Compute the transformed drift matrix G = T A T^{-1}.
    T_inv = torch.linalg.inv(T)
    G = T @ A @ T_inv

    # Build the diagonal matrix V with entries 1 for active noise directions and 0 otherwise.
    V_diag_vector = torch.where(mask, torch.ones_like(S), torch.zeros_like(S))
    return T, T_inv, v, G, V_diag_vector
