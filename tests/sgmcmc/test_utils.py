import torch

from posteriors.sgmcmc.utils import standardize_ou_process


def test_standardize_ou_process_invertible_E():
    A = torch.randn(2, 2)
    b = torch.randn(2)
    E = torch.randn(2, 2)
    T, T_inv, v, G, V_diag_vector = standardize_ou_process(A, b, E)
    assert torch.allclose(T @ T_inv, torch.eye(2), atol=1e-6)
    assert torch.allclose(T_inv @ T, torch.eye(2), atol=1e-6)

    x = torch.randn(2)
    y = T @ x + v
    assert torch.allclose(T_inv @ (y - v), x)

    assert torch.equal(V_diag_vector, torch.ones(2))


def test_standardize_ou_process_non_invertible_E():
    A = torch.randn(2, 2)
    b = torch.randn(2)
    E = torch.tensor([[0.0, 0.0], [0.0, 1.0]])

    T, T_inv, v, G, V_diag_vector = standardize_ou_process(A, b, E)
    assert torch.allclose(T @ T_inv, torch.eye(2), atol=1e-6)
    assert torch.allclose(T_inv @ T, torch.eye(2), atol=1e-6)

    x = torch.randn(2)
    y = T @ x + v
    assert torch.allclose(T_inv @ (y - v), x)

    assert torch.equal(V_diag_vector, torch.tensor([1.0, 0.0]))


def test_standardize_ou_process_non_invertible_E_non_invertible_A():
    A = torch.tensor([[0.0, -1.3], [0.0, 0.8]])
    b = torch.zeros(2)
    E = torch.tensor([[0.0, 0.0], [0.0, 1.0]])

    T, T_inv, v, G, V_diag_vector = standardize_ou_process(A, b, E)
    assert torch.allclose(T @ T_inv, torch.eye(2), atol=1e-6)
    assert torch.allclose(T_inv @ T, torch.eye(2), atol=1e-6)

    x = torch.randn(2)
    y = T @ x + v
    assert torch.allclose(T_inv @ (y - v), x)

    assert torch.equal(V_diag_vector, torch.tensor([1.0, 0.0]))


def test_standardize_ou_process_no_op():
    A = torch.randn(2, 2)
    b = torch.zeros(2)
    E = torch.eye(2)

    T, T_inv, v, G, V_diag_vector = standardize_ou_process(A, b, E)
    assert torch.allclose(T @ T_inv, torch.eye(2), atol=1e-6)
    assert torch.allclose(T_inv @ T, torch.eye(2), atol=1e-6)

    x = torch.randn(2)
    y = T @ x + v
    assert torch.allclose(T_inv @ (y - v), x)

    assert torch.allclose(G, A)
    assert torch.allclose(v, b)
    assert torch.equal(V_diag_vector, torch.ones(2))
