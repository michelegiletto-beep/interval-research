"""Small exact pure-state quantum copula, trained by classical autodiff.

This is a simulator, not an IonQ emulator. Local unitary operations preserve
the uniform register marginals of a maximally entangled state.
"""
from time import perf_counter
import numpy as np
import torch


def apply_gate(state, gate, wires, qubits):
    remaining = [i for i in range(qubits) if i not in wires]
    order = list(wires) + remaining
    inverse = np.argsort(order).tolist()
    tensor = state.reshape([2] * qubits).permute(order).reshape(2 ** len(wires), -1)
    return (gate @ tensor).reshape([2] * qubits).permute(inverse).reshape(-1)


def distribution(theta, bits, layers=2):
    d, n = 2 ** bits, 2 * bits
    state = (torch.eye(d, dtype=torch.complex64) / np.sqrt(d)).reshape(-1)
    index = 0
    for register in range(2):
        for _ in range(layers):
            for wire in range(bits):
                a, b = theta[index], theta[index + 1]
                index += 2
                c, s = torch.cos(a / 2).to(torch.complex64), -1j * torch.sin(a / 2)
                rx = torch.stack([c, s, s, c]).reshape(2, 2)
                rz = torch.diag(torch.stack([torch.exp(-.5j * b), torch.exp(.5j * b)]))
                state = apply_gate(state, rz @ rx, [register * bits + wire], n)
            for wire in range(bits - 1):
                a = theta[index]
                index += 1
                signs = torch.tensor([1., -1., -1., 1.])
                gate = torch.diag(torch.exp(-.5j * a * signs))
                state = apply_gate(state, gate, [register * bits + wire, register * bits + wire + 1], n)
    return state.abs().square().reshape(d, d)


def fit_copula(target, bits, seed=1, steps=100, layers=2):
    torch.manual_seed(seed)
    count = 2 * layers * (2 * bits + bits - 1)
    theta = torch.nn.Parameter(torch.randn(count) * .3)
    optimizer = torch.optim.Adam([theta], lr=.07)
    observed = torch.as_tensor(target, dtype=torch.float32)
    observed = observed / observed.sum()
    start, losses = perf_counter(), []
    best_loss, best_theta = float("inf"), theta.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad()
        p = distribution(theta, bits, layers)
        # Cross entropy; probability floor is numerical stabilization, declared here.
        loss = -(observed * torch.log(p.clamp_min(1e-9))).sum()
        loss.backward()
        losses.append(float(loss.detach()))
        if losses[-1] < best_loss:
            best_loss, best_theta = losses[-1], theta.detach().clone()
        optimizer.step()
    with torch.no_grad():
        p = distribution(best_theta, bits, layers).numpy().astype(float)
    p /= p.sum()
    return p, {"qubits": 2 * bits, "parameters": count, "steps": steps,
               "seconds": perf_counter() - start, "loss": losses,
               "max_marginal_error": float(max(abs(p.sum(0) - 1 / len(p)).max(), abs(p.sum(1) - 1 / len(p)).max())),
               "parameters_trained": best_theta.tolist(), "backend": "torch-pure-state-autodiff"}


def finite_shots(probabilities, shots, seed):
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(shots, probabilities.ravel() / probabilities.sum()).reshape(probabilities.shape)
    return counts / shots, counts


def readout_bit_flip(probabilities, bits, error=.01):
    """Independent terminal measurement-bit flips, NOT a trapped-ion noise model."""
    p = probabilities.reshape(-1).copy()
    ids = np.arange(len(p))
    for wire in range(2 * bits):
        p = (1 - error) * p + error * p[ids ^ (1 << wire)]
    return p.reshape(probabilities.shape)
