"""Common conditional-mean path makes the estimand explicit."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr, ndtri
from .data import Marginal, bins
from .quantum import fit_copula, finite_shots, readout_bit_flip


METHOD_NAMES = {
    "log_ratio": "Log ratio", "copula_rank_mean": "Copula · mean in rank space",
    "copula_intensity_mean": "Copula · mean in intensity space", "smoothed_copula": "Smoothed copula",
    "gaussian_copula": "Gaussian copula · midpoint", "neural_regression": "Neural conditional mean",
    "qcbm_exact": "QCBM · exact simulation", "qcbm_shots": "QCBM · finite measurements",
    "qcbm_readout": "QCBM · measurement-bit noise"}


def conditional_mean(joint, y_values, fallback):
    mass = joint.sum(axis=1)
    return np.divide(joint @ np.asarray(y_values), mass,
                     out=np.full(len(joint), float(fallback)), where=mass > 1e-15)


def fit_estimators(x, y, config):
    d = 2 ** config.bits
    mx, my = Marginal(x), Marginal(y)
    xb, yb = bins(x, mx, d), bins(y, my, d)
    counts = np.zeros((d, d))
    np.add.at(counts, (xb, yb), 1)
    joint = counts / counts.sum()
    centers = (np.arange(d) + .5) / d
    representatives = my.quantile(centers)
    # Both classical and quantum distributions use this same original-intensity functional.
    lut = {"copula_intensity_mean": conditional_mean(joint, representatives, np.mean(y))}
    rank_lut = conditional_mean(joint, centers, .5)
    lut["copula_rank_mean"] = my.quantile(rank_lut)
    smooth = gaussian_filter(counts, sigma=1., mode="reflect") + .01
    lut["smoothed_copula"] = conditional_mean(smooth, representatives, np.mean(y))
    zx = ndtri(np.clip(mx.cdf(x), 1e-4, 1 - 1e-4))
    zy = ndtri(np.clip(my.cdf(y), 1e-4, 1 - 1e-4))
    rho = float(np.clip(np.corrcoef(zx, zy)[0, 1], -.98, .98))
    edges = ndtri(np.linspace(0, 1, d + 1))
    means = rho * ndtri(centers)
    gaussian = np.diff(ndtr((edges[None, :] - means[:, None]) / np.sqrt(1 - rho * rho)), axis=1) / d
    lut["gaussian_copula"] = conditional_mean(gaussian, representatives, np.mean(y))
    p, quantum = fit_copula(joint, config.bits, config.seed, config.quantum_steps)
    lut["qcbm_exact"] = conditional_mean(p, representatives, np.mean(y))
    measured, sample_counts = finite_shots(p, config.shots, config.seed + 701)
    lut["qcbm_shots"] = conditional_mean(measured, representatives, np.mean(y))
    noisy = readout_bit_flip(p, config.bits, config.readout_error)
    noisy_measured, _ = finite_shots(noisy, config.shots, config.seed + 701)
    lut["qcbm_readout"] = conditional_mean(noisy_measured, representatives, np.mean(y))

    torch.manual_seed(config.seed)
    net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32),
                              torch.nn.Tanh(), torch.nn.Linear(32, 1))
    tx = torch.tensor(mx.cdf(x).reshape(-1, 1), dtype=torch.float32)
    scale = max(float(np.std(y)), 1e-6)
    offset = float(np.mean(y))
    ty = torch.tensor(((y - offset) / scale).reshape(-1, 1), dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=.01)
    net.train()
    neural_losses = []
    for _ in range(config.neural_steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(tx), ty)
        loss.backward()
        opt.step()
        neural_losses.append(float(loss.detach()))
    net.eval()
    statistics = {"occupied_bins": int((counts > 0).sum()), "total_bins": int(d * d),
                  "occupancy": float((counts > 0).mean()), "training_pixels": int(len(x)),
                  "empty_conditioning_bins_after_shots": int((sample_counts.sum(1) == 0).sum()),
                  "median_samples_per_condition": float(np.median(sample_counts.sum(1))),
                  "neural_parameters": sum(v.numel() for v in net.parameters()), "rho": rho}
    statistics["median_observations_per_condition"] = float(np.median(counts.sum(1)))
    statistics["minimum_observations_per_condition"] = int(counts.sum(1).min())
    return {"marginal_x": mx, "marginal_y": my, "lookup": lut, "joint": joint,
            "quantum_joint": p, "quantum": quantum, "network": net, "scale": scale,
            "offset": offset, "statistics": statistics, "neural_loss": neural_losses}


def update_measurements(fitted, config):
    """Refresh inference-only controls while retaining the same trained circuit."""
    p = fitted["quantum_joint"]
    d = len(p)
    representatives = fitted["marginal_y"].quantile((np.arange(d) + .5) / d)
    measured, counts = finite_shots(p, config.shots, config.seed + 701)
    noisy, _ = finite_shots(readout_bit_flip(p, config.bits, config.readout_error), config.shots, config.seed + 701)
    fitted["lookup"]["qcbm_shots"] = conditional_mean(measured, representatives, fitted["offset"])
    fitted["lookup"]["qcbm_readout"] = conditional_mean(noisy, representatives, fitted["offset"])
    fitted["statistics"]["empty_conditioning_bins_after_shots"] = int((counts.sum(1) == 0).sum())
    fitted["statistics"]["median_samples_per_condition"] = float(np.median(counts.sum(1)))


def restore_fitted(directory):
    """Restore a local inference bundle without pickle or executable model code."""
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    directory = Path(directory)
    meta = json.loads((directory / "inference.json").read_text())
    arrays = np.load(directory / "inference-arrays.npz", allow_pickle=False)
    net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    net.load_state_dict(load_file(str(directory / "conditional-network.safetensors")))
    net.eval()
    return {"marginal_x": Marginal(arrays["marginal_x"]), "marginal_y": Marginal(arrays["marginal_y"]),
            "lookup": {key.removeprefix("lookup_"): arrays[key] for key in arrays.files if key.startswith("lookup_")},
            "network": net, "scale": meta["scale"], "offset": meta["offset"]}


def predict_background(fitted, method, x):
    if method == "neural_regression":
        with torch.inference_mode():
            tx = torch.tensor(fitted["marginal_x"].cdf(x).reshape(-1, 1), dtype=torch.float32)
            return np.maximum((fitted["network"](tx).numpy().ravel() * fitted["scale"] + fitted["offset"]).reshape(x.shape), 1e-8)
    if method == "log_ratio":
        return x
    lookup = fitted["lookup"][method]
    return lookup[bins(x, fitted["marginal_x"], len(lookup))]


def change_score(fitted, method, scene, score_mode="absolute"):
    background = predict_background(fitted, method, scene.before)
    if method == "log_ratio" or score_mode == "log":
        score = np.abs(np.log(scene.after) - np.log(np.maximum(background, 1e-8)))
    else:
        score = np.abs(scene.after - background)
    return score, background
