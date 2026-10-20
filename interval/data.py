"""Synthetic observations. These are not satellite acquisitions or medical data."""
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class Scene:
    before: np.ndarray
    after: np.ndarray
    changed: np.ndarray
    expected_background: np.ndarray | None = None

    def identity(self) -> str:
        return sha256(self.before.tobytes() + self.after.tobytes()).hexdigest()


def synthetic_scene(seed: int, tail: float = 1.1, side: int = 32,
                    change_fraction: float = .08, shift: float = 0.) -> Scene:
    """Lognormal observation model; conditional background mean is analytic.

    Y0 = exp(0.08 + 0.9 log(X) + shift + epsilon), epsilon ~ N(0, .22²).
    A smooth random field defines a contiguous, independently generated mask.
    Speckle, orbital geometry and complex radar phase are not simulated.
    """
    rng = np.random.default_rng(seed)
    z = gaussian_filter(rng.normal(size=(side, side)), .65)
    z = (z - z.mean()) / max(z.std(), 1e-9)
    x = np.exp(tail * z).astype(np.float64)
    y = np.exp(.08 + .9 * np.log(x) + shift + rng.normal(0, .22, x.shape))
    field = gaussian_filter(rng.normal(size=x.shape), 1.1)
    mask = field > np.quantile(field, 1 - change_fraction)
    direction = gaussian_filter(rng.normal(size=x.shape), 1.8) > 0
    y = np.where(mask, y * np.where(direction, 2.8, .28), y)
    oracle = np.exp(.08 + .9 * np.log(x) + shift + .22 ** 2 / 2)
    return Scene(x, y, mask, oracle)


def split_scenes(seed: int, tail: float, side: int = 32, shift: float = 0.):
    """Distinct scene seeds; labels from training scenes are never returned."""
    train = [synthetic_scene(seed * 10000 + i, tail, side) for i in (11, 13, 17)]
    calibration = [synthetic_scene(seed * 10000 + i, tail, side) for i in (101, 103)]
    test = [synthetic_scene(seed * 10000 + i, tail, side, shift=shift) for i in (1009, 1013)]
    return train, calibration, test


class Marginal:
    """Midrank empirical CDF fitted only to the training observations."""
    def __init__(self, values):
        self.values = np.sort(np.asarray(values, dtype=float).ravel())
        if self.values.size < 2 or not np.isfinite(self.values).all():
            raise ValueError("Marginals require at least two finite values")

    def cdf(self, values):
        lo = np.searchsorted(self.values, values, side="left")
        hi = np.searchsorted(self.values, values, side="right")
        return np.clip((lo + hi) / (2 * len(self.values)), 0, 1 - 1e-9)

    def quantile(self, values):
        return np.quantile(self.values, np.clip(values, 0, 1))


def bins(values, marginal: Marginal, dimension: int):
    return np.minimum((marginal.cdf(values) * dimension).astype(int), dimension - 1)


def validate_uploaded_splits(payload):
    """Load arrays only, never pickled objects. Strict independent-split schema."""
    import io
    import zipfile
    import math
    if len(payload) > 12 * 1024 * 1024:
        raise ValueError("Dataset exceeds the 12 MB limit")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.infolist()) > 8 or sum(item.file_size for item in archive.infolist()) > 24 * 1024 * 1024:
            raise ValueError("The expanded dataset exceeds the supported array budget")
        for entry in archive.infolist():
            if not entry.filename.endswith(".npy") or "/" in entry.filename or "\\" in entry.filename:
                raise ValueError("The dataset may contain only named NPY arrays")
            with archive.open(entry) as array_file:
                version = np.lib.format.read_magic(array_file)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(array_file)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(array_file)
                else:
                    raise ValueError("Use NPY version 1 or 2 arrays")
                if len(shape) != 2 or min(shape) < 8 or max(shape) > 512 or dtype.kind not in "biuf":
                    raise ValueError("Array headers must declare 2D numeric arrays with sides 8–512")
                if array_file.tell() + math.prod(shape) * dtype.itemsize != entry.file_size:
                    raise ValueError("Array header and data byte counts disagree")
    loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    expected = {f"{part}_{name}" for part in ("train", "calibration", "test")
                for name in (("before", "after") if part == "train" else ("before", "after", "mask"))}
    if set(loaded.files) != expected:
        raise ValueError("Expected train_before/train_after and calibration/test before, after, mask arrays")
    result, identities = [], set()
    total = 0
    for part in ("train", "calibration", "test"):
        x, y = (np.asarray(loaded[f"{part}_{name}"], dtype=float) for name in ("before", "after"))
        if x.ndim != 2 or x.shape != y.shape or min(x.shape) < 8 or max(x.shape) > 512:
            raise ValueError("Each before/after pair must have the same 2D shape, with sides 8–512")
        if not (np.isfinite(x).all() and np.isfinite(y).all() and (x > 0).all() and (y > 0).all()):
            raise ValueError("All observations must be finite and strictly positive")
        if max(float(x.max()), float(y.max())) > 1e12:
            raise ValueError("Rescale intensity observations below 1e12 before import")
        if part == "train" and (np.std(x) < 1e-12 or np.std(y) < 1e-12):
            raise ValueError("Training observations must contain intensity variation")
        mask = np.zeros(x.shape, dtype=bool) if part == "train" else loaded[f"{part}_mask"]
        if mask.shape != x.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("Masks must be binary with the same shape as observations")
        scene = Scene(x, y, mask.astype(bool))
        identity = scene.identity()
        if identity in identities:
            raise ValueError("Identical observations occur in more than one split")
        identities.add(identity)
        total += x.size
        result.append([scene])
    if total > 150000:
        raise ValueError("Limit the three splits to 150,000 total pixels")
    return tuple(result)
