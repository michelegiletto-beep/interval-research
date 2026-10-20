"""Calibration-only threshold selection. Test labels are evaluation-only."""
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve


def mask_metrics(truth, predicted):
    truth, predicted = np.asarray(truth, bool).ravel(), np.asarray(predicted, bool).ravel()
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "false_positive_rate": fp / max(fp + tn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def spatial_mask(score, threshold, window=1, fraction=.5):
    mask = np.asarray(score) > threshold
    return mask if window == 1 else uniform_filter(mask.astype(float), size=window, mode="reflect") >= fraction


def calibrate(scores, masks):
    values = np.concatenate([x.ravel() for x in scores])
    candidates = np.unique(np.quantile(values, np.linspace(.50, .995, 45)))
    candidates = np.unique(np.concatenate(([np.nextafter(values.min(), -np.inf)], candidates,
                                            [np.nextafter(values.max(), np.inf)])))
    best = None
    # Same finite grid for every method; only calibration labels are consumed.
    for threshold in candidates:
        for window in (1, 3):
            predicted = [spatial_mask(s, threshold, window) for s in scores]
            f1 = np.mean([mask_metrics(t, p)["f1"] for t, p in zip(masks, predicted)])
            fp = sum(mask_metrics(t, p)["fp"] for t, p in zip(masks, predicted))
            if best is None or (f1, -fp) > (best[0], -best[2]):
                best = (float(f1), {"threshold": float(threshold), "window": window, "fraction": .5}, fp)
    return best[1], best[0]


def continuous_metrics(truth, scores):
    y, s = np.asarray(truth, int).ravel(), np.asarray(scores, float).ravel()
    if len(np.unique(y)) < 2:
        return {"average_precision": None, "recall_at_1pct_fpr": None}
    fpr, tpr, _ = roc_curve(y, s)
    return {"average_precision": float(average_precision_score(y, s)),
            "recall_at_1pct_fpr": float(np.max(tpr[fpr <= .01], initial=0))}


def bootstrap_mean(values, seed=173, resamples=2000):
    values = np.asarray(values, float)
    if not len(values):
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(1)
    return {"mean": float(values.mean()), "low": float(np.quantile(means, .025)),
            "high": float(np.quantile(means, .975)), "n": int(len(values))}
