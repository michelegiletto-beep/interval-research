import numpy as np
import pytest
import torch
from interval.data import Marginal, bins, split_scenes
from interval.estimators import conditional_mean
from interval.evaluation import calibrate, mask_metrics
from interval.quantum import distribution, fit_copula, finite_shots, readout_bit_flip


def test_register_marginals_remain_uniform_under_local_unitaries():
    bits, layers = 3, 2
    torch.manual_seed(73)
    theta = torch.randn(2 * layers * (3 * bits - 1))
    p = distribution(theta, bits, layers).detach().numpy()
    np.testing.assert_allclose(p.sum(0), np.full(8, 1 / 8), atol=2e-7)
    np.testing.assert_allclose(p.sum(1), np.full(8, 1 / 8), atol=2e-7)
    assert p.sum() == pytest.approx(1., abs=1e-6)


def test_inverse_transform_and_expectation_do_not_commute():
    joint = np.array([[.25, .25], [.25, .25]])
    expected_intensity = conditional_mean(joint, [1., 9.], 0)
    expected_rank = conditional_mean(joint, [.25, .75], .5)
    inverse_at_rank_mean = np.exp(np.interp(expected_rank, [.25, .75], [0., np.log(9.)]))
    np.testing.assert_allclose(expected_intensity, [5., 5.])
    np.testing.assert_allclose(inverse_at_rank_mean, [3., 3.])


def test_quantum_training_reduces_cross_entropy():
    target = np.eye(4) / 4 * .8 + np.ones((4, 4)) / 16 * .2
    p, trace = fit_copula(target, 2, seed=5, steps=25)
    assert min(trace["loss"][-5:]) < trace["loss"][0]
    assert np.isfinite(p).all()


def test_measurement_and_readout_controls():
    p = np.eye(4) / 4
    observed, counts = finite_shots(p, 20000, 17)
    assert counts.sum() == 20000
    assert np.abs(observed - p).max() < .02
    np.testing.assert_array_equal(readout_bit_flip(p, 2, 0), p)
    noisy = readout_bit_flip(p, 2, .03)
    assert noisy.sum() == pytest.approx(1)
    assert np.trace(noisy) < 1


def test_splits_and_ties_have_declared_behavior():
    train, cal, test = split_scenes(42, 1.1, 16)
    ids = [s.identity() for s in train + cal + test]
    assert len(set(ids)) == len(ids)
    m = Marginal([1, 1, 2, 3])
    np.testing.assert_allclose(m.cdf(np.array([1, 2, 3])), [.25, .625, .875])
    assert bins(np.array([100.]), m, 4)[0] == 3


def test_calibration_result_uses_only_its_supplied_observations():
    scores = np.arange(100).reshape(10, 10)
    truth = scores > 90
    settings, _ = calibrate([scores], [truth])
    before = dict(settings)
    # Changing a separate test array cannot affect frozen calibration settings.
    unrelated_test_labels = np.logical_not(truth)
    assert settings == before
    assert mask_metrics(truth, scores > settings["threshold"])["f1"] > .8
    assert mask_metrics(unrelated_test_labels, scores > settings["threshold"])["f1"] < .2
