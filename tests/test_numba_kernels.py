import numpy as np

from sucnr1_metaflex.calibration.fit import _robust_scalar_loss
from sucnr1_metaflex.calibration.numba_kernels import (
    NUMBA_AVAILABLE,
    interp_linear,
    log10_transform,
    loss_name_to_code,
    robust_scalar_loss,
    seahorse_shape,
    seahorse_shape_numpy,
    weighted_residuals,
)


def test_numba_numeric_kernels_match_numpy_paths():
    x = np.array([-2.0, 0.0, 1.5])
    np.testing.assert_allclose(log10_transform(x, use_numba=True), 10.0**x, rtol=1e-12, atol=1e-12)

    sim_t = np.array([0.0, 1.0, 2.0, 3.0])
    sim_y = np.array([0.0, 2.0, 4.0, 9.0])
    obs_t = np.array([0.5, 1.5, 2.5])
    np.testing.assert_allclose(interp_linear(obs_t, sim_t, sim_y, use_numba=True), np.interp(obs_t, sim_t, sim_y), rtol=1e-12, atol=1e-12)

    pred = np.array([1.0, 2.0, 4.0])
    obs = np.array([0.5, 2.5, 3.0])
    weights = np.array([1.0, 2.0, 0.25])
    np.testing.assert_allclose(weighted_residuals(pred, obs, weights, use_numba=True), weights * (pred - obs), rtol=1e-12, atol=1e-12)


def test_numba_loss_values_match_python_implementation():
    residuals = np.array([-3.0, -0.5, 0.0, 1.25, 4.0])
    for name in ["linear", "soft_l1", "huber", "cauchy"]:
        expected = _robust_scalar_loss(residuals, loss_name=name, f_scale=1.7)
        actual = robust_scalar_loss(residuals, loss_name_to_code(name), 1.7, use_numba=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_seahorse_shape_numba_matches_numpy():
    t = np.linspace(0.0, 3.0, 11)
    values = np.array([1.0, 0.5, 1.0, 0.2, 0.25, 1.5, 0.3, 1e-6])
    np.testing.assert_allclose(
        seahorse_shape(t, 1, values, use_numba=True),
        seahorse_shape_numpy(t, 1, values),
        rtol=1e-12,
        atol=1e-12,
    )


def test_numba_import_is_optional_contract():
    assert isinstance(NUMBA_AVAILABLE, bool)
