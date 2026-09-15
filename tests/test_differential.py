"""Differential tests: the optimized fork must reproduce upstream LightPipes.

Tolerances are per-routine. Direct/real-space kernels are held near machine
precision; FFT-based paths get slightly more room because operation order
differs. complex64 fields carry ~7 decimal digits, so they are held looser.
"""
import numpy as np
import pytest

from conftest import (
    GRID_SIZE,
    WAVELENGTH,
    assert_arrays_match,
    assert_fields_match,
    both_dtypes,
    make_pair,
)

# Per-dtype tolerance for kernels expected to be near-exact.
EXACT = {np.complex128: 1e-13, np.complex64: 2e-5}
# FFT-based propagators: operation order may differ.
FFT = {np.complex128: 1e-11, np.complex64: 5e-5}


def _aperture(f_new, f_ref, lp, R=1.0e-3):
    new, ref = lp
    return new.CircAperture(f_new, R), ref.CircAperture(f_ref, R)


# --------------------------------------------------------------------------
# Propagators
# --------------------------------------------------------------------------

@both_dtypes
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("z", [0.05, 0.3])
def test_fresnel(lp, N, dtype, z):
    new, ref = lp
    a, b = make_pair(lp, N, dtype)
    a, b = _aperture(a, b, lp)
    assert_fields_match(
        new.Fresnel(a, z), ref.Fresnel(b, z), FFT[dtype], label=f"Fresnel N={N} z={z}"
    )


@both_dtypes
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("z", [0.05, 0.3])
def test_forvard(lp, N, dtype, z):
    new, ref = lp
    a, b = make_pair(lp, N, dtype)
    a, b = _aperture(a, b, lp)
    assert_fields_match(
        new.Forvard(a, z), ref.Forvard(b, z), FFT[dtype], label=f"Forvard N={N} z={z}"
    )


@both_dtypes
def test_forvard_negative_z(lp, dtype):
    """Back-propagation exercises the other sign branch."""
    new, ref = lp
    a, b = make_pair(lp, 64, dtype)
    a, b = _aperture(a, b, lp)
    assert_fields_match(
        new.Forvard(a, -0.1), ref.Forvard(b, -0.1), FFT[dtype], label="Forvard z<0"
    )


@both_dtypes
@pytest.mark.parametrize("old_n,new_n", [(32, 16), (48, 24)])
def test_forward(lp, dtype, old_n, new_n):
    """Direct-integration propagator. Small grids: upstream is O(N^4)."""
    new, ref = lp
    a, b = make_pair(lp, old_n, dtype)
    a, b = _aperture(a, b, lp)
    assert_fields_match(
        new.Forward(a, 0.2, 2 * GRID_SIZE, new_n),
        ref.Forward(b, 0.2, 2 * GRID_SIZE, new_n),
        EXACT[dtype],
        label=f"Forward {old_n}->{new_n}",
    )


@both_dtypes
@pytest.mark.parametrize("save_ram", [False, True], ids=["array", "save_ram"])
def test_steps(lp, dtype, save_ram):
    """Crank-Nicolson BPM; both dispatch paths."""
    new, ref = lp
    a, b = make_pair(lp, 64, dtype)
    a, b = _aperture(a, b, lp)
    assert_fields_match(
        new.Steps(a, 0.01, 5, 1.0, save_ram),
        ref.Steps(b, 0.01, 5, 1.0, save_ram),
        EXACT[dtype],
        label=f"Steps save_ram={save_ram}",
    )


@both_dtypes
def test_steps_with_refraction(lp, dtype):
    """Complex refractive index: exercises absorption + phase terms."""
    new, ref = lp
    N = 64
    a, b = make_pair(lp, N, dtype)
    a, b = _aperture(a, b, lp)
    x = np.linspace(-1, 1, N)
    refr = 1.0 + 0.01 * np.exp(-(x[:, None] ** 2 + x[None, :] ** 2)) + 0.001j
    assert_fields_match(
        new.Steps(a, 0.01, 3, refr),
        ref.Steps(b, 0.01, 3, refr),
        EXACT[dtype],
        label="Steps refr",
    )


@both_dtypes
def test_lens_farfield(lp, dtype):
    new, ref = lp
    a, b = make_pair(lp, 64, dtype)
    a, b = _aperture(a, b, lp)
    assert_fields_match(
        new.LensFarfield(a, 0.5), ref.LensFarfield(b, 0.5), FFT[dtype],
        label="LensFarfield",
    )


# --------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------

@both_dtypes
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"magnif": 1.5},
        {"angle": 30.0},
        {"x_shift": 0.3e-3, "y_shift": -0.2e-3},
        {"angle": 15.0, "magnif": 0.8, "x_shift": 0.1e-3},
    ],
    ids=["plain", "magnif", "rotate", "shift", "combined"],
)
def test_interpol(lp, dtype, kwargs):
    """Bilinear interpolation via Inv_Squares."""
    new, ref = lp
    a, b = make_pair(lp, 64, dtype)
    a, b = _aperture(a, b, lp)
    assert_fields_match(
        new.Interpol(a, GRID_SIZE, 64, **kwargs),
        ref.Interpol(b, GRID_SIZE, 64, **kwargs),
        EXACT[dtype],
        label=f"Interpol {kwargs}",
    )


# --------------------------------------------------------------------------
# Phase unwrapping and Zernike
# --------------------------------------------------------------------------

def _tilted_field(mod, N, dtype):
    """A field whose phase wraps, so unwrapping is non-trivial."""
    f = mod.Begin(GRID_SIZE, WAVELENGTH, N, dtype)
    f = mod.CircAperture(f, 1.5e-3)
    return mod.Zernike(f, 4, 0, 1.5e-3, 3.0 * WAVELENGTH)


@both_dtypes
@pytest.mark.parametrize("N", [32, 64])
def test_phase_unwrap(lp, dtype, N):
    new, ref = lp
    pa = new.Phase(_tilted_field(new, N, dtype))
    pb = ref.Phase(_tilted_field(ref, N, dtype))
    assert_arrays_match(
        new.PhaseUnwrap(pa), ref.PhaseUnwrap(pb), EXACT[dtype], label=f"PhaseUnwrap N={N}"
    )


@both_dtypes
def test_phase_unwrap_via_phase(lp, dtype):
    """Phase(unwrap=True) is the path ZernikeFit uses."""
    new, ref = lp
    assert_arrays_match(
        new.Phase(_tilted_field(new, 64, dtype), unwrap=True),
        ref.Phase(_tilted_field(ref, 64, dtype), unwrap=True),
        EXACT[dtype],
        label="Phase(unwrap=True)",
    )


@both_dtypes
@pytest.mark.parametrize("n,m", [(2, 0), (3, 1), (4, -2), (5, 5), (6, 0)])
def test_zernike(lp, dtype, n, m):
    new, ref = lp
    a, b = make_pair(lp, 64, dtype)
    assert_fields_match(
        new.Zernike(a, n, m, 2.0e-3, WAVELENGTH),
        ref.Zernike(b, n, m, 2.0e-3, WAVELENGTH),
        EXACT[dtype],
        label=f"Zernike n={n} m={m}",
    )


def test_zernike_fit(lp):
    """ZernikeFit chains unwrap + per-term grid eval + lstsq."""
    new, ref = lp
    ca = new.ZernikeFit(_tilted_field(new, 64, np.complex128), 10, 1.5e-3)
    cb = ref.ZernikeFit(_tilted_field(ref, 64, np.complex128), 10, 1.5e-3)
    # ZernikeFit returns (coefficients, fitted_field) in some versions.
    if isinstance(ca, tuple):
        assert_arrays_match(ca[0], cb[0], 1e-9, label="ZernikeFit coeffs")
    else:
        assert_arrays_match(ca, cb, 1e-9, label="ZernikeFit coeffs")


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------

@both_dtypes
def test_intensity_and_power(lp, dtype):
    new, ref = lp
    a, b = make_pair(lp, 64, dtype)
    a, b = _aperture(a, b, lp)
    a2 = new.Fresnel(a, 0.1)
    b2 = ref.Fresnel(b, 0.1)
    assert_arrays_match(new.Intensity(a2), ref.Intensity(b2), FFT[dtype], label="Intensity")
    assert new.Power(a2) == pytest.approx(ref.Power(b2), rel=FFT[dtype])


def test_lptest_golden():
    """The repo's own historical guard, from LPtest() in LightPipes/__init__.py.

    Uses that function's exact (unit-free) parameters so the hardcoded sum applies.
    """
    import OptimLightPipes as L

    F = L.Begin(1.8, 2.5, 55)
    F = L.Fresnel(F, 10)
    assert float(np.sum(L.Intensity(F))) == pytest.approx(16.893173606654138, rel=1e-12)


# --------------------------------------------------------------------------
# FFT backend
# --------------------------------------------------------------------------

@both_dtypes
@pytest.mark.parametrize("N", [64, 128])
def test_fresnel_pyfftw_matches_numpy(lp, dtype, N):
    """The pyFFTW backend must agree with the numpy one (and with upstream)."""
    new, ref = lp
    a, b = make_pair(lp, N, dtype)
    a, b = _aperture(a, b, lp)
    assert_fields_match(
        new.Fresnel(a, 0.3, usepyFFTW=True),
        ref.Fresnel(b, 0.3),
        FFT[dtype],
        label=f"Fresnel pyFFTW N={N}",
    )


@both_dtypes
def test_forvard_pyfftw_matches_numpy(lp, dtype):
    new, ref = lp
    a, b = make_pair(lp, 128, dtype)
    a, b = _aperture(a, b, lp)
    assert_fields_match(
        new.Forvard(a, 0.3, usepyFFTW=True),
        ref.Forvard(b, 0.3),
        FFT[dtype],
        label="Forvard pyFFTW",
    )


def test_fft_thread_control():
    """Thread count is settable and independent of the numba pool."""
    import OptimLightPipes as L
    from OptimLightPipes._fft import get_fft_threads, set_fft_threads

    original = get_fft_threads()
    try:
        assert set_fft_threads(2) == 2
        F = L.CircAperture(L.Begin(GRID_SIZE, WAVELENGTH, 64), 1e-3)
        one = L.Fresnel(F, 0.3, usepyFFTW=True).field
        set_fft_threads(1)
        two = L.Fresnel(F, 0.3, usepyFFTW=True).field
        assert_arrays_match(one, two, 1e-12, label="fft threads")
    finally:
        set_fft_threads(original)


@both_dtypes
def test_fft_backend_switch_agrees(lp, dtype):
    """Both backends must produce the same field, and match upstream."""
    import OptimLightPipes as L

    new, ref = lp
    original = L.get_fft_backend()
    try:
        L.set_fft_backend("numpy")
        a, b = make_pair(lp, 128, dtype)
        a, b = _aperture(a, b, lp)
        with_numpy = new.Fresnel(a, 0.3).field

        L.set_fft_backend("auto")
        a2, _ = make_pair(lp, 128, dtype)
        a2 = new.CircAperture(a2, 1.0e-3)
        with_auto = new.Fresnel(a2, 0.3).field
    finally:
        L.set_fft_backend(original)

    assert_arrays_match(with_auto, with_numpy, FFT[dtype], label="backend switch")
    assert_arrays_match(with_auto, ref.Fresnel(b, 0.3).field, FFT[dtype],
                        label="auto vs upstream")


def test_fft_backend_rejects_unknown():
    import OptimLightPipes as L

    with pytest.raises(ValueError):
        L.set_fft_backend("fftw3")


# --------------------------------------------------------------------------
# Coordinate-grid consumers
# --------------------------------------------------------------------------
# The mgrid_* grids are cached and shared between Field instances, so any
# routine that shifted them in place had to be changed to work on a copy.
# These cover the routines that do so, including the shifted/rotated variants.

@both_dtypes
@pytest.mark.parametrize(
    "name,call",
    [
        ("axicon", lambda m, f: m.Axicon(f, 0.01, 1.5, 1e-4, 1e-4)),
        ("lens", lambda m, f: m.Lens(f, 0.5, 1e-4, 1e-4)),
        ("cyl_lens", lambda m, f: m.CylindricalLens(f, 0.5, 1e-4, 1e-4, 0.3)),
        ("glens", lambda m, f: m.GLens(f, 0.5)),
        ("tilt", lambda m, f: m.Tilt(f, 1e-3, 1e-3)),
    ],
)
def test_grid_consumers(lp, dtype, name, call):
    new, ref = lp
    a, b = make_pair(lp, 64, dtype)
    assert_fields_match(call(new, a), call(ref, b), EXACT[dtype], label=name)


def test_mgrid_cached_and_readonly():
    """Grids are shared, so they must be immutable and identical per geometry."""
    import OptimLightPipes as L

    f1 = L.Begin(GRID_SIZE, WAVELENGTH, 64)
    f2 = L.Begin(GRID_SIZE, WAVELENGTH, 64)
    y1, x1 = f1.mgrid_cartesian
    y2, x2 = f2.mgrid_cartesian
    assert x1 is x2 and y1 is y2, "same geometry should share one cached grid"
    assert not x1.flags.writeable

    # A different geometry must not collide with the cached one.
    f3 = L.Begin(2 * GRID_SIZE, WAVELENGTH, 64)
    y3, x3 = f3.mgrid_cartesian
    assert x3 is not x1
    assert np.allclose(x3, 2 * np.asarray(x1))
