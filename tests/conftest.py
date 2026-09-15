"""Differential-testing harness: upstream LightPipes as a live oracle.

`_ref/LightPipes_ref` is a vendored copy of the unmodified upstream package
(opticspy/lightpipes @ the commit pinned in `_ref/UPSTREAM_COMMIT`), imported
under a distinct name so it can coexist with the optimized fork in one process.

Every test asserts that the fork reproduces the oracle. The oracle is never
modified, so a failure always means the fork changed behaviour.
"""
import sys
import pathlib

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_REF_DIR = _ROOT / "_ref"

# Import the oracle first, from its own directory, then the fork from the repo root.
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import LightPipes_ref as ref  # noqa: E402
import OptimLightPipes as new  # noqa: E402


UPSTREAM_COMMIT = (_REF_DIR / "UPSTREAM_COMMIT").read_text().strip()

# Optical parameters shared by every test, in SI units.
WAVELENGTH = 632.8e-9
GRID_SIZE = 5.0e-3


@pytest.fixture(scope="session")
def lp():
    """The two implementations under comparison: (fork, oracle)."""
    return new, ref


def both_dtypes(fn):
    """Parametrize a test over complex128 and complex64."""
    return pytest.mark.parametrize(
        "dtype", [np.complex128, np.complex64], ids=["complex128", "complex64"]
    )(fn)


def make_pair(mod_pair, N, dtype=np.complex128, size=GRID_SIZE, lam=WAVELENGTH):
    """Build the same starting field in both implementations."""
    new_mod, ref_mod = mod_pair
    return (
        new_mod.Begin(size, lam, N, dtype),
        ref_mod.Begin(size, lam, N, dtype),
    )


def assert_fields_match(f_new, f_ref, rtol, atol=0.0, label=""):
    """Compare two Field objects' complex data."""
    a = np.asarray(f_new.field)
    b = np.asarray(f_ref.field)
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    assert_arrays_match(a, b, rtol, atol, label)


def assert_arrays_match(a, b, rtol, atol=0.0, label=""):
    a = np.asarray(a)
    b = np.asarray(b)
    if atol == 0.0:
        # Scale the absolute floor to the data, so near-zero pixels do not
        # dominate a pure relative comparison.
        scale = np.abs(b).max()
        atol = rtol * scale if scale > 0 else rtol
    diff = np.abs(a - b)
    worst = diff.max() if diff.size else 0.0
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    assert ok, (
        f"{label}: max abs diff {worst:.3e} exceeds "
        f"rtol={rtol:.1e} atol={atol:.3e} (ref max |v|={np.abs(b).max():.3e})"
    )
