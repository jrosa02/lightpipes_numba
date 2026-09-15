"""Optional-numba shim.

numba is a hard dependency in pyproject.toml, but the library must still import
and produce correct results without it (and under NUMBA_DISABLE_JIT=1). This
module exposes `njit` / `guvectorize` that fall back to pure-Python behaviour
when numba is unavailable, plus the shared kernel options.

`fastmath` is OFF by default. See config._FASTMATH: results then stay as close
as possible to upstream, so a differential-test failure is unambiguously a
porting bug rather than floating-point reassociation.
"""
import os

from .config import _FASTMATH

try:
    from numba import guvectorize as _nb_guvectorize
    from numba import njit as _nb_njit
    from numba import get_num_threads as _nb_get_num_threads
    from numba import set_num_threads as _nb_set_num_threads

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False
    _nb_guvectorize = _nb_njit = None
    _nb_get_num_threads = _nb_set_num_threads = None

# NUMBA_DISABLE_JIT still runs through numba's dispatcher (in the interpreter),
# so kernels stay correct; we only use this to skip parallel/cache setup.
JIT_DISABLED = os.environ.get("NUMBA_DISABLE_JIT", "0") not in ("0", "", "false")


def fastmath_enabled():
    """Read the switch at decoration time (see config._FASTMATH)."""
    return bool(_FASTMATH)


if HAVE_NUMBA:

    def njit(*args, **kwargs):
        kwargs.setdefault("cache", True)
        kwargs.setdefault("fastmath", fastmath_enabled())
        return _nb_njit(*args, **kwargs)

    def guvectorize(signatures, layout, **kwargs):
        """guvectorize with this project's defaults.

        target='parallel' makes numba parallelize the gufunc's broadcast axis,
        which is where every kernel here exposes its independent work.
        """
        kwargs.setdefault("nopython", True)
        kwargs.setdefault("target", "parallel")
        kwargs.setdefault("cache", True)
        kwargs.setdefault("fastmath", fastmath_enabled())
        return _nb_guvectorize(signatures, layout, **kwargs)

    def get_num_threads():
        return _nb_get_num_threads()

    def set_num_threads(n):
        """Limit the threads used by the gufunc kernels.

        Note pyFFTW has its own `threads` argument; setting both high can
        oversubscribe the machine.
        """
        return _nb_set_num_threads(n)

else:  # pragma: no cover - fallback path

    def njit(*args, **kwargs):
        """No-op decorator: the Python body runs as written."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn

        return wrap

    def guvectorize(signatures, layout, **kwargs):
        """Emulate a gufunc with numpy broadcasting.

        Only used when numba is missing. The wrapped kernel writes into an
        output argument, so allocate it and loop over the broadcast axes.
        """
        import numpy as np

        def wrap(fn):
            n_core_out = layout.split("->")[1].count("(")
            if n_core_out != 1:
                raise NotImplementedError("fallback supports one output")

            def caller(*args, **kw):
                raise RuntimeError(
                    "OptimLightPipes numba kernels require numba when called via the "
                    "gufunc path; install numba or use the numpy code path."
                )

            caller.__name__ = getattr(fn, "__name__", "gufunc")
            caller._py_func = fn
            return caller

        return wrap

    def get_num_threads():
        return 1

    def set_num_threads(n):
        return None
