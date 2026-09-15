"""FFT backend selection.

The FFT dominates the transform-based propagators: at N=1024 roughly 82% of
Fresnel's runtime is its three 2N x 2N transforms, which numba cannot
accelerate. pyFFTW is therefore the only lever on that part, and it is worth
using properly:

* `pyfftw.interfaces` re-plans on every call unless the interfaces cache is
  enabled. Upstream imports the interfaces but never enables the cache, so
  planning overhead cancels most of the benefit. We enable it once here.
* pyfftw defaults to a single thread. We default to the CPU count, which is
  where the real speedup comes from on large grids.

Falls back to numpy.fft when pyFFTW is not installed.
"""
import multiprocessing

_HAVE_PYFFTW = False
_CACHE_READY = False

try:
    import pyfftw as _pyfftw
    from pyfftw.interfaces.numpy_fft import fft2 as _pyfftw_fft2
    from pyfftw.interfaces.numpy_fft import ifft2 as _pyfftw_ifft2
    import pyfftw.interfaces.cache as _pyfftw_cache

    _HAVE_PYFFTW = True
except ImportError:  # pragma: no cover - depends on environment
    _pyfftw = None


def _cpu_count():
    try:
        return max(1, multiprocessing.cpu_count())
    except NotImplementedError:  # pragma: no cover
        return 1


# Threads used for the FFT. Kept separate from the numba thread count so the
# two pools can be tuned independently (setting both to the core count on a
# small grid oversubscribes the machine).
_FFT_THREADS = _cpu_count()

# Plans are reused via the interfaces cache; ESTIMATE keeps the first call
# cheap, which matters because grid sizes vary between calls in typical use.
_PLANNER_EFFORT = 'FFTW_ESTIMATE'

# How long an unused plan stays alive, in seconds. Long enough to survive a
# loop that alternates between a few grid sizes.
_KEEPALIVE = 60.0


# Backend selection: 'auto' uses pyFFTW whenever it imported successfully and
# falls back to numpy otherwise; 'numpy' and 'pyfftw' force one backend.
_BACKEND = 'auto'
_BACKENDS = ('auto', 'numpy', 'pyfftw')


def set_fft_backend(name):
    """*Choose the FFT backend: 'auto', 'numpy' or 'pyfftw'.*

    'auto' (the default) uses pyFFTW when it is installed and falls back to
    numpy.fft otherwise. Use 'numpy' to opt out at runtime without editing
    config.py.

    :param name: 'auto', 'numpy' or 'pyfftw'
    :type name: str
    :return: the backend that will actually be used
    :rtype: str

    >>> OptimLightPipes.set_fft_backend('numpy') # disable pyFFTW
    """
    global _BACKEND
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown FFT backend {name!r}, expected one of {_BACKENDS}")
    if name == 'pyfftw' and not _HAVE_PYFFTW:
        raise ValueError(
            "pyFFTW is not installed; install it or use 'auto'/'numpy'")
    _BACKEND = name
    return get_fft_backend()


def get_fft_backend():
    """*Return the FFT backend in use: 'numpy' or 'pyfftw'.*"""
    if _BACKEND == 'auto':
        return 'pyfftw' if _HAVE_PYFFTW else 'numpy'
    return _BACKEND


def set_fft_threads(n):
    """*Set the number of threads used for FFTs.*

    :param n: thread count; None restores the CPU count
    :type n: int, None
    """
    global _FFT_THREADS
    _FFT_THREADS = _cpu_count() if n is None else max(1, int(n))
    return _FFT_THREADS


def get_fft_threads():
    """*Return the number of threads currently used for FFTs.*"""
    return _FFT_THREADS


def have_pyfftw():
    """*True if pyFFTW is available as the FFT backend.*"""
    return _HAVE_PYFFTW


def _ensure_cache():
    """Enable the pyFFTW plan cache once; without it every call re-plans."""
    global _CACHE_READY
    if not _CACHE_READY:
        _pyfftw_cache.enable()
        _pyfftw_cache.set_keepalive_time(_KEEPALIVE)
        _CACHE_READY = True


def get_fft(usepyFFTW=False):
    """Return (fft2, ifft2, kwargs, using_pyfftw) for this call.

    pyFFTW is used when the selected backend resolves to it, or when the
    caller explicitly asked via `usepyFFTW`. Falls back to numpy.fft whenever
    pyFFTW is unavailable, so callers never need to handle the missing case.
    """
    if (usepyFFTW or get_fft_backend() == 'pyfftw') and _HAVE_PYFFTW:
        _ensure_cache()
        # overwrite_input lets FFTW skip a copy of the (large) input buffer;
        # the callers reassign the result and never reuse the input afterwards.
        kwargs = {'planner_effort': _PLANNER_EFFORT,
                  'overwrite_input': True,
                  'threads': _FFT_THREADS}
        return _pyfftw_fft2, _pyfftw_ifft2, kwargs, True

    from numpy.fft import fft2, ifft2
    return fft2, ifft2, {}, False


def empty_aligned(shape, dtype):
    """SIMD-aligned zeroed array when pyFFTW is present, else a plain one."""
    if _HAVE_PYFFTW:
        return _pyfftw.zeros_aligned(shape, dtype=dtype)
    import numpy as np
    return np.zeros(shape, dtype=dtype)
