"""Benchmark the accelerated routines against the upstream oracle.

Runs each routine in both implementations at several grid sizes and reports
wall time, speedup and the largest relative difference, so a speed number is
never reported without the accuracy that came with it.

    uv run python benchmarks/bench.py
    uv run python benchmarks/bench.py --sizes 256,512 --routines steps,unwrap
"""
import argparse
import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "_ref"))
sys.path.insert(0, str(_ROOT))

import OptimLightPipes as new  # noqa: E402
import LightPipes_ref as ref  # noqa: E402

LAM = 632.8e-9
SIZE = 5.0e-3


def _time(fn, repeat=3):
    """Best-of-`repeat` wall time, plus the last result.

    Best-of rather than mean: these routines are multi-threaded and the two
    implementations run back to back, so the distribution has a long right
    tail from scheduler contention. The minimum is the most stable estimate
    of the work actually done.
    """
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def _reldiff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    scale = np.abs(b).max()
    if scale == 0:
        return float(np.abs(a - b).max())
    return float(np.abs(a - b).max() / scale)


def _start(mod, N, dtype=np.complex128):
    return mod.CircAperture(mod.Begin(SIZE, LAM, N, dtype), 1.0e-3)


# Each entry: name -> (sizes, fn(mod, N) -> array-like)
def _bench_forward(mod, N):
    return mod.Forward(_start(mod, N), 0.2, 2 * SIZE, N // 2).field


def _bench_steps_ram(mod, N):
    return mod.Steps(_start(mod, N), 0.01, 5, 1.0, True).field


def _bench_steps_array(mod, N):
    return mod.Steps(_start(mod, N), 0.01, 5, 1.0, False).field


_UNWRAP_INPUT = {}


def _bench_unwrap(mod, N):
    # Build the wrapped phase once per size, outside the timed region, so this
    # measures the unwrap itself rather than Zernike+Phase setup.
    if N not in _UNWRAP_INPUT:
        F = ref.Zernike(_start(ref, N), 4, 0, 1.5e-3, 8 * LAM)
        _UNWRAP_INPUT[N] = ref.Phase(F)
    return mod.PhaseUnwrap(_UNWRAP_INPUT[N])


def _bench_fresnel(mod, N):
    return mod.Fresnel(_start(mod, N), 0.3).field


def _bench_forvard(mod, N):
    return mod.Forvard(_start(mod, N), 0.3).field


def _bench_interpol(mod, N):
    return mod.Interpol(_start(mod, N), SIZE, N, angle=15.0, magnif=0.8).field


_ZERNIKE_FIELD = {}


def _bench_zernike_fit(mod, N):
    # Exercises the njit unwrap plus the cached coordinate grids, which are
    # re-read once per Zernike term.
    key = (id(mod), N)
    if key not in _ZERNIKE_FIELD:
        _ZERNIKE_FIELD[key] = mod.Zernike(
            mod.CircAperture(mod.Begin(SIZE, LAM, N), 1.5e-3),
            4, 0, 1.5e-3, 3 * LAM)
    return np.asarray(mod.ZernikeFit(_ZERNIKE_FIELD[key], 20, 1.5e-3)[1])


ROUTINES = {
    # Forward is O(N^3) here but O(N^4) upstream: keep the oracle sizes small.
    "forward": ([32, 64, 96, 128], _bench_forward),
    "steps_save_ram": ([128, 256, 512], _bench_steps_ram),
    "steps_array": ([128, 256, 512], _bench_steps_array),
    "unwrap": ([128, 256, 512], _bench_unwrap),
    "fresnel": ([256, 512, 1024], _bench_fresnel),
    "forvard": ([256, 512, 1024], _bench_forvard),
    "interpol": ([256, 512, 1024], _bench_interpol),
    "zernike_fit": ([128, 256, 512], _bench_zernike_fit),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routines", default=",".join(ROUTINES))
    ap.add_argument("--sizes", default="", help="override grid sizes, e.g. 256,512")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    override = [int(s) for s in args.sizes.split(",") if s] or None

    print(f"numba={new.HAVE_NUMBA}  threads={new.get_num_threads()}  "
          f"fft={new.get_fft_backend()} x{new.get_fft_threads()}")
    print("(upstream column uses its own defaults: numpy.fft, no plan cache)")
    print("warming up kernels...")
    new.warmup()
    print()
    print(f"{'routine':<16}{'N':>6}{'upstream':>11}{'optimized':>11}"
          f"{'speedup':>10}{'rel.diff':>11}")
    print("-" * 65)

    for name in args.routines.split(","):
        name = name.strip()
        if name not in ROUTINES:
            print(f"unknown routine: {name}", file=sys.stderr)
            continue
        sizes, fn = ROUTINES[name]
        for N in override or sizes:
            # Warm both sides: numba compiles on first call, and pyFFTW builds
            # (and caches) an FFT plan per shape. Timing either one-off cost
            # would measure setup rather than the steady-state routine.
            fn(new, N)
            fn(ref, N)
            t_new, r_new = _time(lambda: fn(new, N), args.repeat)
            t_ref, r_ref = _time(lambda: fn(ref, N), args.repeat)
            print(f"{name:<16}{N:>6}{t_ref:>10.4f}s{t_new:>10.4f}s"
                  f"{t_ref / t_new:>9.1f}x{_reldiff(r_new, r_ref):>11.1e}")


if __name__ == "__main__":
    main()
