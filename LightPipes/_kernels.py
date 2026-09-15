"""Compiled kernels for the LightPipes hot paths.

Parallelism is expressed as a gufunc *broadcast axis*: each kernel's core
signature describes the work for one row / column / output pixel, and numba's
target='parallel' runs the independent instances across threads. That fits this
library because nearly every hot spot is "a sequential scalar computation over
one row, repeated N independent times".

Kernels must live in a real source file for cache=True to work (numba needs a
source locator), so nothing here is defined dynamically.
"""
import numpy as np

from ._numba_compat import guvectorize, njit

# ---------------------------------------------------------------------------
# Fresnel integrals S(x), C(x)
# ---------------------------------------------------------------------------
# scipy.special.fresnel cannot be called from nopython code, so this is a port
# of the Cephes fresnl.c rational approximation - the same algorithm scipy
# itself uses. Validated to 3.3e-16 absolute against a 60-digit power series
# over [-5, 5], with correct -> 0.5 asymptotics at x = 10, 50, 1e3, 1e5.

_SN = np.array([
    -2.99181919401019853726e3, 7.08840045257738576863e5,
    -6.29741486205862506537e7, 2.54890880573376359104e9,
    -4.42979518059697779103e10, 3.18016297876567817986e11])
_SD = np.array([
    2.81376268889994315696e2, 4.55847810806532581675e4,
    5.17343888770096400730e6, 4.19320245898111231129e8,
    2.24411795645340920940e10, 6.07366389490084639049e11])
_CN = np.array([
    -4.98843114573573548651e-8, 9.50428062829859605134e-6,
    -6.45191435683965050962e-4, 1.88843319396703850064e-2,
    -2.05525900955013891793e-1, 9.99999999999999998822e-1])
_CD = np.array([
    3.99982968972495980367e-12, 9.15439215774657478799e-10,
    1.25001862479598821474e-7, 1.22262789024179030997e-5,
    8.68029542941784300606e-4, 4.12142090722199792936e-2,
    1.00000000000000000118e0])
_FN = np.array([
    4.21543555043677546506e-1, 1.43407919780758885261e-1,
    1.15220955073585758835e-2, 3.45017939782574027900e-4,
    4.63613749287867322088e-6, 3.05568983790257605827e-8,
    1.02304514164907233465e-10, 1.72010743268161828879e-13,
    1.34283276233062758925e-16, 3.76329711269987889006e-20])
_FD = np.array([
    7.51586398353378947175e-1, 1.16888925859191382142e-1,
    6.44051526508858611005e-3, 1.55934409164153020873e-4,
    1.84627567348930545870e-6, 1.12699224763999035261e-8,
    3.60140029589371370404e-11, 5.88754533621578410010e-14,
    4.52001434074129701496e-17, 1.25443237090011264384e-20])
_GN = np.array([
    5.04442073643383265887e-1, 1.97102833525523411709e-1,
    1.87648584092575249293e-2, 6.84079380915393090172e-4,
    1.15138826111884280931e-5, 9.82852443688422223854e-8,
    4.45344415861750144738e-10, 1.08268041139020870318e-12,
    1.37555460633261799868e-15, 8.36354435630677421531e-19,
    1.86958710162783235106e-22])
_GD = np.array([
    1.47495759925128324529e0, 3.37748989120019970451e-1,
    2.53603741420338795122e-2, 8.14679107184306179049e-4,
    1.27545075667729118702e-5, 1.04314589657571990585e-7,
    4.60680728146520428211e-10, 1.10273215066240270757e-12,
    1.38796531259578871258e-15, 8.39158816283118707363e-19,
    1.86958710162783236342e-22])


@njit(inline='always')
def _polevl(x, coef, N):
    """Evaluate a polynomial with a leading coefficient (Cephes polevl)."""
    ans = coef[0]
    for i in range(1, N + 1):
        ans = ans * x + coef[i]
    return ans


@njit(inline='always')
def _p1evl(x, coef, N):
    """Evaluate a monic polynomial (Cephes p1evl: leading coefficient 1)."""
    ans = x + coef[0]
    for i in range(1, N):
        ans = ans * x + coef[i]
    return ans


@njit(inline='always')
def fresnel_scalar(xxa):
    """Return (S(x), C(x)), the Fresnel sine and cosine integrals."""
    x = abs(xxa)
    x2 = x * x
    if x2 < 2.5625:
        t = x2 * x2
        ss = x * x2 * _polevl(t, _SN, 5) / _p1evl(t, _SD, 6)
        cc = x * _polevl(t, _CN, 5) / _polevl(t, _CD, 6)
    elif x > 36974.0:
        # Beyond this the oscillation is unresolvable; both tend to 1/2.
        ss = 0.5
        cc = 0.5
    else:
        t = np.pi * x2
        u = 1.0 / (t * t)
        t = 1.0 / t
        f = 1.0 - u * _polevl(u, _FN, 9) / _p1evl(u, _FD, 10)
        g = t * _polevl(u, _GN, 10) / _p1evl(u, _GD, 11)
        t = np.pi * 0.5 * x2
        c = np.cos(t)
        s = np.sin(t)
        t = np.pi * x
        cc = 0.5 + (f * s - g * c) / t
        ss = 0.5 - (f * c + g * s) / t
    if xxa < 0.0:
        cc = -cc
        ss = -ss
    return ss, cc


@guvectorize(['void(float64[:], complex128[:])'], '()->()')
def fresnel_A(x, out):
    """A(x) = C(x) + 1j*S(x), elementwise over the broadcast axis."""
    s, c = fresnel_scalar(x[0])
    out[0] = complex(c, s)


# ---------------------------------------------------------------------------
# Forward(): direct-integration propagator
# ---------------------------------------------------------------------------
# Upstream builds 17 N x N outer-product temporaries per output pixel and sums
# them (O(N^4) work, O(N^2) allocation per pixel). That whole expression is
# algebraically equal to, with A_k = Fc_k + 1j*Fs_k:
#
#     sum_ij field[j,i] * (A4[j] - A2[j]) * (A3[i] - A1[i]) * 0.5j
#
# The kernel is rank-1, so the double sum factorizes into two nested single
# sums and the routine becomes O(N^3):
#
#     out[j_new, i_new] = 0.5j * sum_j dA_j[j_new,j] * (sum_i field[j,i] * dA_i[i_new,i])
#
# Verified against upstream to ~4e-15 relative.

@guvectorize(
    ['void(complex128[:,:], complex128[:], complex128[:,:], complex128[:])',
     'void(complex64[:,:], complex64[:], complex64[:,:], complex64[:])'],
    '(m,n),(n),(p,m)->(p)')
def _forward_col(field, dA_i, dA_j, out):
    """One column of the output grid (fixed i_new); broadcast over i_new.

    field : (m, n) input field
    dA_i  : (n,)   A3-A1 for this i_new, over old x
    dA_j  : (p, m) A4-A2 for every j_new, over old y
    out   : (p,)   the output column
    """
    m = field.shape[0]
    n = field.shape[1]
    p = dA_j.shape[0]

    # Contract over the old x axis once: G[j] = sum_i field[j,i]*dA_i[i]
    G = np.zeros(m, dtype=out.dtype)
    for j in range(m):
        acc = 0.0 + 0.0j
        for i in range(n):
            acc += field[j, i] * dA_i[i]
        G[j] = acc

    # Then contract over the old y axis for each new j.
    for jn in range(p):
        acc = 0.0 + 0.0j
        for j in range(m):
            acc += dA_j[jn, j] * G[j]
        out[jn] = 0.5j * acc


# ---------------------------------------------------------------------------
# Tridiagonal double-sweep elimination (Crank-Nicolson BPM, Steps())
# ---------------------------------------------------------------------------
# The forward/backward substitution is sequential along the sweep axis, but the
# N systems (one per row, or per column) are completely independent. Expressing
# one system as the gufunc core '(n),(n),(),()->(n)' puts the recurrence inside
# the core and the N independent systems on the broadcast axis, which
# target='parallel' then spreads across threads.
#
# gufuncs consume non-contiguous inputs directly, so the column-direction sweep
# passes transposed views with no copy.

@guvectorize(
    ['void(complex128[:], complex128[:], complex128, complex128, complex128[:])',
     'void(complex64[:], complex64[:], complex64, complex64, complex64[:])'],
    '(n),(n),(),()->(n)')
def _elim_row(c, p, a, b, uu):
    """Solve one tridiagonal system by double sweep.

    Mirrors subs.elim exactly, including the zeroed edges and the extra
    post-loop step for index N-1.
    """
    N = c.shape[0]
    alpha = np.zeros(N, dtype=uu.dtype)
    beta = np.zeros(N, dtype=uu.dtype)

    # Edge amplitudes are zero.
    alpha[0] = 0.0
    beta[0] = 0.0
    alpha[N - 2] = 0.0
    beta[N - 2] = 0.0

    # Forward elimination.
    for i in range(1, N - 2):
        cc = c[i] - a * alpha[i - 1]
        alpha[i] = b / cc
        beta[i] = (p[i] + a * beta[i - 1]) / cc

    cc = c[N - 1] - a * alpha[N - 2]
    beta[N - 1] = (p[N - 1] + a * beta[N - 2]) / cc

    uu[N - 1] = beta[N - 1]

    # Backward substitution.
    for i in range(N - 2, -1, -1):
        uu[i] = alpha[i] * uu[i + 1] + beta[i]


@guvectorize(
    ['void(complex128[:], complex128[:], complex128[:], complex128[:], complex128,'
     ' complex128, float64, complex128, complex128[:])',
     'void(complex64[:], complex64[:], complex64[:], complex64[:], complex64,'
     ' complex64, float64, complex64, complex64[:])'],
    '(n),(n),(n),(n),(),(),(),()->(n)')
def _steps_line(u_prev, u_mid, u_next, c, a, b, delta2, imPi4lz, uu):
    """Build the RHS stencil for one line and solve it, in a single pass.

    u_prev / u_mid / u_next are the three neighbouring lines of the field.
    Fuses what upstream did as a numpy stencil plus a separate elim() call.
    """
    N = c.shape[0]
    p = np.zeros(N, dtype=uu.dtype)
    inv = -1.0 / delta2
    for k in range(1, N - 1):
        p[k] = inv * (u_prev[k] + u_next[k] - 2.0 * u_mid[k]) + imPi4lz * u_mid[k]

    alpha = np.zeros(N, dtype=uu.dtype)
    beta = np.zeros(N, dtype=uu.dtype)
    alpha[0] = 0.0
    beta[0] = 0.0
    alpha[N - 2] = 0.0
    beta[N - 2] = 0.0

    for i in range(1, N - 2):
        cc = c[i] - a * alpha[i - 1]
        alpha[i] = b / cc
        beta[i] = (p[i] + a * beta[i - 1]) / cc

    cc = c[N - 1] - a * alpha[N - 2]
    beta[N - 1] = (p[N - 1] + a * beta[N - 2]) / cc

    uu[N - 1] = beta[N - 1]
    for i in range(N - 2, -1, -1):
        uu[i] = alpha[i] * uu[i + 1] + beta[i]


def steps_sweep_rows(field, CCX, a, b, delta2, imPi4lz):
    """Solve every interior row j in [1, N-1) at once.

    Safe to batch: upstream writes row j-1 only after reading rows j-1, j, j+1,
    so no elim ever observes an already-updated row. Returns shape (N-2, N).
    """
    dt = field.dtype
    return _elim_pack(field[0:-2], field[1:-1], field[2:], CCX[1:-1],
                      dt.type(a), dt.type(b), float(delta2), dt.type(imPi4lz))


def steps_sweep_cols(field, CCY, a, b, delta2, imPi4lz):
    """Column-direction counterpart; transposed views, no copy."""
    dt = field.dtype
    fT = field.T
    return _elim_pack(fT[0:-2], fT[1:-1], fT[2:], CCY.T[1:-1],
                      dt.type(a), dt.type(b), float(delta2), dt.type(imPi4lz))


def _elim_pack(u_prev, u_mid, u_next, c, a, b, delta2, imPi4lz):
    return _steps_line(u_prev, u_mid, u_next, c, a, b, delta2, imPi4lz)


def elim_rows(N, a, b, c, p, UU):
    """Row-wise (horizontal) double sweep; writes into UU."""
    dt = UU.dtype
    UU[:, :] = _elim_row(c, p, dt.type(a), dt.type(b))


def elim_cols(N, a, b, c, p, UU):
    """Column-wise (vertical) double sweep; writes into UU.

    Transposed views go straight into the gufunc - no contiguous copy needed.
    """
    dt = UU.dtype
    UU[:, :] = _elim_row(c.T, p.T, dt.type(a), dt.type(b)).T


def forward_kernel(field_in, X_old, X_new, Y_old, Y_new, dx_old, R22):
    """Factorized Forward(). Returns the new field, shape (len(Y_new), len(X_new))."""
    dtype = field_in.dtype

    # D[k, l] = old coordinate l minus new coordinate k
    Dx = X_old[None, :] - X_new[:, None]
    Dy = Y_old[None, :] - Y_new[:, None]

    # A1/A3 differ by the sign of dx_old, likewise A4/A2.
    A1 = fresnel_A(R22 * (2.0 * Dx + dx_old))
    A3 = fresnel_A(R22 * (2.0 * Dx - dx_old))
    A2 = fresnel_A(R22 * (2.0 * Dy - dx_old))
    A4 = fresnel_A(R22 * (2.0 * Dy + dx_old))

    dA_i = np.ascontiguousarray((A3 - A1).astype(dtype))
    dA_j = np.ascontiguousarray((A4 - A2).astype(dtype))

    # Broadcast axis is i_new; dA_j is shared by every column.
    field_c = np.ascontiguousarray(field_in)
    out = _forward_col(field_c, dA_i, dA_j)  # -> (n_new_x, n_new_y)
    return np.ascontiguousarray(out.T)
