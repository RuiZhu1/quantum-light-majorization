"""Exact analytic optical state and channel response models for Stages 3 and 4.

Provides closed-form, Fock-cutoff-free no-click probabilities q_+(a) = q_-(a)
and pre-displacement mean photon numbers <n> for all parity-symmetric optical
state families studied in the JPB manuscript.  Stage 4 adds complex-displacement
variants q(alpha) that support per-setting alpha, eta, nu (including LO phase
mismatch alpha_- = -a_- e^{i delta_phi}) while leaving every Stage 3 real-axis
API and numerical result unchanged.

  1. Squeezed vacuum (r > 0 antisqueezed displacement, r < 0 squeezed displacement)
     and squeezed thermal light (n_th >= 0) under pre-displacement pure loss T,
     post-displacement detector efficiency eta, and Poisson dark counts nu.
  2. General phase-rotated odd coherent-state superpositions |C_-(c * exp(i*phi))>
     with independent displacement a, cat amplitude c, and orientation phi
     (phi = 0 recovers Eq. (21) of quantum_light_technical_note.tex; phi = pi/2
     places the cat peaks orthogonal to the displacement axis; c -> 0+ recovers
     Fock |1> smoothly).
  3. Even coherent-state superpositions |C_+(c * exp(i*phi))> (attaining the
     ideal quantum off_off ceiling 1 + exp(-2*a^2) at c = a, phi = 0).
  4. Non-Gaussian 3-level superposition family |psi_theta> = cos(theta)|0> + sin(theta)|2>
     (containing the 2b^2 <= 1 triple-baseline state at theta = -pi/3 and the exact
     3-level dark state |chi_a> at theta = -atan(sqrt(2)/a^2)).

All states above are parity-invariant (commuting with (-1)^n_hat), hence their
post-loss density matrices yield strictly symmetric responses q_+(a) = q_-(-a) = q(a).
Every closed-form formula is independently cross-checked against Fock-space
compression and Kraus-channel evolution in code/tests/test_stage3_scans.py.
"""

from __future__ import annotations

import math


def _validate_channel(
    a: float,
    eta: float,
    source_transmission: float,
    dark_mean: float,
    allow_zero_a: bool = True,
) -> None:
    if allow_zero_a:
        if a < 0.0:
            raise ValueError(f"Displacement magnitude a must be >= 0, got {a}")
    else:
        if a <= 0.0:
            raise ValueError(f"Displacement magnitude a must be > 0, got {a}")
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"Detector efficiency eta must lie in [0, 1], got {eta}")
    if not (0.0 <= source_transmission <= 1.0):
        raise ValueError(
            f"Source transmission T must lie in [0, 1], got {source_transmission}"
        )
    if dark_mean < 0.0:
        raise ValueError(f"Dark-count mean nu must be >= 0, got {dark_mean}")


def _validate_channel_complex(
    alpha: complex,
    eta: float,
    source_transmission: float,
    dark_mean: float,
) -> complex:
    alpha = complex(alpha)
    if not math.isfinite(alpha.real) or not math.isfinite(alpha.imag):
        raise ValueError(f"Displacement alpha must be a finite complex number, got {alpha}")
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"Detector efficiency eta must lie in [0, 1], got {eta}")
    if not (0.0 <= source_transmission <= 1.0):
        raise ValueError(
            f"Source transmission T must lie in [0, 1], got {source_transmission}"
        )
    if dark_mean < 0.0:
        raise ValueError(f"Dark-count mean nu must be >= 0, got {dark_mean}")
    return alpha


# ---------------------------------------------------------------------------
# 1. Squeezed vacuum & squeezed thermal light
# ---------------------------------------------------------------------------

def squeezed_thermal_mean_photons(
    r: float,
    n_th: float = 0.0,
    source_transmission: float = 1.0,
) -> float:
    """Mean photon number before displacement: T * [(n_th + 1/2)*cosh(2r) - 1/2]."""
    if n_th < 0.0:
        raise ValueError(f"Thermal occupation n_th must be >= 0, got {n_th}")
    if not (0.0 <= source_transmission <= 1.0):
        raise ValueError(f"Source transmission T must lie in [0, 1], got {source_transmission}")
    # Numerically accurate form avoiding cancellation near r = 0, n_th = 0:
    # (n_th + 1/2)*cosh(2r) - 1/2 = n_th * cosh(2r) + sinh(r)^2
    return source_transmission * (n_th * math.cosh(2.0 * r) + math.sinh(r) ** 2)


def squeezed_thermal_is_nonclassical(r: float, n_th: float = 0.0) -> bool:
    """Exact source Glauber-Sudarshan P-nonclassicality: (2*n_th + 1)*exp(-2|r|) < 1.

    Preserved under any nonzero pre-displacement pure loss T in (0, 1]; at T = 0
    the output reduces to vacuum (classical).
    """
    if n_th < 0.0:
        raise ValueError(f"Thermal occupation n_th must be >= 0, got {n_th}")
    return (2.0 * n_th + 1.0) * math.exp(-2.0 * abs(r)) < 1.0


def squeezed_quadrature_variance_factor(r: float, n_th: float = 0.0) -> float:
    """Ratio Var(X) / (1/2) along the displaced quadrature X: (2*n_th + 1)*exp(2r).

    Values < 1 mean the displaced quadrature is squeezed below shot noise (r < 0
    with (2*n_th + 1)*exp(2r) < 1), which yields u < 1 and guarantees W_2(a) > 0
    at sufficiently large a for EVERY eta > 0, T > 0, nu >= 0, and finite energy_cap.
    """
    if n_th < 0.0:
        raise ValueError(f"Thermal occupation n_th must be >= 0, got {n_th}")
    return (2.0 * n_th + 1.0) * math.exp(2.0 * r)


def squeezed_thermal_off(
    a: float,
    r: float,
    eta: float,
    n_th: float = 0.0,
    source_transmission: float = 1.0,
    dark_mean: float = 0.0,
) -> float:
    """Exact no-click probability q_+ = q_- for squeezed thermal light.

    Convention (matching quantum_light_technical_note.tex Sec. 5):
      - r > 0: real displacement +/-a lies along the ANTISQUEEZED quadrature X.
      - r < 0: real displacement +/-a lies along the SQUEEZED quadrature X.
    Pre-displacement covariance is diag((2*n_th+1)*e^{2r}, (2*n_th+1)*e^{-2r})/2
    attenuated by source transmission T, then displaced by +/-a, then detected
    with efficiency eta and dark counts nu.
    """
    _validate_channel(a, eta, source_transmission, dark_mean, allow_zero_a=True)
    if n_th < 0.0:
        raise ValueError(f"Thermal occupation n_th must be >= 0, got {n_th}")
    factor = 2.0 * n_th + 1.0
    et = eta * source_transmission
    u = 1.0 + 0.5 * et * (factor * math.exp(2.0 * r) - 1.0)
    w = 1.0 + 0.5 * et * (factor * math.exp(-2.0 * r) - 1.0)
    return math.exp(-dark_mean - eta * a * a / u) / math.sqrt(u * w)


def squeezed_thermal_off_complex(
    alpha: complex,
    r: float,
    eta: float,
    n_th: float = 0.0,
    source_transmission: float = 1.0,
    dark_mean: float = 0.0,
) -> float:
    """Exact no-click probability for squeezed thermal light under complex displacement.

    For alpha = x + i*y the Gaussian Q-function exponent splits between the two
    quadratures: the r > 0 (antisqueezed-X) covariance factors are
      u = 1 + eta*T/2 * [(2*n_th+1)*e^{2r} - 1]
      w = 1 + eta*T/2 * [(2*n_th+1)*e^{-2r} - 1]
    and q(alpha) = kappa / sqrt(u*w) * exp[-eta*(x^2/u + y^2/w)].

    Setting y = 0 and x = a recovers squeezed_thermal_off exactly.
    """
    alpha = _validate_channel_complex(alpha, eta, source_transmission, dark_mean)
    if n_th < 0.0:
        raise ValueError(f"Thermal occupation n_th must be >= 0, got {n_th}")
    factor = 2.0 * n_th + 1.0
    et = eta * source_transmission
    u = 1.0 + 0.5 * et * (factor * math.exp(2.0 * r) - 1.0)
    w = 1.0 + 0.5 * et * (factor * math.exp(-2.0 * r) - 1.0)
    x = alpha.real
    y = alpha.imag
    return math.exp(-dark_mean - eta * (x * x / u + y * y / w)) / math.sqrt(u * w)


# ---------------------------------------------------------------------------
# 2. Fock |1> and general phase-rotated odd/even cat states
# ---------------------------------------------------------------------------

def fock1_off(
    a: float,
    eta: float,
    source_transmission: float = 1.0,
    dark_mean: float = 0.0,
) -> float:
    """Exact no-click probability for single-photon Fock state |1> under loss T."""
    _validate_channel(a, eta, source_transmission, dark_mean, allow_zero_a=True)
    return math.exp(-dark_mean - eta * a * a) * (
        1.0 - eta * source_transmission + (eta * eta * source_transmission) * a * a
    )


def fock1_off_complex(
    alpha: complex,
    eta: float,
    source_transmission: float = 1.0,
    dark_mean: float = 0.0,
) -> float:
    """Exact no-click probability for single-photon Fock state |1> under complex displacement.

    After source transmission T the state is (1-T)|0><0| + T|1><1|; the displaced
    no-click probability is q(alpha) = kappa * exp(-eta*|alpha|^2) *
    [1 - eta*T + eta^2*T*|alpha|^2].
    """
    alpha = _validate_channel_complex(alpha, eta, source_transmission, dark_mean)
    abs2 = alpha.real * alpha.real + alpha.imag * alpha.imag
    return math.exp(-dark_mean - eta * abs2) * (
        1.0 - eta * source_transmission + (eta * eta * source_transmission) * abs2
    )


def odd_cat_mean_photons(c: float, source_transmission: float = 1.0) -> float:
    """Mean photon number before displacement for |C_-(c*e^{i*phi})>: T * c^2 * coth(c^2).

    Smoothly returns T (= 1 * T for Fock |1>) as c -> 0+.
    """
    if c < 0.0:
        raise ValueError(f"Cat amplitude c must be >= 0, got {c}")
    if not (0.0 <= source_transmission <= 1.0):
        raise ValueError(f"Source transmission T must lie in [0, 1], got {source_transmission}")
    c2 = c * c
    if c2 < 1e-6:
        # c^2 * coth(c^2) = 1 + c^4/3 - c^8/45 + ...
        return source_transmission * (1.0 + (c2 * c2) / 3.0)
    return source_transmission * (c2 / math.tanh(c2))


def odd_cat_general_off(
    a: float,
    c: float,
    eta: float,
    phi: float = 0.0,
    source_transmission: float = 1.0,
    dark_mean: float = 0.0,
) -> float:
    """Exact no-click probability q_+ = q_- for odd cat |C_-(c * exp(i*phi))>.

    - phi = 0.0     : real odd cat (|c> - |-c>)/N, Eq. (21) of technical note.
    - phi = pi / 2  : imaginary-axis (orthogonally displaced) odd cat (|ic> - |-ic>)/N.
    - c -> 0+       : smoothly converges to Fock |1> independent of phi.
    """
    _validate_channel(a, eta, source_transmission, dark_mean, allow_zero_a=True)
    if c < 0.0:
        raise ValueError(f"Cat amplitude c must be >= 0, got {c}")
    if c < 1e-12:
        # Exact Fock |1> limit avoiding 0/0 underflow
        return fock1_off(a, eta, source_transmission, dark_mean)

    d = math.sqrt(source_transmission) * c
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    c2 = c * c

    if c <= 1.0 and eta * a * d <= 30.0:
        # Cancellation-free nonnegative-sum identity for small/moderate c:
        #   q = kappa * exp(-eta*a^2) * [ -exp(-eta*d^2)*expm1(-2*c^2*(1-eta*T))
        #       + 2*exp(-eta*d^2)*sinh^2(eta*a*d*cos_phi)
        #       + 2*exp(-(2-eta*T)*c^2)*sin^2(eta*a*d*sin_phi) ] / (-expm1(-2*c^2))
        et = eta * source_transmission
        e_md2 = math.exp(-et * c2)
        t1 = -e_md2 * math.expm1(-2.0 * c2 * (1.0 - et))
        t2 = 2.0 * e_md2 * (math.sinh(eta * a * d * cos_phi) ** 2)
        t3 = 2.0 * math.exp(-(2.0 - et) * c2) * (math.sin(eta * a * d * sin_phi) ** 2)
        denom = -math.expm1(-2.0 * c2)
        val = math.exp(-dark_mean - eta * a * a) * (t1 + t2 + t3) / denom
        return max(0.0, min(1.0, val))

    # For c > 1.0: denom >= 1.72 (no cancellation), and combined exponents never overflow!
    dist_p_sq = (a + d * cos_phi) ** 2 + (d * sin_phi) ** 2
    dist_m_sq = (a - d * cos_phi) ** 2 + (d * sin_phi) ** 2
    term_p = math.exp(-eta * dist_p_sq)
    term_m = math.exp(-eta * dist_m_sq)
    term_x = (
        -2.0
        * math.exp(-eta * a * a - (2.0 - eta * source_transmission) * c2)
        * math.cos(2.0 * eta * a * d * sin_phi)
    )
    denom = -2.0 * math.expm1(-2.0 * c2)
    val = math.exp(-dark_mean) * (term_p + term_m + term_x) / denom
    return max(0.0, min(1.0, val))


def even_cat_mean_photons(c: float, source_transmission: float = 1.0) -> float:
    """Mean photon number before displacement for |C_+(c*e^{i*phi})>: T * c^2 * tanh(c^2)."""
    if c < 0.0:
        raise ValueError(f"Cat amplitude c must be >= 0, got {c}")
    if not (0.0 <= source_transmission <= 1.0):
        raise ValueError(f"Source transmission T must lie in [0, 1], got {source_transmission}")
    c2 = c * c
    return source_transmission * c2 * math.tanh(c2)


def even_cat_general_off(
    a: float,
    c: float,
    eta: float,
    phi: float = 0.0,
    source_transmission: float = 1.0,
    dark_mean: float = 0.0,
) -> float:
    """Exact no-click probability q_+ = q_- for even cat |C_+(c * exp(i*phi))>."""
    _validate_channel(a, eta, source_transmission, dark_mean, allow_zero_a=True)
    if c < 0.0:
        raise ValueError(f"Cat amplitude c must be >= 0, got {c}")
    d = math.sqrt(source_transmission) * c
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    c2 = c * c
    dist_p_sq = (a + d * cos_phi) ** 2 + (d * sin_phi) ** 2
    dist_m_sq = (a - d * cos_phi) ** 2 + (d * sin_phi) ** 2
    term_p = math.exp(-eta * dist_p_sq)
    term_m = math.exp(-eta * dist_m_sq)
    term_x = (
        2.0
        * math.exp(-eta * a * a - (2.0 - eta * source_transmission) * c2)
        * math.cos(2.0 * eta * a * d * sin_phi)
    )
    denom = 2.0 * (1.0 + math.exp(-2.0 * c2))
    val = math.exp(-dark_mean) * (term_p + term_m + term_x) / denom
    return max(0.0, min(1.0, val))


def odd_cat_general_off_complex(
    alpha: complex,
    c: float,
    eta: float,
    phi: float = 0.0,
    source_transmission: float = 1.0,
    dark_mean: float = 0.0,
) -> float:
    """Exact no-click probability for odd cat |C_-(c*e^{i*phi})> under complex displacement.

    The detector displacement alpha = x + i*y enters the two coherent peaks
    alpha +/- d*e^{i*phi} (d = sqrt(T)*c) and the off-diagonal phase
    2*eta*Im[(d*e^{i*phi})^* * alpha].  For real alpha = a this reduces exactly
    to odd_cat_general_off.
    """
    alpha = _validate_channel_complex(alpha, eta, source_transmission, dark_mean)
    if c < 0.0:
        raise ValueError(f"Cat amplitude c must be >= 0, got {c}")
    if c < 1e-12:
        return fock1_off_complex(alpha, eta, source_transmission, dark_mean)

    d = math.sqrt(source_transmission) * c
    c2 = c * c
    x = alpha.real
    y = alpha.imag
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    # Cancellation-free form for small c; safe threshold mirrors the real-axis
    # branch in odd_cat_general_off, with |alpha| replacing the real magnitude a.
    if c <= 1.0 and eta * d * abs(alpha) <= 30.0:
        et = eta * source_transmission
        e_md2 = math.exp(-et * c2)
        t1 = -e_md2 * math.expm1(-2.0 * c2 * (1.0 - et))
        proj_long = x * cos_phi + y * sin_phi
        proj_orth = y * cos_phi - x * sin_phi
        t2 = 2.0 * e_md2 * (math.sinh(eta * d * proj_long) ** 2)
        t3 = 2.0 * math.exp(-(2.0 - et) * c2) * (math.sin(eta * d * proj_orth) ** 2)
        denom = -math.expm1(-2.0 * c2)
        val = math.exp(-dark_mean - eta * (x * x + y * y)) * (t1 + t2 + t3) / denom
        return max(0.0, min(1.0, val))

    gamma = d * complex(cos_phi, sin_phi)
    term_p = math.exp(-eta * abs(alpha + gamma) ** 2)
    term_m = math.exp(-eta * abs(alpha - gamma) ** 2)
    phase = 2.0 * eta * (gamma.imag * x - gamma.real * y)
    term_x = (
        -2.0
        * math.exp(-eta * (x * x + y * y) - (2.0 - eta * source_transmission) * c2)
        * math.cos(phase)
    )
    denom = -2.0 * math.expm1(-2.0 * c2)
    val = math.exp(-dark_mean) * (term_p + term_m + term_x) / denom
    return max(0.0, min(1.0, val))


def even_cat_general_off_complex(
    alpha: complex,
    c: float,
    eta: float,
    phi: float = 0.0,
    source_transmission: float = 1.0,
    dark_mean: float = 0.0,
) -> float:
    """Exact no-click probability for even cat |C_+(c*e^{i*phi})> under complex displacement."""
    alpha = _validate_channel_complex(alpha, eta, source_transmission, dark_mean)
    if c < 0.0:
        raise ValueError(f"Cat amplitude c must be >= 0, got {c}")
    d = math.sqrt(source_transmission) * c
    c2 = c * c
    x = alpha.real
    y = alpha.imag
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    gamma = d * complex(cos_phi, sin_phi)
    term_p = math.exp(-eta * abs(alpha + gamma) ** 2)
    term_m = math.exp(-eta * abs(alpha - gamma) ** 2)
    phase = 2.0 * eta * (gamma.imag * x - gamma.real * y)
    term_x = (
        2.0
        * math.exp(-eta * (x * x + y * y) - (2.0 - eta * source_transmission) * c2)
        * math.cos(phase)
    )
    denom = 2.0 * (1.0 + math.exp(-2.0 * c2))
    val = math.exp(-dark_mean) * (term_p + term_m + term_x) / denom
    return max(0.0, min(1.0, val))


# ---------------------------------------------------------------------------
# 3. Non-Gaussian 3-level superposition family cos(theta)|0> + sin(theta)|2>
# ---------------------------------------------------------------------------

def zero_two_mean_photons(theta: float, source_transmission: float = 1.0) -> float:
    """Exact mean photon number of L_T(|psi_theta><psi_theta|): 2 * T * sin^2(theta)."""
    if not (0.0 <= source_transmission <= 1.0):
        raise ValueError(f"Source transmission T must lie in [0, 1], got {source_transmission}")
    return 2.0 * source_transmission * (math.sin(theta) ** 2)


def zero_two_off(
    a: float,
    theta: float,
    eta: float,
    source_transmission: float = 1.0,
    dark_mean: float = 0.0,
) -> float:
    """Exact no-click probability q_+ = q_- for |psi_theta> = cos(theta)|0> + sin(theta)|2>.

    Under pre-displacement pure loss T, |psi_theta><psi_theta| evolves exactly
    within the 3-dimensional subspace {|0>, |1>, |2>} (zero truncation error):
        rho_T = [c^2 + (1-T)^2 s^2] |0><0| + 2 T (1-T) s^2 |1><1|
                + T^2 s^2 |2><2| + T c s (|0><2| + |2><0|).
    Special points:
      - theta = -pi/3 (-60 deg): (1/2)|0> - (sqrt(3)/2)|2> (2b^2 <= 1 triple-baseline state)
      - theta = -atan(sqrt(2)/a^2): exact 3-level dark state |chi_a> (q = 0 at eta=T=1, nu=0)
    """
    _validate_channel(a, eta, source_transmission, dark_mean, allow_zero_a=True)
    c = math.cos(theta)
    s = math.sin(theta)
    T = source_transmission
    p0 = c * c + ((1.0 - T) ** 2) * (s * s)
    p1 = 2.0 * T * (1.0 - T) * (s * s)
    p2 = (T * s) ** 2
    coh = T * c * s

    ea2 = eta * eta * a * a
    e00 = 1.0
    e11 = (1.0 - eta) + ea2
    e22 = (1.0 - eta) ** 2 + 2.0 * (1.0 - eta) * ea2 + 0.5 * (ea2 * ea2)
    e02 = ea2 / math.sqrt(2.0)

    val = math.exp(-dark_mean - eta * a * a) * (
        p0 * e00 + p1 * e11 + p2 * e22 + 2.0 * coh * e02
    )
    return max(0.0, min(1.0, val))


def zero_two_off_complex(
    alpha: complex,
    theta: float,
    eta: float,
    source_transmission: float = 1.0,
    dark_mean: float = 0.0,
) -> float:
    """Exact no-click probability for |psi_theta> under complex displacement.

    The |0><2| coherence carries the phase of the displacement, so the cross
    term uses Re(alpha^{*2}).  For real alpha = a this reduces exactly to
    zero_two_off.
    """
    alpha = _validate_channel_complex(alpha, eta, source_transmission, dark_mean)
    c = math.cos(theta)
    s = math.sin(theta)
    T = source_transmission
    p0 = c * c + ((1.0 - T) ** 2) * (s * s)
    p1 = 2.0 * T * (1.0 - T) * (s * s)
    p2 = (T * s) ** 2
    coh = T * c * s

    abs2 = alpha.real * alpha.real + alpha.imag * alpha.imag
    ea2 = eta * eta * abs2
    e00 = 1.0
    e11 = (1.0 - eta) + ea2
    e22 = (1.0 - eta) ** 2 + 2.0 * (1.0 - eta) * ea2 + 0.5 * (ea2 * ea2)
    e02 = eta * eta * alpha.conjugate() ** 2 / math.sqrt(2.0)

    val = math.exp(-dark_mean - eta * abs2) * (
        p0 * e00 + p1 * e11 + p2 * e22 + 2.0 * coh * e02.real
    )
    return max(0.0, min(1.0, val))


def dark_state_theta(a: float) -> float:
    """Angle theta in (-pi/2, 0) giving |chi_a> = (-a^2|0> + sqrt(2)|2>)/sqrt(a^4 + 2)."""
    if a <= 0.0:
        raise ValueError(f"Displacement magnitude a must be > 0, got {a}")
    return -math.atan2(math.sqrt(2.0), a * a)
