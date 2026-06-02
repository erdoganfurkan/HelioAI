"""PlasmaPy-based plasma physics calculations exposed as agent tools.

Each function accepts plain SI-ish numbers (nT, cm⁻³, eV) and returns a dict
with value, unit, and a brief physical context — ready for LLM consumption.

SPASE ParticleQuantity / FieldQuantity mapping:
  plasma_beta        → PlasmaBeta (ActivityIndex)
  gyrofrequency      → Gyrofrequency (FieldQuantity + ParticleQuantity)
  debye_length       → (ParticleQuantity implied)
  alfven_speed       → AlfvenVelocity (ParticleQuantity)
  inertial_length    → (ParticleQuantity implied)
  power_spectrum     → Spectrum (MeasurementType)
"""

from __future__ import annotations

import math


def _parse_particle(particle: str):
    """Map user-friendly particle name to PlasmaPy particle string."""
    mapping = {
        "proton": "p+", "p": "p+", "p+": "p+", "ion": "p+",
        "electron": "e-", "e": "e-", "e-": "e-",
        "alpha": "He-4 2+", "alpha particle": "He-4 2+", "he": "He-4 2+",
    }
    return mapping.get(particle.lower().strip(), particle)


async def plasma_beta(B_nT: float, n_cm3: float, T_eV: float) -> dict:
    """Compute plasma beta — ratio of thermal pressure to magnetic pressure.

    Args:
        B_nT:  Magnetic field magnitude in nT
        n_cm3: Number density in cm⁻³
        T_eV:  Temperature in eV

    Returns dict with beta (dimensionless) and regime interpretation.
    """
    try:
        import astropy.units as u
        import plasmapy.formulary as pf

        B = B_nT * u.nT
        n = n_cm3 * u.cm ** -3
        T = T_eV * u.eV

        beta_val = float(pf.beta(T, n, B).value)

        if beta_val < 0.01:
            regime = "magnetically dominated (β ≪ 1) — typical inner magnetosphere / coronal loop"
        elif beta_val < 1.0:
            regime = "sub-Alfvénic plasma (β < 1) — typical solar wind / outer magnetosphere"
        elif beta_val < 10.0:
            regime = "high-β plasma (β ~ 1-10) — typical magnetosheath / plasma sheet"
        else:
            regime = "pressure-dominated (β ≫ 1) — typical ionosphere / dense plasma"

        return {
            "beta": round(beta_val, 6),
            "unit": "dimensionless",
            "regime": regime,
            "inputs": {"B_nT": B_nT, "n_cm3": n_cm3, "T_eV": T_eV},
        }
    except Exception as e:
        return {"error": str(e)}


async def gyrofrequency(B_nT: float, particle: str = "proton") -> dict:
    """Compute particle gyrofrequency (cyclotron frequency).

    Args:
        B_nT:    Magnetic field magnitude in nT
        particle: 'proton', 'electron', 'alpha' (default: proton)

    Returns dict with frequency in Hz and angular frequency in rad/s.
    """
    try:
        import astropy.units as u
        import numpy as np
        import plasmapy.formulary as pf

        B = B_nT * u.nT
        p = _parse_particle(particle)

        omega = pf.gyrofrequency(B, particle=p, signed=False)
        f_hz = float((omega / (2 * math.pi * u.rad)).to(u.Hz).value)
        omega_rad_s = float(omega.to(u.rad / u.s).value)

        return {
            "frequency_Hz": round(f_hz, 4),
            "angular_frequency_rad_s": round(omega_rad_s, 4),
            "particle": particle,
            "B_nT": B_nT,
            "period_s": round(1.0 / f_hz, 6) if f_hz > 0 else None,
        }
    except Exception as e:
        return {"error": str(e)}


async def debye_length(n_cm3: float, T_eV: float) -> dict:
    """Compute electron Debye length.

    Args:
        n_cm3: Electron number density in cm⁻³
        T_eV:  Electron temperature in eV

    Returns dict with Debye length in km and meters.
    """
    try:
        import astropy.units as u
        import plasmapy.formulary as pf

        n = n_cm3 * u.cm ** -3
        T = T_eV * u.eV

        lambda_D = pf.Debye_length(T, n)
        lambda_m = float(lambda_D.to(u.m).value)
        lambda_km = float(lambda_D.to(u.km).value)

        return {
            "debye_length_m": round(lambda_m, 6),
            "debye_length_km": round(lambda_km, 9),
            "inputs": {"n_cm3": n_cm3, "T_eV": T_eV},
        }
    except Exception as e:
        return {"error": str(e)}


async def alfven_speed(B_nT: float, n_cm3: float, mass_amu: float = 1.0) -> dict:
    """Compute Alfvén speed.

    Args:
        B_nT:      Magnetic field magnitude in nT
        n_cm3:     Ion number density in cm⁻³
        mass_amu:  Ion mass in atomic mass units (default 1.0 = proton)

    Returns dict with Alfvén speed in km/s.
    """
    try:
        import astropy.units as u
        import plasmapy.formulary as pf

        B = B_nT * u.nT
        n = n_cm3 * u.cm ** -3
        ion = f"H-1 1+" if mass_amu == 1.0 else f"p+"

        V_A = pf.Alfven_speed(B, n, ion=ion)
        va_km_s = float(V_A.to(u.km / u.s).value)

        return {
            "alfven_speed_km_s": round(va_km_s, 3),
            "alfven_speed_m_s": round(va_km_s * 1000, 1),
            "inputs": {"B_nT": B_nT, "n_cm3": n_cm3, "mass_amu": mass_amu},
            "note": "Typical solar wind: 40-80 km/s. Magnetosphere: 100-1000 km/s.",
        }
    except Exception as e:
        return {"error": str(e)}


async def inertial_length(n_cm3: float, particle: str = "proton") -> dict:
    """Compute ion or electron inertial length (skin depth).

    Args:
        n_cm3:    Number density in cm⁻³
        particle: 'proton' or 'electron' (default: proton)

    Returns dict with inertial length in km and meters.
    """
    try:
        import astropy.units as u
        import plasmapy.formulary as pf

        n = n_cm3 * u.cm ** -3
        p = _parse_particle(particle)

        d = pf.inertial_length(n, particle=p)
        d_km = float(d.to(u.km).value)
        d_m = float(d.to(u.m).value)

        return {
            "inertial_length_km": round(d_km, 4),
            "inertial_length_m": round(d_m, 2),
            "particle": particle,
            "inputs": {"n_cm3": n_cm3},
        }
    except Exception as e:
        return {"error": str(e)}


async def power_spectrum(
    values: list[float],
    dt_s: float,
    nperseg: int | None = None,
) -> dict:
    """Compute power spectral density using Welch's method.

    Args:
        values:  Time series as a list of floats
        dt_s:    Sampling interval in seconds
        nperseg: Samples per FFT segment (default: min(256, len(values)//4))

    Returns dict with frequencies (Hz), PSD values, peak frequency,
    and export-ready summary for LLM interpretation.
    """
    try:
        import numpy as np
        from scipy import signal

        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 8:
            return {"error": f"Need at least 8 finite samples, got {len(arr)}"}

        fs = 1.0 / dt_s
        seg = nperseg or min(256, len(arr) // 4)
        seg = max(seg, 8)

        freqs, psd = signal.welch(arr, fs=fs, nperseg=seg)

        peak_idx = int(np.argmax(psd[1:])) + 1
        peak_freq = float(freqs[peak_idx])
        peak_power = float(psd[peak_idx])

        return {
            "frequencies_Hz": [round(f, 6) for f in freqs.tolist()],
            "psd": [round(p, 8) for p in psd.tolist()],
            "peak_frequency_Hz": round(peak_freq, 6),
            "peak_period_s": round(1.0 / peak_freq, 3) if peak_freq > 0 else None,
            "peak_power": round(peak_power, 8),
            "n_points": len(arr),
            "fs_Hz": round(fs, 6),
            "freq_resolution_Hz": round(freqs[1] - freqs[0], 8) if len(freqs) > 1 else None,
        }
    except Exception as e:
        return {"error": str(e)}
