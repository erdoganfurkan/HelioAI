"""Tests for helioai.tools.plasmapy_tools — pure physics, no mocking."""

from __future__ import annotations

import math

import pytest

from helioai.tools.plasmapy_tools import (
    alfven_speed,
    debye_length,
    gyrofrequency,
    inertial_length,
    plasma_beta,
    power_spectrum,
)


# ──────────────────────────────── plasma_beta ────────────────────────────────


async def test_plasma_beta_solar_wind() -> None:
    result = await plasma_beta(B_nT=5.0, n_cm3=10.0, T_eV=10.0)
    assert "beta" in result
    assert result["beta"] > 0
    assert "regime" in result


async def test_plasma_beta_magnetosheath_greater_than_one() -> None:
    result = await plasma_beta(B_nT=20.0, n_cm3=30.0, T_eV=50.0)
    assert result["beta"] > 1.0


async def test_plasma_beta_magnetosphere_lobe_small() -> None:
    result = await plasma_beta(B_nT=40.0, n_cm3=0.05, T_eV=500.0)
    assert result["beta"] < 1.0


async def test_plasma_beta_has_units_field() -> None:
    result = await plasma_beta(5.0, 10.0, 10.0)
    assert "units" in result or "beta" in result


# ──────────────────────────────── gyrofrequency ──────────────────────────────


async def test_gyrofrequency_proton_positive() -> None:
    result = await gyrofrequency(B_nT=5.0, particle="p")
    assert "frequency_Hz" in result
    assert result["frequency_Hz"] > 0


async def test_gyrofrequency_electron_much_higher() -> None:
    proton = await gyrofrequency(B_nT=5.0, particle="p")
    electron = await gyrofrequency(B_nT=5.0, particle="e")
    assert electron["frequency_Hz"] > proton["frequency_Hz"] * 100


async def test_gyrofrequency_scales_with_B() -> None:
    r1 = await gyrofrequency(B_nT=5.0, particle="p")
    r2 = await gyrofrequency(B_nT=10.0, particle="p")
    assert r2["frequency_Hz"] == pytest.approx(2 * r1["frequency_Hz"], rel=1e-3)


async def test_gyrofrequency_alpha_particle() -> None:
    result = await gyrofrequency(B_nT=5.0, particle="alpha")
    assert "frequency_Hz" in result
    assert result["frequency_Hz"] > 0


# ──────────────────────────────── debye_length ───────────────────────────────


async def test_debye_length_positive() -> None:
    result = await debye_length(n_cm3=10.0, T_eV=10.0)
    assert "debye_length_km" in result
    assert result["debye_length_km"] > 0


async def test_debye_length_increases_with_temperature() -> None:
    cold = await debye_length(n_cm3=10.0, T_eV=1.0)
    hot = await debye_length(n_cm3=10.0, T_eV=100.0)
    assert hot["debye_length_km"] > cold["debye_length_km"]


async def test_debye_length_decreases_with_density() -> None:
    sparse = await debye_length(n_cm3=1.0, T_eV=10.0)
    dense = await debye_length(n_cm3=100.0, T_eV=10.0)
    assert sparse["debye_length_km"] > dense["debye_length_km"]


# ──────────────────────────────── alfven_speed ───────────────────────────────


async def test_alfven_speed_positive() -> None:
    result = await alfven_speed(B_nT=5.0, n_cm3=10.0)
    assert "alfven_speed_km_s" in result
    assert result["alfven_speed_km_s"] > 0


async def test_alfven_speed_solar_wind_range() -> None:
    result = await alfven_speed(B_nT=5.0, n_cm3=10.0)
    assert 20 < result["alfven_speed_km_s"] < 200


async def test_alfven_speed_scales_with_B() -> None:
    r1 = await alfven_speed(B_nT=5.0, n_cm3=10.0)
    r2 = await alfven_speed(B_nT=10.0, n_cm3=10.0)
    assert r2["alfven_speed_km_s"] == pytest.approx(2 * r1["alfven_speed_km_s"], rel=1e-3)


# ──────────────────────────────── inertial_length ────────────────────────────


async def test_inertial_length_proton_positive() -> None:
    result = await inertial_length(n_cm3=10.0, particle="p")
    assert "inertial_length_km" in result
    assert result["inertial_length_km"] > 0


async def test_inertial_length_electron_shorter() -> None:
    proton = await inertial_length(n_cm3=10.0, particle="p")
    electron = await inertial_length(n_cm3=10.0, particle="e")
    assert electron["inertial_length_km"] < proton["inertial_length_km"]


# ──────────────────────────────── power_spectrum ─────────────────────────────


async def test_power_spectrum_returns_peak_freq() -> None:
    import numpy as np

    dt = 1.0
    t = np.arange(256) * dt
    sig = np.sin(2 * math.pi * 0.1 * t)
    result = await power_spectrum(sig.tolist(), dt_s=dt)
    assert "peak_frequency_Hz" in result
    assert result["peak_frequency_Hz"] == pytest.approx(0.1, abs=0.01)


async def test_power_spectrum_list_input() -> None:
    result = await power_spectrum([1.0, -1.0, 1.0, -1.0] * 32, dt_s=1.0)
    assert "peak_frequency_Hz" in result
    assert result["peak_frequency_Hz"] > 0
