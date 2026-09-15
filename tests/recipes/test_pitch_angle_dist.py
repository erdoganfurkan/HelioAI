"""Equal-solid-angle PADs retain particle counts without polar noise weighting."""

from __future__ import annotations

import runpy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.recipes.conftest import RECIPES_DIR

pytestmark = pytest.mark.recipes

PATH = RECIPES_DIR / "pitch_angle_dist.py"
UNITS = {
    "pitch_angles_median_deg": "deg",
    "pitch_angles_mean_deg": "deg",
    "n_particles": "",
    "pad_counts": "",
    "pad_bin_edges_deg": "deg",
    "pad_anisotropy": "",
}


def test_isotropic_velocities_give_flat_equal_solid_angle_counts(recipe):
    run = recipe("pitch_angle_dist", __name__="recipe")
    velocities = np.random.default_rng(42).normal(0, 1, (20000, 3))
    _, counts, edges = run.namespace["compute_pad"](velocities, [0, 0, 1], bins="cos")

    expected = len(velocities) / 18
    assert np.all(np.abs(counts - expected) < 4 * np.sqrt(expected))
    assert counts.sum() == len(velocities)
    assert np.all(np.diff(edges) > 0)
    np.testing.assert_allclose(np.diff(np.cos(np.radians(edges))), -2 / 18)
    assert np.diff(edges)[0] > np.diff(edges)[8]
    assert edges[[0, -1]] == pytest.approx([0, 180])


def test_uniform_degree_bins_have_over_ten_times_fewer_expected_polar_counts():
    n_particles = 20000
    edges = np.radians(np.linspace(0, 180, 19))
    expected = n_particles * (np.cos(edges[:-1]) - np.cos(edges[1:])) / 2

    assert expected.sum() == pytest.approx(n_particles)
    assert expected[8] / expected[0] > 10
    assert (1 / np.sqrt(expected[0])) > 3 * (1 / np.sqrt(expected[8]))


def test_default_cosine_bins_plot_counts_per_steradian_with_degree_widths(recipe, monkeypatch):
    run = recipe("pitch_angle_dist", __name__="recipe")
    monkeypatch.setattr(plt, "show", lambda: None)
    velocities = np.random.default_rng(42).normal(0, 1, (20000, 3))
    _, counts, edges = run.namespace["compute_pad"](velocities, [0, 0, 1])

    ax = plt.gcf().axes[0]
    np.testing.assert_allclose([bar.get_x() for bar in ax.patches], edges[:-1], atol=1e-12)
    np.testing.assert_allclose([bar.get_width() for bar in ax.patches], np.diff(edges))
    heights = np.array([bar.get_height() for bar in ax.patches])
    np.testing.assert_allclose(heights, counts / (4 * np.pi / len(counts)))
    assert heights.sum() * (4 * np.pi / len(counts)) == pytest.approx(len(velocities))
    assert "sr" in ax.get_ylabel()


@pytest.mark.parametrize("n_bins", [9, 18])
def test_degree_bins_preserve_the_legacy_sine_corrected_return_and_plot(
    recipe, monkeypatch, n_bins
):
    run = recipe("pitch_angle_dist", __name__="recipe")
    monkeypatch.setattr(plt, "show", lambda: None)
    velocities = np.random.default_rng(42).normal(0, 1, (20000, 3))
    result = run.namespace["compute_pad"](
        velocities, [0, 0, 1], n_bins, "Legacy comparison", bins="deg"
    )
    pa_deg, counts, edges = result
    raw, expected_edges = np.histogram(pa_deg, bins=n_bins, range=(0, 180))
    centers = 0.5 * (expected_edges[:-1] + expected_edges[1:])
    expected = raw / np.maximum(np.sin(np.radians(centers)), 1e-10)

    np.testing.assert_array_equal(edges, expected_edges)
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(result.pad_counts, raw)
    ax = plt.gcf().axes[0]
    np.testing.assert_array_equal([bar.get_height() for bar in ax.patches], expected)
    assert ax.get_title() == "Legacy comparison"


def test_field_aligned_beam_has_large_anisotropy_and_small_pitch_angles(recipe):
    run = recipe("pitch_angle_dist", __name__="recipe")
    velocities = np.random.default_rng(42).normal(0, 30, (20000, 3)) + [0, 0, 400]
    result = run.namespace["compute_pad"](velocities, [0, 0, 1], bins="cos")

    assert result.pad_anisotropy > 5
    assert result.pitch_angles_median_deg < 15
    assert np.isposinf(result.pad_anisotropy)


def test_pancake_velocities_have_small_anisotropy_and_near_perpendicular_pitch_angles(recipe):
    run = recipe("pitch_angle_dist", __name__="recipe")
    rng = np.random.default_rng(42)
    phi = rng.uniform(0, 2 * np.pi, 20000)
    velocities = 400 * np.column_stack([np.cos(phi), np.sin(phi), np.zeros(len(phi))])
    velocities += rng.normal(0, 30, velocities.shape)
    result = run.namespace["compute_pad"](velocities, [0, 0, 1], bins="cos")

    assert result.pad_anisotropy < 0.2
    assert abs(result.pitch_angles_median_deg - 90) < 5


@pytest.mark.parametrize("n_bins", [17, 18])
def test_anisotropy_compares_both_poles_to_the_one_or_two_central_bins(recipe, n_bins):
    run = recipe("pitch_angle_dist", __name__="recipe")
    cos_edges = np.linspace(1, -1, n_bins + 1)
    cos_centers = (cos_edges[:-1] + cos_edges[1:]) / 2
    expected_counts = np.full(n_bins, 2)
    expected_counts[[0, -1]] = [12, 32]
    middle = slice((n_bins - 1) // 2, n_bins // 2 + 1)
    expected_counts[middle] = np.arange(expected_counts[middle].size) * 4 + 4
    velocities = np.column_stack([np.sqrt(1 - cos_centers**2), np.zeros(n_bins), cos_centers])
    velocities = np.repeat(velocities, expected_counts, axis=0)

    result = run.namespace["compute_pad"](velocities, [0, 0, 1], n_bins=n_bins)

    np.testing.assert_array_equal(result.pad_counts, expected_counts)
    assert result.pad_anisotropy == pytest.approx(
        expected_counts[[0, -1]].mean() / expected_counts[middle].mean()
    )


def test_recipe_source_does_not_select_a_dark_style():
    assert "dark_background" not in PATH.read_text(encoding="utf-8")


def test_computing_a_pad_does_not_change_the_default_figure_style(recipe):
    run = recipe("pitch_angle_dist", __name__="recipe")
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        run.namespace["compute_pad"]([[1, 0, 0]], [0, 0, 1])
        assert plt.rcParams["figure.facecolor"] == matplotlib.rcParamsDefault["figure.facecolor"]


def test_bound_run_exports_every_result_with_units_and_preserves_tuple_unpacking(recipe):
    velocities = np.random.default_rng(42).normal(0, 1, (20000, 3))
    run = recipe("pitch_angle_dist", __name__="recipe", V=velocities, B=[0, 0, 1])

    assert {name: data["units"] for name, data in run.exports.items()} == UNITS
    result = run.namespace["result"]
    assert isinstance(result, tuple)
    pa_deg, counts, edges = result
    assert len(pa_deg) == len(velocities)
    assert run.value("pitch_angles_median_deg") == pytest.approx([np.median(pa_deg)])
    assert run.value("pitch_angles_mean_deg") == pytest.approx([np.mean(pa_deg)])
    assert run.value("n_particles") == pytest.approx([len(velocities)])
    np.testing.assert_array_equal(counts, run.value("pad_counts"))
    np.testing.assert_array_equal(edges, run.value("pad_bin_edges_deg"))
    for name in UNITS:
        np.testing.assert_array_equal(np.atleast_1d(getattr(result, name)), run.value(name))
    assert len(run.figures) == 1
    assert run.figures[0].axes[0].get_title() == "Pitch angle distribution"
    header = next(line for line in PATH.read_text().splitlines() if line.startswith("# outputs:"))
    assert set(header.removeprefix("# outputs:").strip().split(", ")) == set(UNITS)


def test_run_block_honors_the_bound_bin_scheme_count_and_label(recipe):
    run = recipe(
        "pitch_angle_dist",
        __name__="recipe",
        V=[[1, 0, 0], [0, 0, 1]],
        B=[0, 0, 1],
        n_bins=9,
        bins="deg",
        label="Bound settings",
    )

    np.testing.assert_array_equal(run.value("pad_bin_edges_deg"), np.linspace(0, 180, 10))
    assert run.value("pad_counts").shape == (9,)
    assert run.figures[0].axes[0].get_title() == "Bound settings"


@pytest.mark.parametrize("inputs", [{}, {"V": [[1, 0, 0]]}, {"B": [0, 0, 1]}])
def test_unbound_or_partial_run_does_not_export_or_plot_demo_data(recipe, inputs):
    run = recipe("pitch_angle_dist", __name__="recipe", **inputs)
    assert run.exports == {}
    assert run.figures == []


def test_zero_and_nan_vectors_do_not_enter_the_distribution_or_particle_count(recipe):
    run = recipe("pitch_angle_dist", __name__="recipe")
    result = run.namespace["compute_pad"](
        [[1, 0, 0], [0, 0, 1], [0, 0, 0], [np.nan, 1, 0]], [0, 0, 1]
    )

    assert result.n_particles == 2
    assert result.pad_counts.sum() == 2
    assert np.isnan(result[0][2:]).all()


def test_no_valid_particles_leave_empty_counts_and_undefined_statistics(recipe):
    run = recipe("pitch_angle_dist", __name__="recipe")
    result = run.namespace["compute_pad"]([[0, 0, 0]], [0, 0, 1])

    assert result.n_particles == 0
    assert not result.pad_counts.any()
    assert np.isnan(result.pitch_angles_mean_deg)
    assert np.isnan(result.pitch_angles_median_deg)
    assert np.isnan(result.pad_anisotropy)


def test_an_unknown_bin_scheme_is_rejected(recipe):
    compute_pad = recipe("pitch_angle_dist", __name__="recipe").namespace["compute_pad"]
    with pytest.raises(ValueError, match="bins"):
        compute_pad([[1, 0, 0]], [0, 0, 1], bins="invalid")


@pytest.mark.parametrize("n_bins", [0, -1, 1.5])
def test_a_bin_count_must_be_a_positive_integer(recipe, n_bins):
    compute_pad = recipe("pitch_angle_dist", __name__="recipe").namespace["compute_pad"]
    with pytest.raises(ValueError, match="n_bins"):
        compute_pad([[1, 0, 0]], [0, 0, 1], n_bins=n_bins)


@pytest.mark.parametrize("n_bins", [1, 2])
def test_fewer_than_three_bins_cannot_distinguish_poles_from_the_equator(recipe, n_bins):
    compute_pad = recipe("pitch_angle_dist", __name__="recipe").namespace["compute_pad"]
    result = compute_pad([[1, 0, 0]], [0, 0, 1], n_bins=n_bins)

    assert result.pad_counts.sum() == 1
    assert np.isnan(result.pad_anisotropy)


def test_main_demo_also_runs_without_the_sandbox_export_helper(recipe):
    namespace = runpy.run_path(str(PATH), run_name="__main__")
    assert "compute_pad" in namespace
