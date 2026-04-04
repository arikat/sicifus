"""Tests for the sicifus.visualization module."""

import polars as pl
import pytest
from pathlib import Path
import tempfile

from sicifus.visualization import (
    plot_ddg,
    plot_energy_terms,
    plot_position_scan_heatmap,
    plot_alanine_scan,
    plot_ddg_distribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ddg_results_df():
    """Sample ddG results DataFrame."""
    return pl.DataFrame({
        "mutation": ["F13A", "W14L", "G15V", "L16P", "A17D"],
        "ddg_kcal_mol": [1.5, -0.8, 2.3, 0.5, -1.2],
        "ddg_sd": [0.2, 0.15, 0.3, 0.1, 0.25],
    })


@pytest.fixture
def energy_terms_df():
    """Sample energy terms DataFrame."""
    return pl.DataFrame({
        "term": ["HarmonicBondForce", "HarmonicAngleForce", "NonbondedForce", "CustomGBForce"],
        "wt_energy": [150.0, 200.0, -500.0, -100.0],
        "mutant_energy": [152.0, 205.0, -495.0, -98.0],
        "delta": [2.0, 5.0, 5.0, 2.0],
    })


@pytest.fixture
def position_scan_df():
    """Sample position scan DataFrame."""
    rows = []
    for pos in [10, 11, 12]:
        for aa in ["ALA", "GLY", "VAL", "LEU", "ILE"]:
            rows.append({
                "position": pos,
                "wt_residue": "PHE" if pos == 10 else "TRP" if pos == 11 else "GLY",
                "mut_residue": aa,
                "ddg_kcal_mol": (pos - 10) * 0.5 + (ord(aa[0]) - ord('A')) * 0.1,
            })
    return pl.DataFrame(rows)


@pytest.fixture
def alanine_scan_df():
    """Sample alanine scan DataFrame."""
    return pl.DataFrame({
        "chain": ["A"] * 5,
        "position": [10, 11, 12, 13, 14],
        "wt_residue": ["PHE", "TRP", "GLY", "LEU", "ILE"],
        "ddg_kcal_mol": [1.8, 2.5, 0.2, -0.5, 1.2],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_plot_ddg(ddg_results_df):
    """Test ddG bar chart plotting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "ddg_plot.png"

        result_df = plot_ddg(ddg_results_df, output_file=str(output_file))

        # Check file was created
        assert output_file.exists()

        # Check returned DataFrame is sorted
        assert result_df.shape == ddg_results_df.shape
        assert result_df["ddg_kcal_mol"].to_list() == sorted(ddg_results_df["ddg_kcal_mol"].to_list())


def test_plot_ddg_without_error_bars(ddg_results_df):
    """Test ddG plotting without SD column."""
    df_no_sd = ddg_results_df.drop("ddg_sd")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "ddg_plot_no_sd.png"

        result_df = plot_ddg(df_no_sd, output_file=str(output_file), show_error_bars=False)

        assert output_file.exists()
        assert result_df.shape[0] == df_no_sd.shape[0]


def test_plot_energy_terms_grouped(energy_terms_df):
    """Test energy terms plotting with grouped bars."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "energy_terms_grouped.png"

        result_df = plot_energy_terms(energy_terms_df, output_file=str(output_file), plot_type="grouped")

        assert output_file.exists()
        assert result_df.shape[0] == energy_terms_df.shape[0]


def test_plot_energy_terms_stacked(energy_terms_df):
    """Test energy terms plotting with stacked bars."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "energy_terms_stacked.png"

        result_df = plot_energy_terms(energy_terms_df, output_file=str(output_file), plot_type="stacked")

        assert output_file.exists()


def test_plot_position_scan_heatmap(position_scan_df):
    """Test position scan heatmap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "position_scan.png"

        result_df = plot_position_scan_heatmap(position_scan_df, output_file=str(output_file))

        assert output_file.exists()
        assert "amino_acid" in result_df.columns
        assert result_df.shape[0] == 20  # 20 amino acids


def test_plot_alanine_scan(alanine_scan_df):
    """Test alanine scan bar chart."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "alanine_scan.png"

        result_df = plot_alanine_scan(alanine_scan_df, output_file=str(output_file))

        assert output_file.exists()
        assert result_df.shape[0] == alanine_scan_df.shape[0]
        assert result_df["ddg_kcal_mol"].to_list() == sorted(alanine_scan_df["ddg_kcal_mol"].to_list())


def test_plot_ddg_distribution(ddg_results_df):
    """Test ddG distribution histogram."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "ddg_distribution.png"

        result_df = plot_ddg_distribution(ddg_results_df, output_file=str(output_file), bins=5)

        assert output_file.exists()
        assert "bin_center" in result_df.columns
        assert "count" in result_df.columns
        assert result_df.shape[0] == 5  # 5 bins


def test_plot_without_output_file(ddg_results_df):
    """Test plotting without saving to file (interactive mode)."""
    # This should not raise an error, but plt.show() is non-blocking in tests
    result_df = plot_ddg(ddg_results_df, output_file=None)
    assert result_df.shape[0] == ddg_results_df.shape[0]


def test_invalid_plot_type(energy_terms_df):
    """Test error handling for invalid plot type."""
    with pytest.raises(ValueError, match="plot_type must be"):
        plot_energy_terms(energy_terms_df, plot_type="invalid")
