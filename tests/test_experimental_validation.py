"""
Tests for experimental validation demo.
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest

# Import functions from the demo script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from experimental_validation_demo import (
    load_or_create_dataset,
    predict_ddg_batch,
    calculate_metrics,
    plot_experimental_vs_predicted,
    plot_error_distribution
)

import polars as pl
import numpy as np


class TestExperimentalValidation:
    """Test experimental validation demo functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_load_dataset(self, temp_dir):
        """Test dataset loading."""
        df = load_or_create_dataset(output_dir=temp_dir)

        # Check structure
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 13
        assert set(df.columns) == {
            'protein', 'pdb_id', 'mutation', 'chain', 'experimental_ddg', 'reference'
        }

        # Check CSV was created
        csv_path = Path(temp_dir) / "validation_dataset.csv"
        assert csv_path.exists()

        # Check data types
        assert df['experimental_ddg'].dtype == pl.Float64
        assert df['protein'].dtype == pl.Utf8

    def test_predict_ddg_mock(self, temp_dir):
        """Test mock predictions (fast, no OpenMM required)."""
        dataset_df = load_or_create_dataset(output_dir=temp_dir)

        # Run with mock predictions
        results_df = predict_ddg_batch(
            dataset_df,
            n_runs=3,
            max_iterations=1000,
            use_real_predictions=False
        )

        # Check structure
        assert isinstance(results_df, pl.DataFrame)
        assert len(results_df) == 13
        assert 'predicted_ddg' in results_df.columns
        assert 'experimental_ddg' in results_df.columns

        # Check predictions are numeric
        assert results_df['predicted_ddg'].dtype == pl.Float64

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        experimental = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        predicted = np.array([0.6, 1.1, 1.4, 2.1, 2.3])

        metrics = calculate_metrics(experimental, predicted)

        # Check all metrics are present
        assert 'r_squared' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'pearson_r' in metrics
        assert 'spearman_rho' in metrics
        assert 'n' in metrics

        # Check values are reasonable
        assert 0 <= metrics['r_squared'] <= 1
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert -1 <= metrics['pearson_r'] <= 1
        assert -1 <= metrics['spearman_rho'] <= 1
        assert metrics['n'] == 5

        # For perfect predictions, R² should be close to 1
        perfect_predicted = experimental.copy()
        perfect_metrics = calculate_metrics(experimental, perfect_predicted)
        assert perfect_metrics['r_squared'] > 0.99
        assert perfect_metrics['rmse'] < 0.01

    def test_plot_experimental_vs_predicted(self, temp_dir):
        """Test validation plot generation."""
        # Create sample data
        results_df = pl.DataFrame({
            'protein': ['TestProtein'] * 10,
            'pdb_id': ['1ABC'] * 10,
            'mutation': [f'A{i}G' for i in range(10)],
            'chain': ['A'] * 10,
            'experimental_ddg': np.linspace(-1, 2, 10),
            'predicted_ddg': np.linspace(-1, 2, 10) + np.random.normal(0, 0.2, 10),
            'reference': ['Test'] * 10
        })

        output_file = Path(temp_dir) / "test_validation_plot.png"

        # Generate plot
        metrics = plot_experimental_vs_predicted(
            results_df,
            output_file=str(output_file),
            figsize=(8, 8)
        )

        # Check file was created
        assert output_file.exists()
        assert output_file.stat().st_size > 1000  # Non-trivial PNG

        # Check metrics returned
        assert isinstance(metrics, dict)
        assert 'r_squared' in metrics

    def test_plot_error_distribution(self, temp_dir):
        """Test error distribution plot."""
        # Create sample data
        results_df = pl.DataFrame({
            'experimental_ddg': np.linspace(-1, 2, 20),
            'predicted_ddg': np.linspace(-1, 2, 20) + np.random.normal(0, 0.3, 20)
        })

        output_file = Path(temp_dir) / "test_error_dist.png"

        # Generate plot
        plot_error_distribution(
            results_df,
            output_file=str(output_file),
            figsize=(10, 5)
        )

        # Check file was created
        assert output_file.exists()
        assert output_file.stat().st_size > 1000

    def test_end_to_end_workflow(self, temp_dir):
        """Test complete validation workflow."""
        # Load dataset
        dataset_df = load_or_create_dataset(output_dir=temp_dir)

        # Run predictions (mock)
        results_df = predict_ddg_batch(
            dataset_df,
            use_real_predictions=False
        )

        # Save results
        results_csv = Path(temp_dir) / "results.csv"
        results_df.write_csv(results_csv)
        assert results_csv.exists()

        # Generate plots
        plot_file = Path(temp_dir) / "validation.png"
        metrics = plot_experimental_vs_predicted(results_df, output_file=str(plot_file))

        error_file = Path(temp_dir) / "errors.png"
        plot_error_distribution(results_df, output_file=str(error_file))

        # Check all outputs exist
        assert plot_file.exists()
        assert error_file.exists()

        # Check metrics are reasonable (mock data should have good correlation)
        assert metrics['r_squared'] > 0.5
        assert metrics['rmse'] < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
