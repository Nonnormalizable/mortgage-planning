"""Tests for Monte Carlo simulation."""

import pytest
import numpy as np
from src.monte_carlo import (
    RateModel,
    RateSimulationParams,
    simulate_rate_paths,
    simulate_arm_outcomes,
    calculate_simulation_statistics,
    generate_fan_chart_data,
    compare_arm_vs_fixed_monte_carlo,
)
from src.arm import ARMParameters


class TestSimulateRatePaths:
    """Tests for rate path simulation."""

    def test_path_shape(self):
        """Test that output has correct shape."""
        params = RateSimulationParams(
            current_rate=0.04,
            num_simulations=100,
            time_horizon_months=120,
            random_seed=42,
        )

        paths = simulate_rate_paths(params)

        assert paths.shape == (100, 120)

    def test_vasicek_mean_reversion(self):
        """Test that Vasicek model shows mean reversion tendency."""
        params = RateSimulationParams(
            current_rate=0.08,  # Start high
            model=RateModel.VASICEK,
            long_term_mean=0.04,  # Revert to lower
            mean_reversion_speed=0.5,  # Fast reversion
            volatility=0.005,  # Low volatility
            num_simulations=1000,
            time_horizon_months=120,
            random_seed=42,
        )

        paths = simulate_rate_paths(params)

        # Average rate at end should be closer to long-term mean than start
        avg_final = np.mean(paths[:, -1])
        assert avg_final < 0.08  # Should have moved toward 0.04
        assert abs(avg_final - 0.04) < abs(0.08 - 0.04)  # Closer to mean

    def test_rates_non_negative(self):
        """Test that rates don't go negative."""
        params = RateSimulationParams(
            current_rate=0.02,  # Start low
            model=RateModel.VASICEK,
            long_term_mean=0.01,
            volatility=0.03,  # High volatility
            num_simulations=1000,
            time_horizon_months=240,
            random_seed=42,
        )

        paths = simulate_rate_paths(params)

        assert np.all(paths >= 0)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        params = RateSimulationParams(
            current_rate=0.04,
            num_simulations=50,
            time_horizon_months=60,
            random_seed=123,
        )

        paths1 = simulate_rate_paths(params)
        paths2 = simulate_rate_paths(params)

        np.testing.assert_array_equal(paths1, paths2)


class TestSimulateARMOutcomes:
    """Tests for ARM outcome simulation."""

    def test_outcome_structure(self):
        """Test that outcomes have expected keys."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
        )

        rate_params = RateSimulationParams(
            current_rate=0.04,
            num_simulations=100,
            time_horizon_months=360,
            random_seed=42,
        )

        results = simulate_arm_outcomes(arm, rate_params)

        assert 'total_interest' in results
        assert 'total_paid' in results
        assert 'max_payment' in results
        assert 'final_rate' in results
        assert 'avg_rate' in results

    def test_outcome_array_lengths(self):
        """Test that all outcome arrays have correct length."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
        )

        rate_params = RateSimulationParams(
            current_rate=0.04,
            num_simulations=50,
            time_horizon_months=360,
            random_seed=42,
        )

        results = simulate_arm_outcomes(arm, rate_params)

        for key, values in results.items():
            assert len(values) == 50


class TestCalculateSimulationStatistics:
    """Tests for simulation statistics calculation."""

    def test_statistics_structure(self):
        """Test that statistics DataFrame has expected columns."""
        # Mock results
        results = {
            'total_interest': np.random.normal(150000, 20000, 100),
            'max_payment': np.random.normal(2000, 200, 100),
        }

        stats = calculate_simulation_statistics(results)

        assert 'metric' in stats.columns
        assert 'mean' in stats.columns
        assert 'median' in stats.columns
        assert 'p5' in stats.columns
        assert 'p95' in stats.columns

    def test_percentiles_ordered(self):
        """Test that percentiles are in correct order."""
        results = {
            'total_interest': np.random.normal(150000, 20000, 1000),
        }

        stats = calculate_simulation_statistics(results)
        row = stats[stats['metric'] == 'total_interest'].iloc[0]

        assert row['p5'] <= row['p25']
        assert row['p25'] <= row['median']
        assert row['median'] <= row['p75']
        assert row['p75'] <= row['p95']


class TestGenerateFanChartData:
    """Tests for fan chart data generation."""

    def test_fan_chart_shape(self):
        """Test that fan chart data has correct shape."""
        paths = np.random.normal(0.04, 0.01, (100, 60))

        data = generate_fan_chart_data(paths)

        assert len(data) == 60
        assert 'month' in data.columns
        assert 'p5' in data.columns
        assert 'p50' in data.columns
        assert 'p95' in data.columns
        assert 'mean' in data.columns

    def test_fan_chart_percentiles_ordered(self):
        """Test that percentiles are ordered at each time point."""
        paths = np.random.normal(0.04, 0.01, (100, 60))

        data = generate_fan_chart_data(paths)

        for _, row in data.iterrows():
            assert row['p5'] <= row['p25']
            assert row['p25'] <= row['p50']
            assert row['p50'] <= row['p75']
            assert row['p75'] <= row['p95']


class TestCompareARMVsFixedMonteCarlo:
    """Tests for ARM vs fixed Monte Carlo comparison."""

    def test_comparison_structure(self):
        """Test that comparison returns expected keys."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
        )

        rate_params = RateSimulationParams(
            current_rate=0.04,
            num_simulations=100,
            time_horizon_months=360,
            random_seed=42,
        )

        result = compare_arm_vs_fixed_monte_carlo(arm, 0.065, rate_params)

        assert 'fixed_total_interest' in result
        assert 'arm_mean_total_interest' in result
        assert 'probability_arm_saves_money' in result
        assert 'expected_savings_with_arm' in result

    def test_probability_range(self):
        """Test that probability is between 0 and 1."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
        )

        rate_params = RateSimulationParams(
            current_rate=0.04,
            num_simulations=100,
            time_horizon_months=360,
            random_seed=42,
        )

        result = compare_arm_vs_fixed_monte_carlo(arm, 0.065, rate_params)

        assert 0 <= result['probability_arm_saves_money'] <= 1
        assert 0 <= result['probability_arm_costs_more'] <= 1
        assert abs(result['probability_arm_saves_money'] + result['probability_arm_costs_more'] - 1.0) < 0.01

    def test_low_rate_arm_usually_better(self):
        """Test that ARM with much lower initial rate has high probability of being better."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.03,  # Very low initial rate
            term_months=360,
            initial_period_months=60,
            lifetime_cap=0.03,  # Also low lifetime cap
        )

        rate_params = RateSimulationParams(
            current_rate=0.03,
            long_term_mean=0.04,
            volatility=0.005,  # Low volatility
            num_simulations=200,
            time_horizon_months=360,
            random_seed=42,
        )

        result = compare_arm_vs_fixed_monte_carlo(arm, 0.07, rate_params)

        # With very low ARM rate and caps, ARM should usually win
        assert result['probability_arm_saves_money'] > 0.5
