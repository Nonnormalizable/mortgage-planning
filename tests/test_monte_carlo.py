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
from src.export import rate_sim_params_to_dict, dict_to_rate_sim_params


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


class TestCIRModel:
    """Tests for Cox-Ingersoll-Ross model."""

    def test_cir_rates_non_negative(self):
        """Test that CIR model produces non-negative rates."""
        params = RateSimulationParams(
            current_rate=0.02,  # Start low
            model=RateModel.CIR,
            long_term_mean=0.01,
            mean_reversion_speed=0.1,
            volatility=0.03,  # High volatility
            num_simulations=1000,
            time_horizon_months=240,
            random_seed=42,
        )

        paths = simulate_rate_paths(params)

        assert np.all(paths >= 0)

    def test_cir_mean_reversion(self):
        """Test that CIR model shows mean reversion tendency."""
        params = RateSimulationParams(
            current_rate=0.08,  # Start high
            model=RateModel.CIR,
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

    def test_cir_volatility_scaling(self):
        """Test that CIR volatility scales with sqrt(r)."""
        # Run two simulations - one starting high, one starting low
        params_high = RateSimulationParams(
            current_rate=0.08,
            model=RateModel.CIR,
            long_term_mean=0.08,  # Keep constant
            mean_reversion_speed=0.0,  # No mean reversion
            volatility=0.02,
            num_simulations=1000,
            time_horizon_months=12,
            random_seed=42,
        )

        params_low = RateSimulationParams(
            current_rate=0.02,
            model=RateModel.CIR,
            long_term_mean=0.02,  # Keep constant
            mean_reversion_speed=0.0,  # No mean reversion
            volatility=0.02,
            num_simulations=1000,
            time_horizon_months=12,
            random_seed=42,
        )

        paths_high = simulate_rate_paths(params_high)
        paths_low = simulate_rate_paths(params_low)

        # Standard deviation should be higher for higher starting rate
        std_high = np.std(paths_high[:, -1])
        std_low = np.std(paths_low[:, -1])

        # With sqrt(r) scaling, std_high should be roughly 2x std_low (sqrt(0.08/0.02) = 2)
        assert std_high > std_low


class TestJumpDiffusionModel:
    """Tests for Vasicek + Jump Diffusion model."""

    def test_jump_diffusion_rates_non_negative(self):
        """Test that jump diffusion model produces non-negative rates."""
        params = RateSimulationParams(
            current_rate=0.04,
            model=RateModel.VASICEK_JUMP,
            long_term_mean=0.04,
            mean_reversion_speed=0.1,
            volatility=0.01,
            jump_intensity=1.0,  # 1 jump per year on average
            jump_mean=-0.01,  # Negative jumps
            jump_std=0.02,
            num_simulations=1000,
            time_horizon_months=120,
            random_seed=42,
        )

        paths = simulate_rate_paths(params)

        assert np.all(paths >= 0)

    def test_jump_diffusion_produces_jumps(self):
        """Test that jump diffusion model produces discrete jumps."""
        params = RateSimulationParams(
            current_rate=0.04,
            model=RateModel.VASICEK_JUMP,
            long_term_mean=0.04,
            mean_reversion_speed=0.0,  # No mean reversion
            volatility=0.0,  # No diffusion noise
            jump_intensity=2.0,  # 2 jumps per year on average
            jump_mean=0.01,  # 1% average jump
            jump_std=0.001,  # Small variation
            num_simulations=100,
            time_horizon_months=60,
            random_seed=42,
        )

        paths = simulate_rate_paths(params)

        # With no diffusion, changes should only come from jumps
        # Check that some paths have discrete jumps
        changes = np.diff(paths, axis=1)
        # Some months should have significant changes (jumps)
        large_changes = np.abs(changes) > 0.005
        assert np.any(large_changes)

    def test_jump_intensity_affects_frequency(self):
        """Test that higher jump intensity leads to more jumps."""
        params_low = RateSimulationParams(
            current_rate=0.04,
            model=RateModel.VASICEK_JUMP,
            long_term_mean=0.04,
            mean_reversion_speed=0.0,
            volatility=0.0,
            jump_intensity=0.5,  # Low intensity
            jump_mean=0.01,
            jump_std=0.001,
            num_simulations=500,
            time_horizon_months=120,
            random_seed=42,
        )

        params_high = RateSimulationParams(
            current_rate=0.04,
            model=RateModel.VASICEK_JUMP,
            long_term_mean=0.04,
            mean_reversion_speed=0.0,
            volatility=0.0,
            jump_intensity=2.0,  # High intensity
            jump_mean=0.01,
            jump_std=0.001,
            num_simulations=500,
            time_horizon_months=120,
            random_seed=43,
        )

        paths_low = simulate_rate_paths(params_low)
        paths_high = simulate_rate_paths(params_high)

        # Count significant changes (jumps)
        changes_low = np.abs(np.diff(paths_low, axis=1))
        changes_high = np.abs(np.diff(paths_high, axis=1))

        jumps_low = np.sum(changes_low > 0.005)
        jumps_high = np.sum(changes_high > 0.005)

        # High intensity should have more jumps
        assert jumps_high > jumps_low

    def test_jump_mean_affects_direction(self):
        """Test that jump mean affects the direction of rate movement."""
        params_up = RateSimulationParams(
            current_rate=0.04,
            model=RateModel.VASICEK_JUMP,
            long_term_mean=0.04,
            mean_reversion_speed=0.0,
            volatility=0.001,  # Very low diffusion
            jump_intensity=2.0,
            jump_mean=0.02,  # Positive jumps
            jump_std=0.001,
            num_simulations=500,
            time_horizon_months=60,
            random_seed=42,
        )

        params_down = RateSimulationParams(
            current_rate=0.04,
            model=RateModel.VASICEK_JUMP,
            long_term_mean=0.04,
            mean_reversion_speed=0.0,
            volatility=0.001,
            jump_intensity=2.0,
            jump_mean=-0.02,  # Negative jumps
            jump_std=0.001,
            num_simulations=500,
            time_horizon_months=60,
            random_seed=42,
        )

        paths_up = simulate_rate_paths(params_up)
        paths_down = simulate_rate_paths(params_down)

        # Positive jump mean should lead to higher final rates
        avg_final_up = np.mean(paths_up[:, -1])
        avg_final_down = np.mean(paths_down[:, -1])

        assert avg_final_up > avg_final_down


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing models."""

    def test_vasicek_still_works(self):
        """Test that Vasicek model still works correctly."""
        params = RateSimulationParams(
            current_rate=0.04,
            model=RateModel.VASICEK,
            long_term_mean=0.04,
            mean_reversion_speed=0.1,
            volatility=0.01,
            num_simulations=100,
            time_horizon_months=60,
            random_seed=42,
        )

        paths = simulate_rate_paths(params)

        assert paths.shape == (100, 60)
        assert np.all(paths >= 0)

    def test_gbm_still_works(self):
        """Test that GBM model still works correctly."""
        params = RateSimulationParams(
            current_rate=0.04,
            model=RateModel.GBM,
            drift=0.0,
            volatility=0.01,
            num_simulations=100,
            time_horizon_months=60,
            random_seed=42,
        )

        paths = simulate_rate_paths(params)

        assert paths.shape == (100, 60)
        assert np.all(paths >= 0)

    def test_default_num_simulations_changed(self):
        """Test that default number of simulations is now 300."""
        params = RateSimulationParams(current_rate=0.04)

        assert params.num_simulations == 300

    def test_jump_params_have_defaults(self):
        """Test that jump parameters have sensible defaults."""
        params = RateSimulationParams(current_rate=0.04)

        assert params.jump_intensity == 0.5
        assert params.jump_mean == 0.0025
        assert params.jump_std == 0.005


class TestExportImportWithNewParams:
    """Tests for export/import with new parameters."""

    def test_export_includes_jump_params(self):
        """Test that export includes jump parameters."""
        params = RateSimulationParams(
            current_rate=0.04,
            model=RateModel.VASICEK_JUMP,
            jump_intensity=0.8,
            jump_mean=0.005,
            jump_std=0.01,
        )

        data = rate_sim_params_to_dict(params)

        assert 'jump_intensity' in data
        assert 'jump_mean' in data
        assert 'jump_std' in data
        assert data['jump_intensity'] == 0.8
        assert data['jump_mean'] == 0.005
        assert data['jump_std'] == 0.01

    def test_import_handles_missing_jump_params(self):
        """Test that import handles old data without jump params."""
        old_data = {
            'current_rate': 0.04,
            'model': 'vasicek',
            'long_term_mean': 0.04,
            'mean_reversion_speed': 0.1,
            'volatility': 0.01,
        }

        params = dict_to_rate_sim_params(old_data)

        # Should use defaults for missing jump params
        assert params.jump_intensity == 0.5
        assert params.jump_mean == 0.0025
        assert params.jump_std == 0.005

    def test_import_cir_model(self):
        """Test that CIR model can be imported."""
        data = {
            'current_rate': 0.04,
            'model': 'cir',
            'long_term_mean': 0.04,
        }

        params = dict_to_rate_sim_params(data)

        assert params.model == RateModel.CIR

    def test_import_jump_model(self):
        """Test that jump model can be imported."""
        data = {
            'current_rate': 0.04,
            'model': 'vasicek_jump',
            'jump_intensity': 1.0,
            'jump_mean': 0.01,
            'jump_std': 0.02,
        }

        params = dict_to_rate_sim_params(data)

        assert params.model == RateModel.VASICEK_JUMP
        assert params.jump_intensity == 1.0
        assert params.jump_mean == 0.01
        assert params.jump_std == 0.02
