"""Monte Carlo simulation for interest rate scenarios."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

from .arm import ARMParameters, generate_arm_schedule


class RateModel(Enum):
    """Interest rate simulation models."""
    GBM = "gbm"  # Geometric Brownian Motion
    VASICEK = "vasicek"  # Mean-reverting (Vasicek model)


@dataclass
class RateSimulationParams:
    """Parameters for rate simulation."""

    current_rate: float  # Current index rate (e.g., SOFR)
    model: RateModel = RateModel.VASICEK

    # Vasicek parameters
    long_term_mean: float = 0.04  # Long-term average rate
    mean_reversion_speed: float = 0.1  # How fast rate reverts to mean
    volatility: float = 0.01  # Annual volatility

    # GBM parameters (uses volatility from above)
    drift: float = 0.0  # Annual drift rate

    # Simulation settings
    num_simulations: int = 1000
    time_horizon_months: int = 360  # How far to simulate
    random_seed: Optional[int] = None


def simulate_rate_paths(params: RateSimulationParams) -> np.ndarray:
    """Simulate interest rate paths.

    Returns:
        Array of shape (num_simulations, time_horizon_months) with rate paths
    """
    if params.random_seed is not None:
        np.random.seed(params.random_seed)

    n_sims = params.num_simulations
    n_months = params.time_horizon_months

    # Time step (monthly)
    dt = 1 / 12

    # Initialize paths
    paths = np.zeros((n_sims, n_months + 1))
    paths[:, 0] = params.current_rate

    # Generate random shocks
    dW = np.random.normal(0, np.sqrt(dt), (n_sims, n_months))

    if params.model == RateModel.VASICEK:
        # Vasicek model: dr = a(b - r)dt + σdW
        # a = mean reversion speed
        # b = long-term mean
        # σ = volatility
        a = params.mean_reversion_speed
        b = params.long_term_mean
        sigma = params.volatility

        for t in range(n_months):
            r = paths[:, t]
            dr = a * (b - r) * dt + sigma * dW[:, t]
            paths[:, t + 1] = np.maximum(0, r + dr)  # Floor at 0

    elif params.model == RateModel.GBM:
        # Geometric Brownian Motion: dr = μr*dt + σr*dW
        mu = params.drift
        sigma = params.volatility

        for t in range(n_months):
            r = paths[:, t]
            dr = mu * r * dt + sigma * r * dW[:, t]
            paths[:, t + 1] = np.maximum(0, r + dr)  # Floor at 0

    return paths[:, 1:]  # Exclude initial rate


def simulate_arm_outcomes(
    arm_params: ARMParameters,
    rate_sim_params: RateSimulationParams,
) -> dict:
    """Run Monte Carlo simulation for ARM outcomes.

    Returns distribution of total costs, payments, etc.
    """
    # Generate rate paths
    rate_paths = simulate_rate_paths(rate_sim_params)

    # Determine adjustment schedule
    adjustments_needed = []
    month = arm_params.initial_period_months + 1

    while month <= arm_params.term_months:
        adjustments_needed.append(month)
        month += arm_params.adjustment_period_months

    # Run each simulation
    results = {
        'total_interest': [],
        'total_paid': [],
        'max_payment': [],
        'final_rate': [],
        'avg_rate': [],
    }

    for sim_idx in range(rate_sim_params.num_simulations):
        # Extract index values at adjustment points
        index_values = []
        for adj_month in adjustments_needed:
            if adj_month - 1 < rate_paths.shape[1]:
                index_values.append(rate_paths[sim_idx, adj_month - 1])
            else:
                # Use last available rate
                index_values.append(rate_paths[sim_idx, -1])

        # Generate schedule for this rate path
        schedule, _ = generate_arm_schedule(arm_params, index_values)

        results['total_interest'].append(schedule['cumulative_interest'].iloc[-1])
        results['total_paid'].append(schedule['payment'].sum())
        results['max_payment'].append(schedule['payment'].max())
        results['final_rate'].append(schedule['rate'].iloc[-1])
        results['avg_rate'].append(schedule['rate'].mean())

    # Convert to arrays for statistics
    for key in results:
        results[key] = np.array(results[key])

    return results


def calculate_simulation_statistics(results: dict) -> pd.DataFrame:
    """Calculate summary statistics from simulation results."""
    stats = []

    for metric, values in results.items():
        stats.append({
            'metric': metric,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'p5': np.percentile(values, 5),
            'p25': np.percentile(values, 25),
            'median': np.percentile(values, 50),
            'p75': np.percentile(values, 75),
            'p95': np.percentile(values, 95),
            'max': np.max(values),
        })

    df = pd.DataFrame(stats)

    # Round appropriately
    for col in df.columns:
        if col != 'metric':
            df[col] = df[col].round(2)

    return df


def generate_fan_chart_data(
    rate_paths: np.ndarray,
    percentiles: List[int] = [5, 25, 50, 75, 95],
) -> pd.DataFrame:
    """Generate data for fan chart visualization.

    Args:
        rate_paths: Array of shape (n_simulations, n_months)
        percentiles: Percentiles to compute

    Returns:
        DataFrame with percentile bands over time
    """
    n_months = rate_paths.shape[1]
    data = {'month': list(range(1, n_months + 1))}

    for p in percentiles:
        data[f'p{p}'] = np.percentile(rate_paths, p, axis=0)

    data['mean'] = np.mean(rate_paths, axis=0)

    return pd.DataFrame(data)


def compare_arm_vs_fixed_monte_carlo(
    arm_params: ARMParameters,
    fixed_rate: float,
    rate_sim_params: RateSimulationParams,
) -> dict:
    """Compare ARM to fixed rate using Monte Carlo simulation.

    Returns probability that ARM beats fixed, expected savings, etc.
    """
    from .mortgage import Mortgage

    # Fixed mortgage baseline
    fixed = Mortgage(arm_params.principal, fixed_rate, arm_params.term_months)
    fixed_total_interest = fixed.total_interest
    fixed_payment = fixed.monthly_payment

    # Simulate ARM outcomes
    arm_results = simulate_arm_outcomes(arm_params, rate_sim_params)

    # Compare
    arm_beats_fixed = arm_results['total_interest'] < fixed_total_interest
    probability_arm_better = np.mean(arm_beats_fixed)

    savings_if_arm = fixed_total_interest - arm_results['total_interest']

    # Calculate ARM initial payment using standard mortgage formula
    initial_monthly_rate = arm_params.initial_rate / 12
    n = arm_params.term_months
    if initial_monthly_rate > 0:
        arm_initial = arm_params.principal * (initial_monthly_rate * (1 + initial_monthly_rate)**n) / ((1 + initial_monthly_rate)**n - 1)
    else:
        arm_initial = arm_params.principal / n

    # Wrap all values with float() to ensure Python native types for Streamlit metrics
    return {
        'fixed_total_interest': float(round(fixed_total_interest, 2)),
        'fixed_payment': float(round(fixed_payment, 2)),

        'arm_mean_total_interest': float(round(np.mean(arm_results['total_interest']), 2)),
        'arm_median_total_interest': float(round(np.median(arm_results['total_interest']), 2)),

        'probability_arm_saves_money': float(round(probability_arm_better, 4)),
        'probability_arm_costs_more': float(round(1 - probability_arm_better, 4)),

        'expected_savings_with_arm': float(round(np.mean(savings_if_arm), 2)),
        'savings_p5': float(round(np.percentile(savings_if_arm, 5), 2)),
        'savings_p95': float(round(np.percentile(savings_if_arm, 95), 2)),

        'arm_max_payment_p95': float(round(np.percentile(arm_results['max_payment'], 95), 2)),
        'arm_initial_payment': float(round(arm_initial, 2)),
    }


def run_sensitivity_analysis(
    arm_params: ARMParameters,
    base_sim_params: RateSimulationParams,
    volatilities: List[float] = [0.005, 0.01, 0.015, 0.02],
    long_term_means: List[float] = [0.03, 0.04, 0.05, 0.06],
) -> pd.DataFrame:
    """Run sensitivity analysis varying key parameters.

    Returns DataFrame showing how outcomes change with different assumptions.
    """
    results = []

    for vol in volatilities:
        for ltm in long_term_means:
            params = RateSimulationParams(
                current_rate=base_sim_params.current_rate,
                model=base_sim_params.model,
                long_term_mean=ltm,
                mean_reversion_speed=base_sim_params.mean_reversion_speed,
                volatility=vol,
                num_simulations=base_sim_params.num_simulations,
                time_horizon_months=base_sim_params.time_horizon_months,
            )

            sim_results = simulate_arm_outcomes(arm_params, params)
            stats = calculate_simulation_statistics(sim_results)

            total_interest_row = stats[stats['metric'] == 'total_interest'].iloc[0]

            results.append({
                'volatility': vol,
                'long_term_mean': ltm,
                'mean_total_interest': total_interest_row['mean'],
                'p5_total_interest': total_interest_row['p5'],
                'p95_total_interest': total_interest_row['p95'],
                'std_total_interest': total_interest_row['std'],
            })

    return pd.DataFrame(results)
