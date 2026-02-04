"""Adjustable Rate Mortgage (ARM) calculations."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class ARMParameters:
    """Parameters for an Adjustable Rate Mortgage."""

    principal: float
    initial_rate: float  # Initial interest rate (decimal)
    term_months: int  # Total loan term
    initial_period_months: int  # Fixed rate period (e.g., 60 for 5/1 ARM)
    adjustment_period_months: int = 12  # How often rate adjusts after initial period

    # Rate caps
    initial_cap: float = 0.05  # Max first adjustment (e.g., 5%)
    periodic_cap: float = 0.01  # Max subsequent adjustments (e.g., 1%)
    lifetime_cap: float = 0.05  # Max increase over initial rate
    lifetime_floor: float = 0.0  # Minimum rate (often equals margin)

    # Index and margin
    margin: float = 0.025  # Added to index to get rate

    start_date: str | None = None

    @property
    def arm_type(self) -> str:
        """Return ARM type string (e.g., '5/1 ARM')."""
        initial_years = self.initial_period_months // 12
        adjustment_years = self.adjustment_period_months // 12
        return f"{initial_years}/{adjustment_years} ARM"


@dataclass
class RateAdjustment:
    """Represents a rate adjustment event."""
    month: int
    new_rate: float
    index_value: float
    is_initial_adjustment: bool


def calculate_arm_rate(
    current_rate: float,
    index_value: float,
    params: ARMParameters,
    is_first_adjustment: bool,
) -> float:
    """Calculate new ARM rate after adjustment.

    Args:
        current_rate: Current interest rate
        index_value: Current index value (e.g., SOFR)
        params: ARM parameters including caps and margin
        is_first_adjustment: True if this is the first adjustment after initial period

    Returns:
        New interest rate (capped appropriately)
    """
    # Calculate fully indexed rate
    fully_indexed_rate = index_value + params.margin

    # Determine applicable cap
    if is_first_adjustment:
        cap = params.initial_cap
    else:
        cap = params.periodic_cap

    # Apply periodic cap (limit change from current rate)
    max_rate = current_rate + cap
    min_rate = current_rate - cap
    new_rate = max(min_rate, min(max_rate, fully_indexed_rate))

    # Apply lifetime cap and floor
    max_lifetime = params.initial_rate + params.lifetime_cap
    new_rate = min(new_rate, max_lifetime)
    new_rate = max(new_rate, params.lifetime_floor)

    return new_rate


def generate_arm_schedule(
    params: ARMParameters,
    future_index_values: list[float] | None = None,
) -> tuple[pd.DataFrame, list[RateAdjustment]]:
    """Generate amortization schedule for an ARM.

    Args:
        params: ARM parameters
        future_index_values: List of index values for each adjustment period.
                           If None, assumes rate stays at initial rate.

    Returns:
        Tuple of (amortization DataFrame, list of RateAdjustment events)
    """
    schedule = []
    adjustments = []

    balance = params.principal
    current_rate = params.initial_rate
    cumulative_interest = 0.0
    cumulative_principal = 0.0

    # Track adjustment periods
    adjustment_count = 0
    months_since_last_adjustment = 0
    in_initial_period = True

    for month in range(1, params.term_months + 1):
        # Check for rate adjustment
        if month > params.initial_period_months:
            if in_initial_period:
                # First adjustment after initial period
                in_initial_period = False
                months_since_last_adjustment = 0

                if future_index_values and len(future_index_values) > adjustment_count:
                    index_value = future_index_values[adjustment_count]
                    new_rate = calculate_arm_rate(
                        current_rate, index_value, params, is_first_adjustment=True
                    )
                    adjustments.append(RateAdjustment(
                        month=month,
                        new_rate=new_rate,
                        index_value=index_value,
                        is_initial_adjustment=True,
                    ))
                    current_rate = new_rate
                    adjustment_count += 1

            elif months_since_last_adjustment >= params.adjustment_period_months:
                # Subsequent adjustment
                months_since_last_adjustment = 0

                if future_index_values and len(future_index_values) > adjustment_count:
                    index_value = future_index_values[adjustment_count]
                    new_rate = calculate_arm_rate(
                        current_rate, index_value, params, is_first_adjustment=False
                    )
                    adjustments.append(RateAdjustment(
                        month=month,
                        new_rate=new_rate,
                        index_value=index_value,
                        is_initial_adjustment=False,
                    ))
                    current_rate = new_rate
                    adjustment_count += 1

        # Calculate payment for remaining term at current rate
        remaining_months = params.term_months - month + 1
        monthly_rate = current_rate / 12

        if monthly_rate == 0:
            payment = balance / remaining_months
        else:
            payment = balance * (monthly_rate * (1 + monthly_rate)**remaining_months) / \
                     ((1 + monthly_rate)**remaining_months - 1)

        # Calculate interest and principal portions
        interest = balance * monthly_rate
        principal_paid = payment - interest

        # Handle final payment
        if month == params.term_months:
            principal_paid = balance
            payment = principal_paid + interest

        balance -= principal_paid
        cumulative_interest += interest
        cumulative_principal += principal_paid
        months_since_last_adjustment += 1

        schedule.append({
            'month': month,
            'payment': round(payment, 2),
            'principal': round(principal_paid, 2),
            'interest': round(interest, 2),
            'balance': round(max(0, balance), 2),
            'rate': round(current_rate, 6),
            'cumulative_interest': round(cumulative_interest, 2),
            'cumulative_principal': round(cumulative_principal, 2),
        })

    return pd.DataFrame(schedule), adjustments


def calculate_arm_worst_case(params: ARMParameters) -> dict:
    """Calculate worst-case scenario for an ARM (all caps hit).

    Returns summary of maximum possible payments and costs.
    """
    # Simulate worst case: rate increases by maximum at each adjustment
    worst_case_indices = []
    current_rate = params.initial_rate

    # Calculate number of adjustments
    remaining_months = params.term_months - params.initial_period_months
    num_adjustments = 1 + (remaining_months - 1) // params.adjustment_period_months

    for i in range(num_adjustments):
        # Index value that would max out the cap
        if i == 0:
            target_rate = current_rate + params.initial_cap
        else:
            target_rate = current_rate + params.periodic_cap

        # Cap at lifetime max
        target_rate = min(target_rate, params.initial_rate + params.lifetime_cap)

        # Index needed to reach target (minus margin)
        needed_index = target_rate - params.margin
        worst_case_indices.append(needed_index)
        current_rate = target_rate

    # Generate schedule with worst case
    schedule, adjustments = generate_arm_schedule(params, worst_case_indices)

    return {
        'max_rate': params.initial_rate + params.lifetime_cap,
        'max_payment': schedule['payment'].max(),
        'total_interest': schedule['cumulative_interest'].iloc[-1],
        'total_paid': schedule['payment'].sum(),
        'adjustments': len(adjustments),
        'schedule': schedule,
    }


def calculate_arm_best_case(params: ARMParameters) -> dict:
    """Calculate best-case scenario for an ARM (rate stays at initial or decreases)."""
    # Best case: index stays low enough that rate doesn't increase
    best_case_indices = [params.initial_rate - params.margin] * 50  # Enough for any term

    schedule, adjustments = generate_arm_schedule(params, best_case_indices)

    return {
        'min_rate': params.initial_rate,  # Can't go below floor in best case
        'min_payment': schedule['payment'].min(),
        'total_interest': schedule['cumulative_interest'].iloc[-1],
        'total_paid': schedule['payment'].sum(),
        'schedule': schedule,
    }


def compare_arm_to_fixed(
    arm_params: ARMParameters,
    fixed_rate: float,
    index_values: list[float] | None = None,
) -> dict:
    """Compare ARM to equivalent fixed-rate mortgage.

    Args:
        arm_params: ARM parameters
        fixed_rate: Fixed rate to compare against
        index_values: Projected index values for ARM adjustments

    Returns:
        Comparison dictionary
    """
    from .mortgage import Mortgage

    # Generate ARM schedule
    arm_schedule, adjustments = generate_arm_schedule(arm_params, index_values)

    # Generate fixed schedule
    fixed = Mortgage(arm_params.principal, fixed_rate, arm_params.term_months)
    fixed_schedule = fixed.amortization_schedule()

    arm_total_interest = arm_schedule['cumulative_interest'].iloc[-1]
    fixed_total_interest = fixed_schedule['cumulative_interest'].iloc[-1]

    return {
        'arm_total_interest': arm_total_interest,
        'fixed_total_interest': fixed_total_interest,
        'interest_difference': arm_total_interest - fixed_total_interest,
        'arm_max_payment': arm_schedule['payment'].max(),
        'fixed_payment': fixed.monthly_payment,
        'arm_initial_payment': arm_schedule['payment'].iloc[0],
        'initial_savings': fixed.monthly_payment - arm_schedule['payment'].iloc[0],
        'arm_schedule': arm_schedule,
        'fixed_schedule': fixed_schedule,
        'adjustments': adjustments,
    }
