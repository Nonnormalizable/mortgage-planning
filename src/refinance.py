"""Refinance comparison and break-even analysis."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List

from .mortgage import Mortgage


@dataclass
class RefinanceScenario:
    """Represents a refinance option."""

    new_principal: float  # Amount being refinanced (current balance + any cash out)
    new_rate: float  # New interest rate (decimal)
    new_term_months: int  # New loan term
    closing_costs: float = 0.0  # Total closing costs
    points: float = 0.0  # Discount points paid (as decimal, e.g., 0.01 = 1 point)
    cash_out: float = 0.0  # Cash-out amount (added to principal)
    roll_costs_into_loan: bool = False  # Whether to add closing costs to principal


def calculate_refinance_comparison(
    current_mortgage: Mortgage,
    current_month: int,  # How many payments have been made
    refinance: RefinanceScenario,
) -> dict:
    """Compare current mortgage to refinance option.

    Args:
        current_mortgage: The existing mortgage
        current_month: Number of payments already made
        refinance: The refinance scenario to evaluate

    Returns:
        Dictionary with comparison metrics
    """
    # Current mortgage remaining
    current_balance = current_mortgage.balance_at_month(current_month)
    remaining_months = current_mortgage.term_months - current_month
    current_payment = current_mortgage.monthly_payment

    # Calculate remaining interest on current mortgage
    current_remaining_schedule = []
    balance = current_balance
    current_remaining_interest = 0.0

    for month in range(remaining_months):
        interest = balance * current_mortgage.monthly_rate
        principal = current_payment - interest
        if month == remaining_months - 1:
            principal = balance
        balance -= principal
        current_remaining_interest += interest

    # New mortgage details
    new_principal = refinance.new_principal
    if refinance.roll_costs_into_loan:
        new_principal += refinance.closing_costs

    # Points cost (typically 1 point = 1% of loan amount)
    points_cost = refinance.points * new_principal

    # Total upfront costs (if not rolled in)
    upfront_costs = points_cost
    if not refinance.roll_costs_into_loan:
        upfront_costs += refinance.closing_costs

    new_mortgage = Mortgage(new_principal, refinance.new_rate, refinance.new_term_months)
    new_payment = new_mortgage.monthly_payment
    new_total_interest = new_mortgage.total_interest

    # Monthly savings (or cost if payment increases)
    monthly_savings = current_payment - new_payment

    # Break-even calculation
    total_costs = refinance.closing_costs + points_cost
    if monthly_savings > 0:
        break_even_months = total_costs / monthly_savings
    else:
        break_even_months = float('inf')  # Never breaks even

    # Total cost comparison over new loan term
    # Current path: remaining payments + what you'd save/invest after payoff
    # New path: new loan payments + upfront costs

    # Simple comparison: total out of pocket for each path
    current_total_remaining = current_payment * remaining_months
    new_total = new_payment * refinance.new_term_months + upfront_costs

    # Interest comparison
    interest_savings = current_remaining_interest - new_total_interest

    # Net savings (accounting for costs)
    net_savings = current_total_remaining - new_total

    return {
        'current_balance': round(current_balance, 2),
        'current_payment': round(current_payment, 2),
        'current_remaining_months': remaining_months,
        'current_remaining_interest': round(current_remaining_interest, 2),
        'current_total_remaining': round(current_total_remaining, 2),

        'new_principal': round(new_principal, 2),
        'new_payment': round(new_payment, 2),
        'new_term_months': refinance.new_term_months,
        'new_total_interest': round(new_total_interest, 2),
        'new_total_cost': round(new_total, 2),

        'closing_costs': round(refinance.closing_costs, 2),
        'points_cost': round(points_cost, 2),
        'total_upfront_costs': round(upfront_costs, 2),

        'monthly_savings': round(monthly_savings, 2),
        'break_even_months': round(break_even_months, 1) if break_even_months != float('inf') else None,
        'interest_savings': round(interest_savings, 2),
        'net_savings': round(net_savings, 2),

        'recommendation': _get_recommendation(break_even_months, remaining_months, net_savings),
    }


def _get_recommendation(break_even_months: float, remaining_months: int, net_savings: float) -> str:
    """Generate a recommendation based on the analysis."""
    if break_even_months == float('inf'):
        return "Not recommended: Monthly payment would increase with no long-term savings."

    if break_even_months > remaining_months:
        return f"Not recommended: Break-even ({break_even_months:.0f} months) exceeds remaining term ({remaining_months} months)."

    if net_savings < 0:
        return "Not recommended: Total cost would be higher than keeping current mortgage."

    if break_even_months <= 12:
        return f"Strongly recommended: Quick break-even in {break_even_months:.0f} months with ${net_savings:,.0f} total savings."

    if break_even_months <= 36:
        return f"Recommended: Reasonable break-even in {break_even_months:.0f} months with ${net_savings:,.0f} total savings."

    return f"Consider carefully: Break-even in {break_even_months:.0f} months. Beneficial if you stay in the home long enough."


def generate_break_even_chart_data(
    current_mortgage: Mortgage,
    current_month: int,
    refinance: RefinanceScenario,
    months_to_show: int = 120,
) -> pd.DataFrame:
    """Generate data for break-even visualization.

    Shows cumulative cost of each option over time.
    """
    comparison = calculate_refinance_comparison(current_mortgage, current_month, refinance)

    current_payment = comparison['current_payment']
    new_payment = comparison['new_payment']
    upfront_costs = comparison['total_upfront_costs']
    remaining_months = comparison['current_remaining_months']

    data = []
    current_cumulative = 0.0
    new_cumulative = upfront_costs  # Start with upfront costs

    for month in range(1, months_to_show + 1):
        # Current mortgage
        if month <= remaining_months:
            current_cumulative += current_payment

        # New mortgage
        if month <= refinance.new_term_months:
            new_cumulative += new_payment

        data.append({
            'month': month,
            'current_cumulative': round(current_cumulative, 2),
            'new_cumulative': round(new_cumulative, 2),
            'savings': round(current_cumulative - new_cumulative, 2),
        })

    return pd.DataFrame(data)


def find_optimal_refinance_rate(
    current_mortgage: Mortgage,
    current_month: int,
    new_term_months: int,
    closing_costs: float,
    target_break_even_months: int = 36,
    rate_search_range: tuple = (0.02, 0.10),
) -> Optional[float]:
    """Find the maximum rate at which refinancing achieves target break-even.

    Useful for determining: "Rates need to drop to X% for refinancing to make sense."
    """
    current_balance = current_mortgage.balance_at_month(current_month)
    current_payment = current_mortgage.monthly_payment

    # Binary search for the rate
    low, high = rate_search_range
    tolerance = 0.0001  # 0.01% precision

    while high - low > tolerance:
        mid_rate = (low + high) / 2

        scenario = RefinanceScenario(
            new_principal=current_balance,
            new_rate=mid_rate,
            new_term_months=new_term_months,
            closing_costs=closing_costs,
        )

        comparison = calculate_refinance_comparison(current_mortgage, current_month, scenario)
        break_even = comparison['break_even_months']

        if break_even is None or break_even > target_break_even_months:
            high = mid_rate  # Need lower rate
        else:
            low = mid_rate  # Can afford higher rate

    # Verify the found rate achieves target
    scenario = RefinanceScenario(
        new_principal=current_balance,
        new_rate=low,
        new_term_months=new_term_months,
        closing_costs=closing_costs,
    )
    comparison = calculate_refinance_comparison(current_mortgage, current_month, scenario)

    if comparison['break_even_months'] and comparison['break_even_months'] <= target_break_even_months:
        return round(low, 4)

    return None


def compare_multiple_refinance_options(
    current_mortgage: Mortgage,
    current_month: int,
    options: List[RefinanceScenario],
) -> pd.DataFrame:
    """Compare multiple refinance options side by side."""
    results = []

    for i, option in enumerate(options):
        comparison = calculate_refinance_comparison(current_mortgage, current_month, option)
        comparison['option'] = i + 1
        comparison['rate'] = option.new_rate
        comparison['term_years'] = option.new_term_months // 12
        results.append(comparison)

    df = pd.DataFrame(results)

    # Reorder columns for readability
    cols = ['option', 'rate', 'term_years', 'new_payment', 'monthly_savings',
            'break_even_months', 'net_savings', 'total_upfront_costs', 'recommendation']
    return df[[c for c in cols if c in df.columns]]
