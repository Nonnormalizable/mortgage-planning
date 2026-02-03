"""Extra payment and payoff strategy calculations."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

from .mortgage import Mortgage


class PaymentFrequency(Enum):
    MONTHLY = "monthly"
    BIWEEKLY = "biweekly"


@dataclass
class ExtraPayment:
    """Represents an extra payment strategy."""

    amount: float  # Extra amount per payment
    frequency: PaymentFrequency = PaymentFrequency.MONTHLY
    start_month: int = 1  # When to start extra payments
    end_month: Optional[int] = None  # When to stop (None = until payoff)


@dataclass
class LumpSumPayment:
    """Represents a one-time lump sum payment."""

    amount: float
    month: int  # Which month to apply the payment


def calculate_payoff_with_extra_payments(
    mortgage: Mortgage,
    extra_payments: List[ExtraPayment] = None,
    lump_sums: List[LumpSumPayment] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Calculate amortization schedule with extra payments.

    Args:
        mortgage: The base mortgage
        extra_payments: List of recurring extra payment strategies
        lump_sums: List of one-time lump sum payments

    Returns:
        Tuple of (amortization schedule DataFrame, summary statistics)
    """
    extra_payments = extra_payments or []
    lump_sums = lump_sums or []

    # Sort lump sums by month
    lump_sums_dict = {ls.month: ls.amount for ls in lump_sums}

    schedule = []
    balance = mortgage.principal
    cumulative_interest = 0.0
    cumulative_principal = 0.0
    cumulative_extra = 0.0
    base_payment = mortgage.monthly_payment

    month = 0
    while balance > 0.01 and month < mortgage.term_months * 2:  # Safety limit
        month += 1

        # Calculate interest for this month
        interest = balance * mortgage.monthly_rate

        # Base principal payment
        base_principal = min(base_payment - interest, balance)

        # Calculate extra payments for this month
        extra_this_month = 0.0

        for ep in extra_payments:
            if month < ep.start_month:
                continue
            if ep.end_month and month > ep.end_month:
                continue

            if ep.frequency == PaymentFrequency.MONTHLY:
                extra_this_month += ep.amount
            elif ep.frequency == PaymentFrequency.BIWEEKLY:
                # Biweekly results in 26 payments/year vs 12 monthly
                # Equivalent to ~2.17 extra monthly payments per year
                # Simplify: add 1/12 of annual extra each month
                # Annual extra from biweekly = (ep.amount * 26) - (ep.amount * 2 * 12)
                # = ep.amount * 26 - ep.amount * 24 = ep.amount * 2
                extra_this_month += ep.amount * 2 / 12

        # Add any lump sum for this month
        if month in lump_sums_dict:
            extra_this_month += lump_sums_dict[month]

        # Cap extra payment at remaining balance
        total_principal = base_principal + extra_this_month
        if total_principal > balance:
            extra_this_month = balance - base_principal
            total_principal = balance

        # Update balance
        balance -= total_principal
        cumulative_interest += interest
        cumulative_principal += total_principal
        cumulative_extra += extra_this_month

        total_payment = interest + total_principal

        schedule.append({
            'month': month,
            'payment': round(total_payment, 2),
            'principal': round(total_principal, 2),
            'interest': round(interest, 2),
            'extra_payment': round(extra_this_month, 2),
            'balance': round(max(0, balance), 2),
            'cumulative_interest': round(cumulative_interest, 2),
            'cumulative_principal': round(cumulative_principal, 2),
            'cumulative_extra': round(cumulative_extra, 2),
        })

    df = pd.DataFrame(schedule)

    # Calculate summary statistics
    original_schedule = mortgage.amortization_schedule()
    original_interest = original_schedule['cumulative_interest'].iloc[-1]
    original_months = len(original_schedule)

    summary = {
        'original_term_months': original_months,
        'new_term_months': len(df),
        'months_saved': original_months - len(df),
        'years_saved': round((original_months - len(df)) / 12, 1),
        'original_total_interest': round(original_interest, 2),
        'new_total_interest': round(cumulative_interest, 2),
        'interest_saved': round(original_interest - cumulative_interest, 2),
        'total_extra_paid': round(cumulative_extra, 2),
        'net_savings': round(original_interest - cumulative_interest - cumulative_extra, 2),
        'roi_on_extra': round(
            (original_interest - cumulative_interest) / cumulative_extra * 100, 1
        ) if cumulative_extra > 0 else 0,
    }

    return df, summary


def calculate_biweekly_schedule(mortgage: Mortgage) -> Tuple[pd.DataFrame, dict]:
    """Calculate schedule with biweekly payments.

    Biweekly = 26 half-payments per year = 13 full payments
    vs monthly = 12 full payments per year
    """
    half_payment = mortgage.monthly_payment / 2
    biweekly_periods_per_year = 26

    schedule = []
    balance = mortgage.principal
    cumulative_interest = 0.0
    cumulative_principal = 0.0
    period = 0

    while balance > 0.01 and period < mortgage.term_months * 3:
        period += 1

        # Interest accrues daily, but simplify to per-period
        # Biweekly period = ~14.17 days
        daily_rate = mortgage.annual_rate / 365
        period_rate = daily_rate * (365 / biweekly_periods_per_year)

        interest = balance * period_rate
        principal = min(half_payment - interest, balance)

        if principal < 0:
            principal = 0
            half_payment = interest + principal

        balance -= principal
        cumulative_interest += interest
        cumulative_principal += principal

        # Convert to equivalent month for comparison
        equivalent_month = period * 12 / biweekly_periods_per_year

        schedule.append({
            'period': period,
            'equivalent_month': round(equivalent_month, 2),
            'payment': round(half_payment, 2),
            'principal': round(principal, 2),
            'interest': round(interest, 2),
            'balance': round(max(0, balance), 2),
            'cumulative_interest': round(cumulative_interest, 2),
        })

    df = pd.DataFrame(schedule)

    # Summary
    original_schedule = mortgage.amortization_schedule()
    original_interest = original_schedule['cumulative_interest'].iloc[-1]
    original_months = len(original_schedule)

    final_equivalent_month = df['equivalent_month'].iloc[-1]

    summary = {
        'original_term_months': original_months,
        'new_term_months': round(final_equivalent_month),
        'months_saved': round(original_months - final_equivalent_month),
        'years_saved': round((original_months - final_equivalent_month) / 12, 1),
        'biweekly_payment': round(half_payment, 2),
        'equivalent_monthly': round(half_payment * biweekly_periods_per_year / 12, 2),
        'original_total_interest': round(original_interest, 2),
        'new_total_interest': round(cumulative_interest, 2),
        'interest_saved': round(original_interest - cumulative_interest, 2),
    }

    return df, summary


def find_payoff_date_for_extra(
    mortgage: Mortgage,
    extra_monthly: float,
) -> dict:
    """Find when loan will be paid off with extra monthly payment."""
    extra = ExtraPayment(amount=extra_monthly, frequency=PaymentFrequency.MONTHLY)
    _, summary = calculate_payoff_with_extra_payments(mortgage, [extra])
    return summary


def find_extra_for_target_payoff(
    mortgage: Mortgage,
    target_months: int,
) -> Optional[float]:
    """Find the extra monthly payment needed to pay off in target months.

    Returns None if target is not achievable (already shorter or impossible).
    """
    if target_months >= mortgage.term_months:
        return 0.0

    if target_months <= 0:
        return None

    # Binary search for the right extra payment
    low = 0.0
    high = mortgage.principal / target_months  # Upper bound

    while high - low > 0.01:
        mid = (low + high) / 2
        extra = ExtraPayment(amount=mid, frequency=PaymentFrequency.MONTHLY)
        _, summary = calculate_payoff_with_extra_payments(mortgage, [extra])

        if summary['new_term_months'] > target_months:
            low = mid
        else:
            high = mid

    return round(high, 2)


def compare_payoff_strategies(mortgage: Mortgage) -> pd.DataFrame:
    """Compare common payoff strategies."""
    strategies = []

    # Baseline
    original = mortgage.amortization_schedule()
    strategies.append({
        'strategy': 'Original Schedule',
        'extra_monthly': 0,
        'term_months': mortgage.term_months,
        'total_interest': original['cumulative_interest'].iloc[-1],
        'interest_saved': 0,
        'months_saved': 0,
    })

    # Common extra payment amounts
    extra_amounts = [100, 200, 500, 1000]
    for extra in extra_amounts:
        ep = ExtraPayment(amount=extra, frequency=PaymentFrequency.MONTHLY)
        _, summary = calculate_payoff_with_extra_payments(mortgage, [ep])
        strategies.append({
            'strategy': f'+${extra}/month',
            'extra_monthly': extra,
            'term_months': summary['new_term_months'],
            'total_interest': summary['new_total_interest'],
            'interest_saved': summary['interest_saved'],
            'months_saved': summary['months_saved'],
        })

    # Biweekly
    _, biweekly_summary = calculate_biweekly_schedule(mortgage)
    strategies.append({
        'strategy': 'Biweekly Payments',
        'extra_monthly': biweekly_summary['equivalent_monthly'] - mortgage.monthly_payment,
        'term_months': biweekly_summary['new_term_months'],
        'total_interest': biweekly_summary['new_total_interest'],
        'interest_saved': biweekly_summary['interest_saved'],
        'months_saved': biweekly_summary['months_saved'],
    })

    # One extra payment per year
    annual_extra = mortgage.monthly_payment
    monthly_equivalent = annual_extra / 12
    ep = ExtraPayment(amount=monthly_equivalent, frequency=PaymentFrequency.MONTHLY)
    _, summary = calculate_payoff_with_extra_payments(mortgage, [ep])
    strategies.append({
        'strategy': '1 Extra Payment/Year',
        'extra_monthly': monthly_equivalent,
        'term_months': summary['new_term_months'],
        'total_interest': summary['new_total_interest'],
        'interest_saved': summary['interest_saved'],
        'months_saved': summary['months_saved'],
    })

    df = pd.DataFrame(strategies)
    df['total_interest'] = df['total_interest'].round(2)
    df['interest_saved'] = df['interest_saved'].round(2)
    df['extra_monthly'] = df['extra_monthly'].round(2)

    return df
