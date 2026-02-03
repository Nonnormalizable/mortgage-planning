"""Core mortgage and amortization calculations."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class Mortgage:
    """Represents a fixed-rate mortgage."""

    principal: float
    annual_rate: float  # as decimal, e.g., 0.065 for 6.5%
    term_months: int
    start_date: Optional[str] = None  # YYYY-MM format

    @property
    def monthly_rate(self) -> float:
        """Convert annual rate to monthly rate."""
        return self.annual_rate / 12

    @property
    def monthly_payment(self) -> float:
        """Calculate monthly payment using standard amortization formula.

        M = P * [r(1+r)^n] / [(1+r)^n - 1]
        """
        r = self.monthly_rate
        n = self.term_months
        p = self.principal

        if r == 0:
            return p / n

        return p * (r * (1 + r)**n) / ((1 + r)**n - 1)

    @property
    def total_payment(self) -> float:
        """Total amount paid over the life of the loan."""
        return self.monthly_payment * self.term_months

    @property
    def total_interest(self) -> float:
        """Total interest paid over the life of the loan."""
        return self.total_payment - self.principal

    def balance_at_month(self, month: int) -> float:
        """Calculate remaining balance after a specific number of payments.

        B = P * [(1+r)^n - (1+r)^p] / [(1+r)^n - 1]
        where p is payments made
        """
        if month <= 0:
            return self.principal
        if month >= self.term_months:
            return 0.0

        r = self.monthly_rate
        n = self.term_months
        p = self.principal

        if r == 0:
            return p * (1 - month / n)

        return p * ((1 + r)**n - (1 + r)**month) / ((1 + r)**n - 1)

    def amortization_schedule(self) -> pd.DataFrame:
        """Generate full amortization schedule.

        Returns DataFrame with columns:
        - month: payment number (1-indexed)
        - payment: monthly payment amount
        - principal: principal portion of payment
        - interest: interest portion of payment
        - balance: remaining balance after payment
        - cumulative_interest: total interest paid to date
        - cumulative_principal: total principal paid to date
        """
        schedule = []
        balance = self.principal
        cumulative_interest = 0.0
        cumulative_principal = 0.0
        payment = self.monthly_payment

        for month in range(1, self.term_months + 1):
            interest = balance * self.monthly_rate
            principal_paid = payment - interest

            # Handle final payment rounding
            if month == self.term_months:
                principal_paid = balance
                payment = principal_paid + interest

            balance -= principal_paid
            cumulative_interest += interest
            cumulative_principal += principal_paid

            schedule.append({
                'month': month,
                'payment': round(payment, 2),
                'principal': round(principal_paid, 2),
                'interest': round(interest, 2),
                'balance': round(max(0, balance), 2),
                'cumulative_interest': round(cumulative_interest, 2),
                'cumulative_principal': round(cumulative_principal, 2),
            })

        return pd.DataFrame(schedule)


def calculate_monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
    """Standalone function for monthly payment calculation."""
    mortgage = Mortgage(principal, annual_rate, term_months)
    return mortgage.monthly_payment


def calculate_affordability(
    monthly_income: float,
    monthly_debts: float,
    annual_rate: float,
    term_months: int,
    down_payment: float = 0.0,
    max_dti: float = 0.43,
    max_housing_ratio: float = 0.28,
    annual_property_tax_rate: float = 0.01,
    annual_insurance: float = 1200,
) -> dict:
    """Calculate maximum affordable home price.

    Uses standard DTI ratios:
    - Front-end ratio (housing): typically 28%
    - Back-end ratio (total debt): typically 43%

    Args:
        monthly_income: Gross monthly income
        monthly_debts: Existing monthly debt payments
        annual_rate: Mortgage interest rate (decimal)
        term_months: Loan term in months
        down_payment: Cash available for down payment
        max_dti: Maximum debt-to-income ratio
        max_housing_ratio: Maximum housing expense ratio
        annual_property_tax_rate: Property tax as % of home value
        annual_insurance: Annual homeowner's insurance

    Returns:
        Dictionary with affordability details
    """
    # Calculate maximum monthly housing payment based on both ratios
    max_housing_front = monthly_income * max_housing_ratio
    max_housing_back = (monthly_income * max_dti) - monthly_debts
    max_housing = min(max_housing_front, max_housing_back)

    # Estimate taxes and insurance per month (will be refined iteratively)
    # Start with rough estimate
    estimated_home_price = 300000  # initial guess

    for _ in range(10):  # iterate to converge
        monthly_tax = (estimated_home_price * annual_property_tax_rate) / 12
        monthly_insurance = annual_insurance / 12

        # Maximum available for principal + interest
        max_pi = max_housing - monthly_tax - monthly_insurance

        if max_pi <= 0:
            max_loan = 0
        else:
            # Reverse the payment formula to get principal
            r = annual_rate / 12
            n = term_months
            if r == 0:
                max_loan = max_pi * n
            else:
                max_loan = max_pi * ((1 + r)**n - 1) / (r * (1 + r)**n)

        new_home_price = max_loan + down_payment
        if abs(new_home_price - estimated_home_price) < 100:
            break
        estimated_home_price = new_home_price

    max_home_price = max_loan + down_payment

    # Handle zero income case
    if monthly_income <= 0:
        front_end_ratio = 0.0
        back_end_ratio = 0.0
    else:
        front_end_ratio = round(max_housing / monthly_income, 4)
        back_end_ratio = round((max_housing + monthly_debts) / monthly_income, 4)

    return {
        'max_home_price': round(max_home_price, 2),
        'max_loan_amount': round(max_loan, 2),
        'down_payment': down_payment,
        'estimated_monthly_payment': round(max_housing, 2),
        'principal_and_interest': round(max_pi, 2),
        'estimated_monthly_tax': round(monthly_tax, 2),
        'estimated_monthly_insurance': round(monthly_insurance, 2),
        'front_end_ratio': front_end_ratio,
        'back_end_ratio': back_end_ratio,
    }
