"""Tests for core mortgage calculations."""

import pytest
import numpy as np
from src.mortgage import Mortgage, calculate_monthly_payment, calculate_affordability


class TestMortgage:
    """Tests for Mortgage class."""

    def test_monthly_payment_calculation(self):
        """Test standard amortization formula."""
        # $300,000 at 6.5% for 30 years
        mortgage = Mortgage(
            principal=300000,
            annual_rate=0.065,
            term_months=360,
        )

        # Expected: ~$1,896.20
        assert abs(mortgage.monthly_payment - 1896.20) < 0.10

    def test_monthly_payment_zero_rate(self):
        """Test edge case of 0% interest."""
        mortgage = Mortgage(
            principal=120000,
            annual_rate=0.0,
            term_months=120,
        )

        assert mortgage.monthly_payment == 1000.0

    def test_total_interest(self):
        """Test total interest calculation."""
        mortgage = Mortgage(
            principal=300000,
            annual_rate=0.065,
            term_months=360,
        )

        # Total paid - principal = interest
        expected_total = mortgage.monthly_payment * 360
        expected_interest = expected_total - 300000

        assert abs(mortgage.total_interest - expected_interest) < 1.0

    def test_balance_at_month(self):
        """Test remaining balance calculation."""
        mortgage = Mortgage(
            principal=300000,
            annual_rate=0.065,
            term_months=360,
        )

        # Balance at month 0 should equal principal
        assert mortgage.balance_at_month(0) == 300000

        # Balance at end should be 0
        assert mortgage.balance_at_month(360) == 0.0

        # Balance should decrease over time
        balance_60 = mortgage.balance_at_month(60)
        balance_120 = mortgage.balance_at_month(120)
        assert balance_60 > balance_120

    def test_amortization_schedule_length(self):
        """Test that schedule has correct number of rows."""
        mortgage = Mortgage(
            principal=200000,
            annual_rate=0.05,
            term_months=180,
        )

        schedule = mortgage.amortization_schedule()
        assert len(schedule) == 180

    def test_amortization_schedule_final_balance(self):
        """Test that final balance is zero."""
        mortgage = Mortgage(
            principal=250000,
            annual_rate=0.055,
            term_months=360,
        )

        schedule = mortgage.amortization_schedule()
        assert schedule.iloc[-1]['balance'] == 0.0

    def test_amortization_schedule_principal_sum(self):
        """Test that total principal paid equals original principal."""
        mortgage = Mortgage(
            principal=200000,
            annual_rate=0.06,
            term_months=240,
        )

        schedule = mortgage.amortization_schedule()
        total_principal = schedule['principal'].sum()

        # Allow for small rounding differences
        assert abs(total_principal - 200000) < 1.0

    def test_15_year_vs_30_year(self):
        """Test that 15-year loan has higher payment but less total interest."""
        mortgage_30 = Mortgage(principal=300000, annual_rate=0.065, term_months=360)
        mortgage_15 = Mortgage(principal=300000, annual_rate=0.060, term_months=180)

        # 15-year has higher payment
        assert mortgage_15.monthly_payment > mortgage_30.monthly_payment

        # 15-year has less total interest
        assert mortgage_15.total_interest < mortgage_30.total_interest


class TestCalculateMonthlyPayment:
    """Tests for standalone payment function."""

    def test_matches_mortgage_class(self):
        """Test that standalone function matches class method."""
        principal = 250000
        rate = 0.07
        term = 360

        standalone = calculate_monthly_payment(principal, rate, term)
        mortgage = Mortgage(principal, rate, term)

        assert standalone == mortgage.monthly_payment


class TestCalculateAffordability:
    """Tests for affordability calculation."""

    def test_basic_affordability(self):
        """Test basic affordability calculation."""
        result = calculate_affordability(
            monthly_income=10000,
            monthly_debts=500,
            annual_rate=0.065,
            term_months=360,
            down_payment=60000,
        )

        # Should return reasonable home price
        assert result['max_home_price'] > 0
        assert result['max_loan_amount'] > 0
        assert result['max_loan_amount'] + result['down_payment'] == result['max_home_price']

        # Front-end ratio should not exceed max
        assert result['front_end_ratio'] <= 0.28 + 0.01  # Allow small tolerance

    def test_affordability_with_high_debts(self):
        """Test that high debts reduce affordability."""
        result_low_debt = calculate_affordability(
            monthly_income=10000,
            monthly_debts=100,
            annual_rate=0.065,
            term_months=360,
        )

        result_high_debt = calculate_affordability(
            monthly_income=10000,
            monthly_debts=2000,
            annual_rate=0.065,
            term_months=360,
        )

        assert result_high_debt['max_home_price'] < result_low_debt['max_home_price']

    def test_affordability_zero_income(self):
        """Test edge case of zero income."""
        result = calculate_affordability(
            monthly_income=0,
            monthly_debts=0,
            annual_rate=0.065,
            term_months=360,
        )

        assert result['max_loan_amount'] == 0
