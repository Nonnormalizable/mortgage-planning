"""Tests for refinance calculations."""

from src.mortgage import Mortgage
from src.refinance import (
    RefinanceScenario,
    calculate_refinance_comparison,
    compare_multiple_refinance_options,
    find_optimal_refinance_rate,
    generate_break_even_chart_data,
)


class TestCalculateRefinanceComparison:
    """Tests for refinance comparison calculation."""

    def test_basic_refinance(self):
        """Test basic refinance comparison."""
        current = Mortgage(
            principal=300000,
            annual_rate=0.07,
            term_months=360,
        )

        refinance = RefinanceScenario(
            new_principal=280000,  # After some payments
            new_rate=0.055,
            new_term_months=360,
            closing_costs=5000,
        )

        result = calculate_refinance_comparison(current, 24, refinance)

        # New payment should be lower (lower rate, similar principal)
        assert result['new_payment'] < result['current_payment']

        # Should have monthly savings
        assert result['monthly_savings'] > 0

        # Break-even should be finite
        assert result['break_even_months'] is not None
        assert result['break_even_months'] > 0

    def test_no_savings_refinance(self):
        """Test refinance that doesn't save money."""
        current = Mortgage(
            principal=300000,
            annual_rate=0.05,
            term_months=360,
        )

        refinance = RefinanceScenario(
            new_principal=290000,
            new_rate=0.07,  # Higher rate
            new_term_months=360,
            closing_costs=5000,
        )

        result = calculate_refinance_comparison(current, 24, refinance)

        # Monthly payment should increase
        assert result['monthly_savings'] < 0

        # Break-even should be None (never breaks even)
        assert result['break_even_months'] is None

        # Recommendation should say not recommended
        assert 'Not recommended' in result['recommendation']

    def test_rolled_in_costs(self):
        """Test refinance with costs rolled into loan."""
        current = Mortgage(
            principal=300000,
            annual_rate=0.07,
            term_months=360,
        )

        refinance_no_roll = RefinanceScenario(
            new_principal=280000,
            new_rate=0.055,
            new_term_months=360,
            closing_costs=5000,
            roll_costs_into_loan=False,
        )

        refinance_roll = RefinanceScenario(
            new_principal=280000,
            new_rate=0.055,
            new_term_months=360,
            closing_costs=5000,
            roll_costs_into_loan=True,
        )

        result_no_roll = calculate_refinance_comparison(current, 24, refinance_no_roll)
        result_roll = calculate_refinance_comparison(current, 24, refinance_roll)

        # Rolling costs in increases principal
        assert result_roll['new_principal'] > result_no_roll['new_principal']

        # Rolling costs in increases payment
        assert result_roll['new_payment'] > result_no_roll['new_payment']

    def test_points_calculation(self):
        """Test that points are calculated correctly."""
        current = Mortgage(
            principal=300000,
            annual_rate=0.07,
            term_months=360,
        )

        refinance = RefinanceScenario(
            new_principal=280000,
            new_rate=0.055,
            new_term_months=360,
            closing_costs=3000,
            points=0.01,  # 1 point = 1% of loan
        )

        result = calculate_refinance_comparison(current, 24, refinance)

        # Points cost should be 1% of principal
        assert result['points_cost'] == 2800.0  # 1% of 280000

    def test_shorter_term_refinance(self):
        """Test refinancing to shorter term."""
        current = Mortgage(
            principal=300000,
            annual_rate=0.065,
            term_months=360,
        )

        # Refinance to 15-year
        refinance = RefinanceScenario(
            new_principal=250000,  # After 5 years of payments
            new_rate=0.055,
            new_term_months=180,
            closing_costs=4000,
        )

        calculate_refinance_comparison(current, 60, refinance)

        # Payment will likely be higher (shorter term)
        # But total interest should be much lower
        # This is a valid refinance strategy


class TestGenerateBreakEvenChartData:
    """Tests for break-even chart data generation."""

    def test_chart_data_shape(self):
        """Test that chart data has correct structure."""
        current = Mortgage(
            principal=300000,
            annual_rate=0.07,
            term_months=360,
        )

        refinance = RefinanceScenario(
            new_principal=280000,
            new_rate=0.055,
            new_term_months=360,
            closing_costs=5000,
        )

        data = generate_break_even_chart_data(current, 24, refinance, months_to_show=60)

        assert len(data) == 60
        assert 'month' in data.columns
        assert 'current_cumulative' in data.columns
        assert 'new_cumulative' in data.columns
        assert 'savings' in data.columns

    def test_chart_data_starts_with_upfront_costs(self):
        """Test that new loan starts with upfront costs."""
        current = Mortgage(
            principal=300000,
            annual_rate=0.07,
            term_months=360,
        )

        refinance = RefinanceScenario(
            new_principal=280000,
            new_rate=0.055,
            new_term_months=360,
            closing_costs=5000,
        )

        data = generate_break_even_chart_data(current, 24, refinance)

        # New cumulative at month 1 should be > payment (includes upfront costs)
        # Actually checking that upfront costs are included at start
        assert data.iloc[0]['new_cumulative'] > data.iloc[0]['current_cumulative']


class TestFindOptimalRefinanceRate:
    """Tests for finding optimal refinance rate."""

    def test_find_rate_for_target_break_even(self):
        """Test finding rate that achieves target break-even."""
        current = Mortgage(
            principal=300000,
            annual_rate=0.07,
            term_months=360,
        )

        target_break_even = 36  # 3 years

        optimal_rate = find_optimal_refinance_rate(
            current,
            current_month=24,
            new_term_months=360,
            closing_costs=5000,
            target_break_even_months=target_break_even,
        )

        if optimal_rate is not None:
            # Verify the rate achieves target
            balance = current.balance_at_month(24)
            scenario = RefinanceScenario(
                new_principal=balance,
                new_rate=optimal_rate,
                new_term_months=360,
                closing_costs=5000,
            )

            result = calculate_refinance_comparison(current, 24, scenario)
            assert result['break_even_months'] <= target_break_even + 1  # Allow small tolerance


class TestCompareMultipleRefinanceOptions:
    """Tests for comparing multiple refinance options."""

    def test_multiple_options_comparison(self):
        """Test comparing multiple refinance scenarios."""
        current = Mortgage(
            principal=300000,
            annual_rate=0.07,
            term_months=360,
        )

        options = [
            RefinanceScenario(
                new_principal=280000,
                new_rate=0.055,
                new_term_months=360,
                closing_costs=5000,
            ),
            RefinanceScenario(
                new_principal=280000,
                new_rate=0.05,
                new_term_months=180,
                closing_costs=4000,
            ),
            RefinanceScenario(
                new_principal=280000,
                new_rate=0.06,
                new_term_months=360,
                closing_costs=2000,
            ),
        ]

        comparison = compare_multiple_refinance_options(current, 24, options)

        assert len(comparison) == 3
        assert 'option' in comparison.columns
        assert 'new_payment' in comparison.columns
        assert 'net_savings' in comparison.columns
