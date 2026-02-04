"""Tests for ARM calculations."""

from src.arm import (
    ARMParameters,
    calculate_arm_best_case,
    calculate_arm_rate,
    calculate_arm_worst_case,
    compare_arm_to_fixed,
    generate_arm_schedule,
)


class TestARMParameters:
    """Tests for ARMParameters dataclass."""

    def test_arm_type_5_1(self):
        """Test 5/1 ARM type string."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.055,
            term_months=360,
            initial_period_months=60,
            adjustment_period_months=12,
        )
        assert arm.arm_type == "5/1 ARM"

    def test_arm_type_7_1(self):
        """Test 7/1 ARM type string."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.055,
            term_months=360,
            initial_period_months=84,
            adjustment_period_months=12,
        )
        assert arm.arm_type == "7/1 ARM"


class TestCalculateARMRate:
    """Tests for ARM rate calculation with caps."""

    def test_rate_within_caps(self):
        """Test rate calculation when fully indexed rate is within caps."""
        params = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
            initial_cap=0.02,
            periodic_cap=0.02,
            lifetime_cap=0.05,
            margin=0.025,
        )

        # Index that would give rate within caps
        index_value = 0.04
        new_rate = calculate_arm_rate(0.05, index_value, params, is_first_adjustment=True)

        # Fully indexed = 0.04 + 0.025 = 0.065
        # Within initial cap of 0.05 + 0.02 = 0.07
        assert abs(new_rate - 0.065) < 0.0001

    def test_initial_cap_applied(self):
        """Test that initial cap limits first adjustment."""
        params = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
            initial_cap=0.02,
            periodic_cap=0.01,
            lifetime_cap=0.05,
            margin=0.025,
        )

        # Index that would push rate above initial cap
        index_value = 0.10  # Would give 0.125, way above 0.05 + 0.02 = 0.07
        new_rate = calculate_arm_rate(0.05, index_value, params, is_first_adjustment=True)

        # Should be capped at initial_rate + initial_cap = 0.07
        assert abs(new_rate - 0.07) < 0.0001

    def test_periodic_cap_applied(self):
        """Test that periodic cap limits subsequent adjustments."""
        params = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
            initial_cap=0.02,
            periodic_cap=0.01,
            lifetime_cap=0.05,
            margin=0.025,
        )

        # Current rate after first adjustment
        current_rate = 0.07

        # Index that would push above periodic cap
        index_value = 0.10
        new_rate = calculate_arm_rate(current_rate, index_value, params, is_first_adjustment=False)

        # Should be capped at current_rate + periodic_cap = 0.08
        assert abs(new_rate - 0.08) < 0.0001

    def test_lifetime_cap_applied(self):
        """Test that lifetime cap limits total increase."""
        params = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
            initial_cap=0.05,  # High initial cap
            periodic_cap=0.05,  # High periodic cap
            lifetime_cap=0.03,  # But lifetime cap is lower
            margin=0.025,
        )

        # Index that would push way above lifetime cap
        index_value = 0.15
        new_rate = calculate_arm_rate(0.05, index_value, params, is_first_adjustment=True)

        # Should be capped at initial_rate + lifetime_cap = 0.08
        assert abs(new_rate - 0.08) < 0.0001


class TestGenerateARMSchedule:
    """Tests for ARM amortization schedule generation."""

    def test_schedule_length(self):
        """Test that schedule has correct number of rows."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.055,
            term_months=360,
            initial_period_months=60,
        )

        schedule, _ = generate_arm_schedule(arm)
        assert len(schedule) == 360

    def test_schedule_final_balance(self):
        """Test that final balance is zero."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.055,
            term_months=360,
            initial_period_months=60,
        )

        schedule, _ = generate_arm_schedule(arm)
        assert schedule.iloc[-1]['balance'] == 0.0

    def test_rate_constant_during_initial_period(self):
        """Test that rate stays constant during initial fixed period."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.055,
            term_months=360,
            initial_period_months=60,
        )

        schedule, _ = generate_arm_schedule(arm, [0.08, 0.09, 0.10])

        # First 60 months should have initial rate
        initial_rates = schedule[schedule['month'] <= 60]['rate']
        assert all(abs(r - 0.055) < 0.0001 for r in initial_rates)

    def test_rate_adjusts_after_initial_period(self):
        """Test that rate changes after initial period."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.055,
            term_months=360,
            initial_period_months=60,
            margin=0.025,
        )

        # Provide index values
        schedule, adjustments = generate_arm_schedule(arm, [0.04, 0.05, 0.06])

        # Should have at least one adjustment
        assert len(adjustments) >= 1

        # Rate at month 61 should be different from initial
        rate_61 = schedule[schedule['month'] == 61]['rate'].iloc[0]
        # With index 0.04 + margin 0.025 = 0.065
        assert abs(rate_61 - 0.065) < 0.0001


class TestARMWorstCase:
    """Tests for worst-case ARM calculation."""

    def test_worst_case_hits_lifetime_cap(self):
        """Test that worst case reaches lifetime cap."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
            lifetime_cap=0.05,
        )

        result = calculate_arm_worst_case(arm)

        assert result['max_rate'] == 0.10  # initial + lifetime cap


class TestARMBestCase:
    """Tests for best-case ARM calculation."""

    def test_best_case_less_interest_than_worst(self):
        """Test that best case has less interest than worst case."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
        )

        best = calculate_arm_best_case(arm)
        worst = calculate_arm_worst_case(arm)

        assert best['total_interest'] < worst['total_interest']


class TestCompareARMToFixed:
    """Tests for ARM vs fixed comparison."""

    def test_comparison_returns_both_schedules(self):
        """Test that comparison includes both schedules."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.05,
            term_months=360,
            initial_period_months=60,
        )

        result = compare_arm_to_fixed(arm, 0.065, [0.04, 0.04, 0.04])

        assert 'arm_schedule' in result
        assert 'fixed_schedule' in result
        assert len(result['arm_schedule']) == 360
        assert len(result['fixed_schedule']) == 360

    def test_initial_savings_calculation(self):
        """Test that initial savings reflects rate difference."""
        arm = ARMParameters(
            principal=300000,
            initial_rate=0.05,  # Lower initial rate
            term_months=360,
            initial_period_months=60,
        )

        result = compare_arm_to_fixed(arm, 0.065, None)  # Higher fixed rate

        # ARM should have initial savings (lower initial payment)
        assert result['initial_savings'] > 0
        assert result['arm_initial_payment'] < result['fixed_payment']
