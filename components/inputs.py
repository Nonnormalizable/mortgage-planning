"""Streamlit input components for mortgage parameters."""

import streamlit as st
from typing import Tuple, Optional
from src.mortgage import Mortgage
from src.arm import ARMParameters
from src.refinance import RefinanceScenario
from src.payoff import ExtraPayment, LumpSumPayment, PaymentFrequency
from src.monte_carlo import RateSimulationParams, RateModel


def mortgage_input_form(key_prefix: str = "mortgage") -> Optional[Mortgage]:
    """Create input form for fixed-rate mortgage parameters.

    Returns Mortgage object or None if inputs are invalid.
    """
    col1, col2 = st.columns(2)

    with col1:
        principal = st.number_input(
            "Loan Amount ($)",
            min_value=10000.0,
            max_value=10000000.0,
            value=1021440.0,
            step=5000.0,
            format="%.2f",
            key=f"{key_prefix}_principal",
            help="The total amount borrowed",
        )

        annual_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=4.875,
            step=0.125,
            format="%.3f",
            key=f"{key_prefix}_rate",
            help="Annual interest rate as a percentage",
        )

    with col2:
        term_years = st.selectbox(
            "Loan Term",
            options=[10, 15, 20, 25, 30],
            index=4,  # Default to 30 years
            key=f"{key_prefix}_term",
            help="Length of the loan in years",
        )

        start_date = st.text_input(
            "Start Date (optional)",
            value="2025-11",
            placeholder="YYYY-MM",
            key=f"{key_prefix}_start",
            help="When the loan started (for tracking purposes)",
        )

    if principal > 0 and annual_rate >= 0:
        return Mortgage(
            principal=principal,
            annual_rate=annual_rate / 100,
            term_months=term_years * 12,
            start_date=start_date if start_date else None,
        )

    return None


def arm_input_form(key_prefix: str = "arm") -> Optional[ARMParameters]:
    """Create input form for ARM parameters.

    Returns ARMParameters object or None if inputs are invalid.
    """
    col1, col2 = st.columns(2)

    with col1:
        principal = st.number_input(
            "Loan Amount ($)",
            min_value=10000.0,
            max_value=10000000.0,
            value=1021440.0,
            step=5000.0,
            format="%.2f",
            key=f"{key_prefix}_principal",
        )

        initial_rate = st.number_input(
            "Initial Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=4.875,
            step=0.125,
            format="%.3f",
            key=f"{key_prefix}_initial_rate",
            help="The introductory rate during the fixed period",
        )

        term_years = st.selectbox(
            "Total Loan Term",
            options=[15, 20, 30],
            index=2,
            key=f"{key_prefix}_term",
        )

    with col2:
        arm_type = st.selectbox(
            "ARM Type",
            options=["5/1", "5/6", "7/1", "10/1", "3/1"],
            index=0,
            key=f"{key_prefix}_type",
            help="Fixed period / adjustment frequency (e.g., 5/1 = 5 years fixed, then adjusts yearly; 5/6 = 5 years fixed, then adjusts every 6 months)",
        )

        margin = st.number_input(
            "Margin (%)",
            min_value=0.0,
            max_value=5.0,
            value=2.5,
            step=0.25,
            format="%.3f",
            key=f"{key_prefix}_margin",
            help="Added to index rate after fixed period",
        )

    # Parse ARM type
    initial_years = int(arm_type.split("/")[0])
    adjustment_part = int(arm_type.split("/")[1])
    # For 5/6 ARM, adjustment is in months (6); for 5/1, 7/1 etc., it's in years
    adjustment_months = adjustment_part if adjustment_part == 6 else adjustment_part * 12

    st.subheader("Rate Caps")
    col1, col2, col3 = st.columns(3)

    with col1:
        initial_cap = st.number_input(
            "Initial Cap (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            format="%.3f",
            key=f"{key_prefix}_initial_cap",
            help="Maximum rate increase at first adjustment",
        )

    with col2:
        periodic_cap = st.number_input(
            "Periodic Cap (%)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.5,
            format="%.3f",
            key=f"{key_prefix}_periodic_cap",
            help="Maximum rate change at subsequent adjustments",
        )

    with col3:
        lifetime_cap = st.number_input(
            "Lifetime Cap (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            format="%.3f",
            key=f"{key_prefix}_lifetime_cap",
            help="Maximum total increase over initial rate",
        )

    if principal > 0:
        return ARMParameters(
            principal=principal,
            initial_rate=initial_rate / 100,
            term_months=term_years * 12,
            initial_period_months=initial_years * 12,
            adjustment_period_months=adjustment_months,
            initial_cap=initial_cap / 100,
            periodic_cap=periodic_cap / 100,
            lifetime_cap=lifetime_cap / 100,
            margin=margin / 100,
        )

    return None


def refinance_input_form(
    current_balance: float,
    key_prefix: str = "refi"
) -> Optional[RefinanceScenario]:
    """Create input form for refinance scenario.

    Args:
        current_balance: Current loan balance (pre-populated)
        key_prefix: Unique key prefix for form elements

    Returns RefinanceScenario object or None if inputs are invalid.
    """
    col1, col2 = st.columns(2)

    with col1:
        new_principal = st.number_input(
            "New Loan Amount ($)",
            min_value=10000.0,
            max_value=10000000.0,
            value=float(current_balance),
            step=1000.0,
            format="%.2f",
            key=f"{key_prefix}_principal",
            help="Amount to refinance (typically current balance)",
        )

        new_rate = st.number_input(
            "New Interest Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.5,
            step=0.125,
            format="%.3f",
            key=f"{key_prefix}_rate",
        )

        new_term = st.selectbox(
            "New Loan Term",
            options=[10, 15, 20, 25, 30],
            index=4,
            key=f"{key_prefix}_term",
        )

    with col2:
        closing_costs = st.number_input(
            "Closing Costs ($)",
            min_value=0.0,
            max_value=50000.0,
            value=5000.0,
            step=500.0,
            format="%.2f",
            key=f"{key_prefix}_closing",
            help="Total closing costs (appraisal, title, fees, etc.)",
        )

        points = st.number_input(
            "Discount Points",
            min_value=0.0,
            max_value=4.0,
            value=0.0,
            step=0.25,
            format="%.3f",
            key=f"{key_prefix}_points",
            help="Points paid to lower rate (1 point = 1% of loan)",
        )

        roll_costs = st.checkbox(
            "Roll closing costs into loan",
            value=False,
            key=f"{key_prefix}_roll",
            help="Add closing costs to loan principal instead of paying upfront",
        )

    cash_out = st.number_input(
        "Cash-out Amount ($)",
        min_value=0.0,
        max_value=500000.0,
        value=0.0,
        step=5000.0,
        format="%.2f",
        key=f"{key_prefix}_cashout",
        help="Additional cash to take out (increases loan amount)",
    )

    if new_principal > 0:
        return RefinanceScenario(
            new_principal=new_principal + cash_out,
            new_rate=new_rate / 100,
            new_term_months=new_term * 12,
            closing_costs=closing_costs,
            points=points / 100,
            cash_out=cash_out,
            roll_costs_into_loan=roll_costs,
        )

    return None


def extra_payment_input(key_prefix: str = "extra") -> Tuple[list, list]:
    """Create input form for extra payment strategies.

    Returns tuple of (extra_payments list, lump_sums list).
    """
    extra_payments = []
    lump_sums = []

    st.subheader("Extra Monthly Payment")

    extra_enabled = st.checkbox(
        "Add extra monthly payment",
        key=f"{key_prefix}_enabled",
    )

    if extra_enabled:
        col1, col2 = st.columns(2)

        with col1:
            extra_amount = st.number_input(
                "Extra Amount ($)",
                min_value=0.0,
                max_value=10000.0,
                value=200.0,
                step=50.0,
                format="%.2f",
                key=f"{key_prefix}_amount",
            )

        with col2:
            frequency = st.selectbox(
                "Payment Frequency",
                options=["Monthly", "Biweekly"],
                index=0,
                key=f"{key_prefix}_frequency",
            )

        if extra_amount > 0:
            extra_payments.append(ExtraPayment(
                amount=extra_amount,
                frequency=PaymentFrequency.BIWEEKLY if frequency == "Biweekly" else PaymentFrequency.MONTHLY,
            ))

    st.subheader("Lump Sum Payment")

    lump_enabled = st.checkbox(
        "Add one-time lump sum payment",
        key=f"{key_prefix}_lump_enabled",
    )

    if lump_enabled:
        col1, col2 = st.columns(2)

        with col1:
            lump_amount = st.number_input(
                "Lump Sum Amount ($)",
                min_value=0.0,
                max_value=500000.0,
                value=10000.0,
                step=1000.0,
                format="%.2f",
                key=f"{key_prefix}_lump_amount",
            )

        with col2:
            lump_month = st.number_input(
                "Apply at Month #",
                min_value=1,
                max_value=360,
                value=12,
                step=1,
                key=f"{key_prefix}_lump_month",
            )

        if lump_amount > 0:
            lump_sums.append(LumpSumPayment(
                amount=lump_amount,
                month=lump_month,
            ))

    return extra_payments, lump_sums


def monte_carlo_input_form(key_prefix: str = "mc") -> RateSimulationParams:
    """Create input form for Monte Carlo simulation parameters.

    Returns RateSimulationParams object.
    """
    st.subheader("Rate Model Parameters")

    col1, col2 = st.columns(2)

    with col1:
        current_rate = st.number_input(
            "Current Index Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=4.0,
            step=0.25,
            format="%.3f",
            key=f"{key_prefix}_current_rate",
            help="Current value of the rate index (e.g., SOFR)",
        )

        model = st.selectbox(
            "Rate Model",
            options=["Vasicek (Mean-Reverting)", "Geometric Brownian Motion"],
            index=0,
            key=f"{key_prefix}_model",
            help="Vasicek: rates tend to return to long-term average. GBM: rates follow random walk with drift.",
        )

        volatility = st.number_input(
            "Volatility (%)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            format="%.3f",
            key=f"{key_prefix}_volatility",
            help="Annual volatility of rate changes",
        )

    with col2:
        long_term_mean = st.number_input(
            "Long-term Mean Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=4.0,
            step=0.25,
            format="%.3f",
            key=f"{key_prefix}_ltm",
            help="Rate that the model tends toward over time (Vasicek only)",
        )

        mean_reversion = st.number_input(
            "Mean Reversion Speed",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.05,
            format="%.3f",
            key=f"{key_prefix}_reversion",
            help="How quickly rates revert to mean (higher = faster). Vasicek only.",
        )

        num_sims = st.selectbox(
            "Number of Simulations",
            options=[100, 500, 1000, 5000],
            index=2,
            key=f"{key_prefix}_nsims",
            help="More simulations = more accurate but slower",
        )

    return RateSimulationParams(
        current_rate=current_rate / 100,
        model=RateModel.VASICEK if "Vasicek" in model else RateModel.GBM,
        long_term_mean=long_term_mean / 100,
        mean_reversion_speed=mean_reversion,
        volatility=volatility / 100,
        num_simulations=num_sims,
    )


def shotwell_arm_input_form(key_prefix: str = "shotwell_arm") -> Optional[ARMParameters]:
    """Create input form for Shotwell 7/6 ARM parameters.

    Fixed 7/6 ARM structure (84 months fixed, 6-month adjustments).
    Returns ARMParameters object or None if inputs are invalid.
    """
    col1, col2 = st.columns(2)

    with col1:
        principal = st.number_input(
            "Loan Amount ($)",
            min_value=10000.0,
            max_value=10000000.0,
            value=1021440.0,
            step=5000.0,
            format="%.2f",
            key=f"{key_prefix}_principal",
        )

        initial_rate = st.number_input(
            "Initial Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=4.875,
            step=0.125,
            format="%.3f",
            key=f"{key_prefix}_initial_rate",
            help="The introductory rate during the 7-year fixed period",
        )

    with col2:
        term_years = st.selectbox(
            "Total Loan Term",
            options=[15, 20, 30],
            index=2,
            key=f"{key_prefix}_term",
        )

        margin = st.number_input(
            "Margin (%)",
            min_value=0.0,
            max_value=5.0,
            value=2.75,
            step=0.25,
            format="%.3f",
            key=f"{key_prefix}_margin",
            help="Added to index rate after fixed period (Shotwell default: 2.75%)",
        )

    # Fixed 7/6 ARM structure
    initial_period_months = 84  # 7 years
    adjustment_period_months = 6  # 6-month adjustments

    st.caption("**7/6 ARM**: 7-year fixed period, then adjusts every 6 months")

    # Rate caps in expandable section
    with st.expander("Advanced: Rate Caps"):
        col1, col2, col3 = st.columns(3)

        with col1:
            initial_cap = st.number_input(
                "Initial Cap (%)",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                format="%.3f",
                key=f"{key_prefix}_initial_cap",
                help="Maximum rate increase at first adjustment",
            )

        with col2:
            periodic_cap = st.number_input(
                "Periodic Cap (%)",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.5,
                format="%.3f",
                key=f"{key_prefix}_periodic_cap",
                help="Maximum rate change at subsequent adjustments",
            )

        with col3:
            lifetime_cap = st.number_input(
                "Lifetime Cap (%)",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                format="%.3f",
                key=f"{key_prefix}_lifetime_cap",
                help="Maximum total increase over initial rate",
            )

    if principal > 0:
        return ARMParameters(
            principal=principal,
            initial_rate=initial_rate / 100,
            term_months=term_years * 12,
            initial_period_months=initial_period_months,
            adjustment_period_months=adjustment_period_months,
            initial_cap=initial_cap / 100,
            periodic_cap=periodic_cap / 100,
            lifetime_cap=lifetime_cap / 100,
            margin=margin / 100,
        )

    return None


def shotwell_refinance_params_form(
    arm_term_months: int = 360,
    key_prefix: str = "shotwell_refi"
) -> dict:
    """Create input form for Shotwell refinance parameters.

    Args:
        arm_term_months: Total ARM term in months (for slider max)
        key_prefix: Unique key prefix for form elements

    Returns:
        Dictionary with refinance parameters
    """
    col1, col2 = st.columns(2)

    with col1:
        term_years = st.selectbox(
            "Refinance Term (years)",
            options=[15, 20, 25],
            index=2,  # Default to 25 years
            key=f"{key_prefix}_term",
            help="Length of the new fixed-rate mortgage",
        )

        fixed_rate = st.number_input(
            "Refinance Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=5.5,
            step=0.125,
            format="%.3f",
            key=f"{key_prefix}_rate",
            help="Interest rate for the new fixed-rate mortgage",
        )

    with col2:
        refinance_costs = st.number_input(
            "Refinancing Costs ($)",
            min_value=0.0,
            max_value=50000.0,
            value=5000.0,
            step=500.0,
            format="%.2f",
            key=f"{key_prefix}_costs",
            help="Total closing costs for refinancing",
        )

        # Default to month 85 (first month after 7-year fixed period)
        refinance_month = st.slider(
            "Refinance Month",
            min_value=1,
            max_value=arm_term_months - 12,
            value=85,
            key=f"{key_prefix}_month",
            help="Month at which to refinance (85 = first month after 7-year fixed period)",
        )

    return {
        'term_months': term_years * 12,
        'fixed_rate': fixed_rate / 100,
        'refinance_costs': refinance_costs,
        'refinance_month': refinance_month,
    }


def current_loan_status_input(mortgage: Mortgage, key_prefix: str = "status") -> int:
    """Input for current status of existing loan.

    Returns number of payments already made.
    """
    st.subheader("Current Loan Status")

    input_method = st.radio(
        "Specify current status by:",
        options=["Payments Made", "Current Balance"],
        horizontal=True,
        key=f"{key_prefix}_method",
    )

    if input_method == "Payments Made":
        payments_made = st.number_input(
            "Payments Made",
            min_value=0,
            max_value=mortgage.term_months - 1,
            value=24,
            step=1,
            key=f"{key_prefix}_payments",
            help="Number of monthly payments already made",
        )
        current_balance = mortgage.balance_at_month(payments_made)
        st.info(f"Current Balance: ${current_balance:,.2f}")
        return payments_made

    else:
        current_balance = st.number_input(
            "Current Balance ($)",
            min_value=0.0,
            max_value=float(mortgage.principal),
            value=float(mortgage.principal * 0.9),
            step=1000.0,
            format="%.2f",
            key=f"{key_prefix}_balance",
        )

        # Estimate payments made from balance
        # Binary search
        low, high = 0, mortgage.term_months
        while low < high:
            mid = (low + high) // 2
            bal = mortgage.balance_at_month(mid)
            if bal > current_balance:
                low = mid + 1
            else:
                high = mid

        st.info(f"Estimated payments made: {low}")
        return low
