"""Mortgage Planning & Refinancing Tool - Streamlit Application."""


import numpy as np
import streamlit as st

from components.charts import (
    create_amortization_chart,
    create_arm_vs_fixed_comparison_chart,
    create_cost_distribution_comparison_chart,
    create_cumulative_cost_fan_chart,
    create_monte_carlo_fan_chart,
    create_monte_carlo_histogram,
    create_payment_breakdown_chart,
    create_payoff_comparison_chart,
    create_rate_history_chart,
    create_refinance_break_even_chart,
)

# Import UI components
from components.inputs import (
    arm_input_form,
    current_loan_status_input,
    extra_payment_input,
    monte_carlo_input_form,
    mortgage_input_form,
    refinance_input_form,
    shotwell_arm_input_form,
    shotwell_refinance_params_form,
)
from components.tables import (
    display_amortization_table,
    display_arm_vs_refi_schedule_comparison,
    display_monte_carlo_stats,
    display_payoff_strategy_table,
    display_refinance_summary,
)
from src.arm import (
    calculate_arm_worst_case,
    compare_arm_to_fixed,
    generate_arm_schedule,
)
from src.export import (
    Scenario,
    mortgage_to_dict,
)
from src.monte_carlo import (
    RateModel,
    calculate_simulation_statistics,
    compare_arm_vs_fixed_monte_carlo,
    generate_fan_chart_data,
    get_arm_schedule_for_simulation,
    simulate_arm_outcomes,
    simulate_arm_vs_refinance_monte_carlo,
    simulate_rate_paths,
)

# Import core modules
from src.payoff import (
    calculate_payoff_with_extra_payments,
    compare_payoff_strategies,
)
from src.refinance import (
    calculate_refinance_comparison,
    generate_break_even_chart_data,
)

# Page configuration
st.set_page_config(
    page_title="Mortgage Planning Tool",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric label, .stMetric [data-testid="stMetricValue"], .stMetric [data-testid="stMetricDelta"] {
        color: #262730 !important;
    }
    .info-box {
        background-color: #e7f3fe;
        border-left: 6px solid #2196F3;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def _render_rate_context_widget() -> None:
    """Render the market context widget in the sidebar."""
    st.sidebar.markdown("### Market Context")

    try:
        from src.rates import get_rate_provider

        provider = get_rate_provider()

        # Try to get current rates
        try:
            sofr_rate, sofr_date = provider.get_current_sofr()
            st.sidebar.metric(
                "Current SOFR",
                f"{sofr_rate * 100:.2f}%",
                help=f"As of {sofr_date}",
            )
        except Exception:
            st.sidebar.caption("SOFR: Unavailable")

        try:
            mortgage_rate, mortgage_date = provider.get_current_mortgage_rate()
            st.sidebar.metric(
                "30-Year Fixed",
                f"{mortgage_rate * 100:.2f}%",
                help=f"As of {mortgage_date}",
            )
        except Exception:
            st.sidebar.caption("30-Year Fixed: Unavailable")

        # Historical context
        try:
            stats = provider.get_rate_statistics()
            st.sidebar.caption(
                f"Historical Range: {stats['min'] * 100:.1f}% - {stats['max'] * 100:.1f}%"
            )

            # Current percentile
            try:
                current_sofr, _ = provider.get_current_sofr()
                percentile = provider.get_current_rate_percentile(current_sofr)
                st.sidebar.caption(f"Current vs Historical: {percentile:.0f}th percentile")
            except Exception:
                pass

        except Exception:
            pass

    except Exception:
        st.sidebar.caption("Rate data unavailable")


def main():
    """Main application entry point."""
    st.title("🏠 Mortgage Planning & Refinancing Tool")
    st.markdown("*Educational tool for understanding mortgage decisions*")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Tool",
        options=[
            "Mortgage Calculator",
            "ARM Analysis",
            "Refinance Comparison",
            "Payoff Strategies",
            "Monte Carlo Simulation",
            "Shotwell Refinance",
            "Data Management",
        ],
    )

    st.sidebar.divider()

    # Market Context Widget
    _render_rate_context_widget()

    st.sidebar.divider()

    # Quick Reference - Expandable sections
    st.sidebar.markdown("### Quick Reference")
    with st.sidebar.expander("Amortization"):
        st.markdown("""
        **How loans are paid off over time.**

        Each monthly payment is split between principal and interest. Early payments are mostly interest; later payments are mostly principal. The amortization schedule shows this progression month by month.
        """)
    with st.sidebar.expander("ARM (Adjustable Rate Mortgage)"):
        st.markdown("""
        **Rate changes after an initial fixed period.**

        - **Initial period**: Fixed rate for 3, 5, 7, or 10 years
        - **Adjustment frequency**: How often rate changes (every 6 or 12 months)
        - **Caps**: Limits on rate increases (initial, periodic, lifetime)
        - **Index + Margin**: New rate = market index (e.g., SOFR) + lender's margin
        """)
    with st.sidebar.expander("Break-even"):
        st.markdown("""
        **When refinance savings exceed costs.**

        Divide total refinance costs by monthly savings to find how many months until you recover your investment. If you plan to stay longer than the break-even point, refinancing may make sense.
        """)
    with st.sidebar.expander("Monte Carlo"):
        st.markdown("""
        **Probabilistic simulation of future scenarios.**

        Runs thousands of simulations with random interest rate paths to show the range of possible outcomes. Results show confidence intervals (e.g., 90% of outcomes fall within this range) and percentiles.
        """)
    with st.sidebar.expander("MC Rate Models"):
        st.markdown("""
        **Vasicek**: Mean-reverting model for typical market conditions. Rates tend toward a long-term average.

        **CIR**: Cox-Ingersoll-Ross model prevents negative rates. Good for low-rate environments.

        **Vasicek + Jumps**: Captures sudden Fed moves or market shocks with discrete rate jumps.

        **GBM**: Geometric Brownian Motion - random walk without mean reversion. Models extreme uncertainty.
        """)
    with st.sidebar.expander("MC Key Parameters"):
        st.markdown("""
        **Volatility**: 0.5-1% for stable markets, 1.5-2%+ for uncertain conditions.

        **Long-term Mean**: Historical Fed Funds average ~4.6%. Current neutral rate estimate: 3-4%.

        **Mean Reversion Speed**: 0.05-0.1 slow (5-10 years), 0.15-0.25 moderate, 0.3+ fast (1-2 years).

        **Jump Intensity**: Expected jumps per year. 0.5-1.0 typical for Fed policy surprises.

        **Jump Mean/Std**: Fed typically moves in 0.25% (25bp) increments.
        """)
    with st.sidebar.expander("Interpreting MC Results"):
        st.markdown("""
        **Fan Charts**: Dark band = 50% of outcomes, light band = 90%.

        **Probability Thresholds**: >60% favorable, <40% unfavorable, 40-60% uncertain.

        **Percentiles**: P5/P95 = extreme scenarios, P25/P75 = likely range.

        **Caution**: Models assume historical patterns continue. Actual rates may differ from any model.
        """)

    # Route to appropriate page
    if page == "Mortgage Calculator":
        mortgage_calculator_page()
    elif page == "ARM Analysis":
        arm_analysis_page()
    elif page == "Refinance Comparison":
        refinance_page()
    elif page == "Payoff Strategies":
        payoff_page()
    elif page == "Monte Carlo Simulation":
        monte_carlo_page()
    elif page == "Shotwell Refinance":
        shotwell_refinance_page()
    elif page == "Data Management":
        data_management_page()


def mortgage_calculator_page():
    """Basic mortgage calculator with amortization."""
    st.header("Mortgage Calculator")

    st.markdown("""
    Calculate monthly payments and view the complete amortization schedule for a fixed-rate mortgage.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Loan Parameters")
        mortgage = mortgage_input_form()

        if mortgage:
            st.divider()
            st.subheader("Summary")
            st.metric("Monthly Payment", f"${mortgage.monthly_payment:,.2f}")
            st.metric("Total Interest", f"${mortgage.total_interest:,.2f}")
            st.metric("Total Cost", f"${mortgage.total_payment:,.2f}")

            # Save to session state
            st.session_state['current_mortgage'] = mortgage

    with col2:
        if mortgage:
            schedule = mortgage.amortization_schedule()

            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["Balance Chart", "Payment Breakdown", "Amortization Table"])

            with tab1:
                fig = create_amortization_chart(schedule)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig = create_payment_breakdown_chart(schedule)
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                display_amortization_table(schedule)

            # Download button
            csv = schedule.to_csv(index=False)
            st.download_button(
                "Download Schedule (CSV)",
                csv,
                "amortization_schedule.csv",
                "text/csv",
            )


def arm_analysis_page():
    """ARM analysis and comparison to fixed rate."""
    st.header("Adjustable Rate Mortgage (ARM) Analysis")

    st.markdown("""
    Analyze ARMs and compare them to fixed-rate alternatives. ARMs offer lower initial rates
    but carry risk of rate increases.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("ARM Parameters")
        arm = arm_input_form()

        if arm:
            st.divider()
            st.subheader("Fixed Rate Comparison")
            fixed_rate = st.number_input(
                "Compare to Fixed Rate (%)",
                min_value=0.0,
                max_value=15.0,
                value=6.5,
                step=0.125,
                format="%.3f",
            )

    with col2:
        if arm:
            tab1, tab2, tab3 = st.tabs(["Schedule", "Worst/Best Case", "Fixed Comparison"])

            with tab1:
                schedule, adjustments = generate_arm_schedule(arm)

                st.subheader(f"{arm.arm_type} Schedule")

                # Initial payment
                initial_payment = schedule['payment'].iloc[0]
                st.metric("Initial Payment", f"${initial_payment:,.2f}")

                if adjustments:
                    st.info(f"Rate adjusts {len(adjustments)} times after the {arm.initial_period_months}-month fixed period.")

                fig = create_amortization_chart(schedule)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Scenario Analysis")

                worst = calculate_arm_worst_case(arm)
                from src.arm import calculate_arm_best_case
                best = calculate_arm_best_case(arm)

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**Best Case** (rates stay low)")
                    st.metric("Total Interest", f"${best['total_interest']:,.2f}")
                    st.metric("Minimum Payment", f"${best['min_payment']:,.2f}")

                with col_b:
                    st.markdown("**Worst Case** (all caps hit)")
                    st.metric("Total Interest", f"${worst['total_interest']:,.2f}")
                    st.metric("Maximum Payment", f"${worst['max_payment']:,.2f}")
                    st.metric("Maximum Rate", f"{worst['max_rate']:.3%}")

            with tab3:
                comparison = compare_arm_to_fixed(arm, fixed_rate / 100)

                st.subheader("ARM vs Fixed Comparison")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"**{arm.arm_type}**")
                    st.metric("Initial Payment", f"${comparison['arm_initial_payment']:,.2f}")
                    st.metric("Max Payment", f"${comparison['arm_max_payment']:,.2f}")

                with col_b:
                    st.markdown(f"**{fixed_rate}% Fixed**")
                    st.metric("Payment", f"${comparison['fixed_payment']:,.2f}")

                st.metric(
                    "Initial Monthly Savings (ARM)",
                    f"${comparison['initial_savings']:,.2f}",
                    help="How much less you pay initially with ARM vs fixed"
                )

                fig = create_arm_vs_fixed_comparison_chart(
                    comparison['arm_schedule'],
                    comparison['fixed_schedule'],
                )
                st.plotly_chart(fig, use_container_width=True)


def refinance_page():
    """Refinance comparison and break-even analysis."""
    st.header("Refinance Comparison")

    st.markdown("""
    Compare your current mortgage to refinancing options. See when refinancing pays off
    and whether it makes financial sense.
    """)

    # Check for existing mortgage in session
    if 'current_mortgage' not in st.session_state:
        st.info("Enter your current mortgage details or go to Mortgage Calculator first.")

    st.subheader("Current Mortgage")
    current_mortgage = mortgage_input_form(key_prefix="current")

    if current_mortgage:
        payments_made = current_loan_status_input(current_mortgage)
        current_balance = current_mortgage.balance_at_month(payments_made)

        st.divider()

        st.subheader("Refinance Option")
        refinance = refinance_input_form(current_balance, key_prefix="refi")

        if refinance:
            comparison = calculate_refinance_comparison(
                current_mortgage, payments_made, refinance
            )

            # Display summary
            display_refinance_summary(comparison)

            # Charts
            st.subheader("Break-even Analysis")

            months_to_show = min(
                max(comparison['current_remaining_months'], refinance.new_term_months),
                240,  # Cap at 20 years
            )

            break_even_data = generate_break_even_chart_data(
                current_mortgage,
                payments_made,
                refinance,
                months_to_show=months_to_show,
            )

            fig = create_refinance_break_even_chart(break_even_data)
            st.plotly_chart(fig, use_container_width=True)


def payoff_page():
    """Extra payment and payoff strategy analysis."""
    st.header("Payoff Strategies")

    st.markdown("""
    Explore how extra payments can reduce your loan term and save interest.
    Compare different strategies to find what works best.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Loan Details")
        mortgage = mortgage_input_form(key_prefix="payoff")

        if mortgage:
            st.divider()
            extra_payments, lump_sums = extra_payment_input()

    with col2:
        if mortgage:
            tab1, tab2 = st.tabs(["Custom Strategy", "Compare Strategies"])

            with tab1:
                if extra_payments or lump_sums:
                    modified_schedule, summary = calculate_payoff_with_extra_payments(
                        mortgage, extra_payments, lump_sums
                    )
                    original_schedule = mortgage.amortization_schedule()

                    # Summary metrics
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.metric(
                            "Payoff Time",
                            f"{summary['new_term_months']} months",
                            delta=f"-{summary['months_saved']} months",
                        )

                    with col_b:
                        st.metric(
                            "Interest Saved",
                            f"${summary['interest_saved']:,.2f}",
                        )

                    with col_c:
                        st.metric(
                            "ROI on Extra Payments",
                            f"{summary['roi_on_extra']:.2f}%",
                            help="Interest saved per dollar of extra payments",
                        )

                    # Chart
                    fig = create_payoff_comparison_chart(
                        original_schedule,
                        modified_schedule,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("Configure extra payments in the sidebar to see the impact.")

            with tab2:
                st.subheader("Strategy Comparison")
                strategies = compare_payoff_strategies(mortgage)
                display_payoff_strategy_table(strategies)


def monte_carlo_page():
    """Monte Carlo simulation for ARM rate scenarios."""
    st.header("Monte Carlo Rate Simulation")

    st.markdown("""
    Model uncertain future interest rates using Monte Carlo simulation.
    Understand the range of possible outcomes for adjustable-rate mortgages.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("ARM Parameters")
        arm = arm_input_form(key_prefix="mc_arm")

        if arm:
            st.divider()
            sim_params = monte_carlo_input_form()

            st.divider()
            fixed_rate = st.number_input(
                "Compare to Fixed Rate (%)",
                min_value=0.0,
                max_value=15.0,
                value=6.5,
                step=0.125,
                format="%.3f",
                key="mc_fixed_rate",
            )

            run_sim = st.button("Run Simulation", type="primary")

    with col2:
        if arm and 'run_sim' in dir() and run_sim:
            with st.spinner("Running simulation..."):
                # Simulate rate paths
                sim_params.time_horizon_months = arm.term_months
                rate_paths = simulate_rate_paths(sim_params)

                # Simulate ARM outcomes
                results = simulate_arm_outcomes(arm, sim_params)
                stats = calculate_simulation_statistics(results)

                # Compare to fixed
                comparison = compare_arm_vs_fixed_monte_carlo(
                    arm, fixed_rate / 100, sim_params
                )

                # Store for history tab
                st.session_state['mc_sim_params'] = sim_params

            tab1, tab2, tab3, tab4 = st.tabs([
                "Rate Paths", "Cost Distribution", "ARM vs Fixed", "Rate History"
            ])

            with tab1:
                st.subheader("Projected Interest Rate Paths")
                fan_data = generate_fan_chart_data(rate_paths)
                fig = create_monte_carlo_fan_chart(fan_data)
                st.plotly_chart(fig, use_container_width=True)

                # Show different description for historical model
                if sim_params.model == RateModel.HISTORICAL:
                    st.markdown(f"""
                    **Historical Simulation Results:**
                    Using actual rate history from {sim_params.historical_start_year or 1975}
                    to {sim_params.historical_end_year or 2024}.

                    - **Number of scenarios**: {rate_paths.shape[0]}
                    - Each scenario represents a different starting year from history
                    - **Dark band**: 50% of historical scenarios
                    - **Light band**: 90% of historical scenarios
                    """)
                else:
                    st.markdown("""
                    The fan chart shows the range of possible rate paths:
                    - **Dark band**: 50% of simulations fall within this range
                    - **Light band**: 90% of simulations fall within this range
                    """)

            with tab2:
                st.subheader("Total Interest Distribution")
                fig = create_monte_carlo_histogram(
                    results['total_interest'],
                    title="Distribution of Total Interest Paid",
                    xlabel="Total Interest ($)",
                )
                st.plotly_chart(fig, use_container_width=True)

                display_monte_carlo_stats(stats)

            with tab3:
                st.subheader("ARM vs Fixed Rate Analysis")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric(
                        "Probability ARM Saves Money",
                        f"{comparison['probability_arm_saves_money']:.0%}",
                    )
                    st.metric(
                        "Expected Savings with ARM",
                        f"${round(comparison['expected_savings_with_arm'], -2):,.0f}",
                    )

                with col_b:
                    st.metric(
                        "ARM Max Payment (95th percentile)",
                        f"${round(comparison['arm_max_payment_p95'], -2):,.0f}",
                    )
                    st.metric(
                        "Fixed Payment",
                        f"${round(comparison['fixed_payment'], -2):,.0f}",
                    )

                st.markdown(f"""
                **Analysis Summary:**
                - There is a **{comparison['probability_arm_saves_money']:.0%}** chance the ARM will cost less than the {fixed_rate}% fixed rate.
                - In the best 5% of scenarios, you save at least **${round(comparison['savings_p95'], -2):,.0f}** with the ARM.
                - In the worst 5% of scenarios, the ARM costs **${round(-comparison['savings_p5'], -2):,.0f}** more than fixed.
                """)

            with tab4:
                st.subheader("Historical Rate Context")
                st.markdown("""
                This chart shows the historical path of interest rates to provide context
                for understanding rate simulation assumptions.
                """)

                try:
                    from src.rates import get_rate_provider

                    provider = get_rate_provider()
                    historical_df = provider.get_historical_index_rates(1975, 2024)

                    # Get current rate
                    try:
                        current_sofr, _ = provider.get_current_sofr()
                    except Exception:
                        current_sofr = None

                    fig = create_rate_history_chart(historical_df, current_sofr)
                    st.plotly_chart(fig, use_container_width=True)

                    # Summary statistics
                    stats = provider.get_rate_statistics()
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.metric("Historical Minimum", f"{stats['min'] * 100:.2f}%")
                        st.metric("25th Percentile", f"{stats['p25'] * 100:.2f}%")

                    with col_b:
                        st.metric("Historical Average", f"{stats['mean'] * 100:.2f}%")
                        st.metric("Median", f"{stats['median'] * 100:.2f}%")

                    with col_c:
                        st.metric("Historical Maximum", f"{stats['max'] * 100:.2f}%")
                        st.metric("75th Percentile", f"{stats['p75'] * 100:.2f}%")

                except Exception as e:
                    st.warning(f"Unable to load historical rate data: {e}")


def shotwell_refinance_page():
    """Shotwell Refinance comparison: ARM vs Refinance to Fixed."""
    st.header("Shotwell Refinance Analysis")

    st.markdown("""
    Compare two mortgage strategies for the Shotwell 7/6 ARM:
    1. **Stay in ARM** for the full term (with Monte Carlo simulation for rate uncertainty)
    2. **Refinance to Fixed** at a specified point in time
    """)

    # Top: Input Section (two columns)
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("7/6 ARM Parameters")
        arm = shotwell_arm_input_form()

        if arm:
            st.divider()
            st.subheader("Monte Carlo Parameters")
            sim_params = monte_carlo_input_form(key_prefix="shotwell_mc")

    with col_right:
        st.subheader("Refinance Parameters")
        if arm:
            refi_params = shotwell_refinance_params_form(
                arm_term_months=arm.term_months,
                key_prefix="shotwell_refi"
            )

            st.divider()
            run_comparison = st.button("Run Comparison", type="primary", use_container_width=True)
        else:
            st.info("Configure ARM parameters first.")
            run_comparison = False
            refi_params = None

    # Results section
    # Run simulation when button clicked and store in session state
    if arm and refi_params and run_comparison:
        with st.spinner("Running Monte Carlo simulation..."):
            # Run simulation
            results = simulate_arm_vs_refinance_monte_carlo(
                arm_params=arm,
                refinance_month=refi_params['refinance_month'],
                fixed_rate=refi_params['fixed_rate'],
                fixed_term_months=refi_params['term_months'],
                refinance_costs=refi_params['refinance_costs'],
                rate_sim_params=sim_params,
            )
            # Store results in session state
            st.session_state['shotwell_results'] = results
            st.session_state['shotwell_arm'] = arm
            st.session_state['shotwell_refi_params'] = refi_params

    # Display results if available in session state
    if 'shotwell_results' in st.session_state:
        results = st.session_state['shotwell_results']
        stored_arm = st.session_state['shotwell_arm']
        stored_refi_params = st.session_state['shotwell_refi_params']

        st.divider()

        # Summary Metrics (4 columns)
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Probability ARM Wins",
                f"{results['arm_wins_probability']:.0%}",
                help="Probability that staying in the ARM costs less than refinancing",
            )

        with col2:
            savings_val = results['expected_arm_savings']
            st.metric(
                "Expected ARM Savings",
                f"${round(savings_val, -2):,.0f}",
                help="Expected savings from staying in ARM vs refinancing (positive = ARM saves money)",
            )

        with col3:
            st.metric(
                "ARM Max Payment (95th %ile)",
                f"${round(results['arm_max_payment_p95'], -2):,.0f}",
                help="95th percentile of maximum monthly ARM payment across simulations",
            )

        with col4:
            st.metric(
                "Fixed Payment (after refi)",
                f"${round(results['fixed_payment'], -2):,.0f}",
                help="Monthly payment after refinancing to fixed rate",
            )

        # Detailed Stats (2 columns)
        st.subheader("Detailed Statistics")
        col_arm, col_refi = st.columns(2)

        with col_arm:
            st.markdown("**ARM Total Cost Distribution**")
            st.metric("Median", f"${round(results['arm_median_total'], -2):,.0f}")
            st.metric("5th Percentile (Best)", f"${round(results['arm_p5_total'], -2):,.0f}")
            st.metric("95th Percentile (Worst)", f"${round(results['arm_p95_total'], -2):,.0f}")

        with col_refi:
            st.markdown("**Refinance Total Cost Distribution**")
            st.metric("Median", f"${round(results['refi_median_total'], -2):,.0f}")
            st.metric("5th Percentile (Best)", f"${round(results['refi_p5_total'], -2):,.0f}")
            st.metric("95th Percentile (Worst)", f"${round(results['refi_p95_total'], -2):,.0f}")

        st.divider()

        # Charts (stacked vertically)
        st.subheader("Rate Paths")
        fan_data = generate_fan_chart_data(results['rate_paths'])
        fig_rates = create_monte_carlo_fan_chart(fan_data)
        st.plotly_chart(fig_rates, use_container_width=True)

        st.markdown("""
        The fan chart shows simulated interest rate paths:
        - **Dark band**: 50% of simulations fall within this range
        - **Light band**: 90% of simulations fall within this range
        """)

        st.subheader("Total Cost Distribution")
        fig_dist = create_cost_distribution_comparison_chart(
            results['arm_total_paid'],
            results['refi_total_paid'],
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Cumulative Payments Over Time")
        fig_cumulative = create_cumulative_cost_fan_chart(
            results['arm_cumulative_by_month'],
            results['refi_cumulative_by_month'],
            results['refinance_month'],
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)

        # Interpretation text box
        st.divider()
        st.subheader("Interpretation")

        if results['arm_wins_probability'] > 0.5:
            outcome_text = f"""
            **Based on the simulation, staying in the ARM is more likely to cost less than refinancing.**

            - There is a **{results['arm_wins_probability']:.0%}** probability that the ARM will have lower total costs.
            - On average, the ARM is expected to save **${round(results['expected_arm_savings'], -2):,.0f}** compared to refinancing.
            - However, the ARM carries more uncertainty - in the worst 5% of scenarios, total costs could reach **${round(results['arm_p95_total'], -2):,.0f}**.
            """
        else:
            outcome_text = f"""
            **Based on the simulation, refinancing is more likely to cost less than staying in the ARM.**

            - There is only a **{results['arm_wins_probability']:.0%}** probability that the ARM will have lower total costs.
            - On average, refinancing is expected to save **${round(-results['expected_arm_savings'], -2):,.0f}** compared to the ARM.
            - Refinancing provides payment certainty with a fixed **${round(results['fixed_payment'], -2):,.0f}/month** payment.
            """

        st.info(outcome_text)

        st.markdown("""
        **Key considerations:**
        - ARM outcomes depend heavily on future interest rate movements
        - The simulation uses the Vasicek model which assumes rates tend to revert to a long-term mean
        - Actual market conditions may differ from model assumptions
        - Consider your risk tolerance and financial flexibility when choosing between strategies
        """)

        # Schedule table for a selected simulation
        st.divider()
        st.subheader("Detailed Schedule for Selected Simulation")

        st.markdown("""
        Explore the month-by-month comparison for a specific simulation.
        Select by rank (1 = lowest ARM cost, N = highest ARM cost).
        """)

        # Sort simulations by ARM total cost
        n_sims = len(results['arm_total_paid'])
        sorted_indices = np.argsort(results['arm_total_paid'])

        # Slider to select simulation by rank
        selected_rank = st.slider(
            "Select Simulation by Rank",
            min_value=1,
            max_value=n_sims,
            value=n_sims // 2,  # Default to median
            key="shotwell_sim_rank",
            help="1 = lowest ARM cost (best case), N = highest ARM cost (worst case)",
        )

        # Get the simulation index for this rank
        sim_idx = sorted_indices[selected_rank - 1]

        # Show metrics for selected simulation
        col_sel1, col_sel2, col_sel3 = st.columns(3)
        with col_sel1:
            st.metric(
                "Selected Rank",
                f"#{selected_rank} of {n_sims}",
                help="Rank among all simulations by ARM total cost",
            )
        with col_sel2:
            st.metric(
                "ARM Total Cost",
                f"${results['arm_total_paid'][sim_idx]:,.0f}",
            )
        with col_sel3:
            st.metric(
                "Refi Total Cost",
                f"${results['refi_total_paid'][sim_idx]:,.0f}",
            )

        # Regenerate ARM schedule for this simulation
        arm_schedule, _ = get_arm_schedule_for_simulation(
            stored_arm, results['rate_paths'], sim_idx
        )

        # Generate fixed schedule for the refinance path
        from src.mortgage import Mortgage
        # Get balance at refinance month from ARM schedule
        if stored_refi_params['refinance_month'] > 1:
            refi_balance = arm_schedule[
                arm_schedule['month'] == stored_refi_params['refinance_month'] - 1
            ]['balance'].values[0]
        else:
            refi_balance = stored_arm.principal

        fixed_mortgage = Mortgage(
            refi_balance,
            stored_refi_params['fixed_rate'],
            stored_refi_params['term_months']
        )
        fixed_schedule = fixed_mortgage.amortization_schedule()

        # Display the comparison table
        display_arm_vs_refi_schedule_comparison(
            arm_schedule=arm_schedule,
            fixed_schedule=fixed_schedule,
            refinance_month=stored_refi_params['refinance_month'],
            refinance_costs=stored_refi_params['refinance_costs'],
            key_prefix="shotwell_schedule",
        )


def data_management_page():
    """Import/export scenarios and load examples."""
    st.header("Data Management")

    tab1, tab2, tab3 = st.tabs(["Export Scenario", "Import Scenario", "Example Scenarios"])

    with tab1:
        st.subheader("Export Current Scenario")

        name = st.text_input("Scenario Name", value="My Mortgage Scenario")
        description = st.text_area("Description", value="")

        if st.button("Export to JSON"):
            # Gather current state
            scenario = Scenario(
                name=name,
                description=description,
            )

            if 'current_mortgage' in st.session_state:
                scenario.mortgage = mortgage_to_dict(st.session_state['current_mortgage'])

            # Create downloadable JSON
            import json
            from datetime import datetime

            data = {
                'name': scenario.name,
                'description': scenario.description,
                'mortgage': scenario.mortgage,
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
            }

            json_str = json.dumps(data, indent=2)

            st.download_button(
                "Download JSON",
                json_str,
                f"{name.lower().replace(' ', '_')}.json",
                "application/json",
            )

    with tab2:
        st.subheader("Import Scenario")

        uploaded_file = st.file_uploader("Choose a JSON file", type="json")

        if uploaded_file is not None:
            import json
            data = json.load(uploaded_file)

            st.success(f"Loaded scenario: {data.get('name', 'Unknown')}")
            st.json(data)

            if st.button("Apply Scenario"):
                if data.get('mortgage'):
                    from src.export import dict_to_mortgage
                    st.session_state['current_mortgage'] = dict_to_mortgage(data['mortgage'])
                    st.success("Scenario applied! Go to Mortgage Calculator to view.")

    with tab3:
        st.subheader("Example Scenarios")

        examples = {
            "First-Time Homebuyer": {
                "description": "Typical 30-year fixed mortgage for a first home",
                "mortgage": {
                    "principal": 1021440,
                    "annual_rate": 0.04875,
                    "term_months": 360,
                },
            },
            "Refinance Opportunity": {
                "description": "Homeowner with older high-rate mortgage considering refinance",
                "mortgage": {
                    "principal": 1021440,
                    "annual_rate": 0.04875,
                    "term_months": 360,
                },
            },
            "15-Year Payoff": {
                "description": "Aggressive payoff with 15-year term",
                "mortgage": {
                    "principal": 1021440,
                    "annual_rate": 0.04875,
                    "term_months": 180,
                },
            },
        }

        for name, example in examples.items():
            with st.expander(name):
                st.markdown(example["description"])
                st.json(example["mortgage"])

                if st.button(f"Load {name}", key=f"load_{name}"):
                    from src.export import dict_to_mortgage
                    st.session_state['current_mortgage'] = dict_to_mortgage(example["mortgage"])
                    st.success(f"Loaded '{name}'! Go to Mortgage Calculator to view.")


if __name__ == "__main__":
    main()
