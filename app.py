"""Mortgage Planning & Refinancing Tool - Streamlit Application."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Import core modules
from src.mortgage import Mortgage, calculate_affordability
from src.arm import ARMParameters, generate_arm_schedule, calculate_arm_worst_case, compare_arm_to_fixed
from src.refinance import (
    RefinanceScenario,
    calculate_refinance_comparison,
    generate_break_even_chart_data,
)
from src.payoff import (
    calculate_payoff_with_extra_payments,
    calculate_biweekly_schedule,
    compare_payoff_strategies,
)
from src.monte_carlo import (
    RateSimulationParams,
    simulate_rate_paths,
    simulate_arm_outcomes,
    calculate_simulation_statistics,
    generate_fan_chart_data,
    compare_arm_vs_fixed_monte_carlo,
)
from src.export import (
    Scenario,
    export_scenario,
    import_scenario,
    mortgage_to_dict,
    dict_to_mortgage,
    arm_to_dict,
    refinance_to_dict,
    extra_payment_to_dict,
    rate_sim_params_to_dict,
)

# Import UI components
from components.inputs import (
    mortgage_input_form,
    arm_input_form,
    refinance_input_form,
    extra_payment_input,
    monte_carlo_input_form,
    current_loan_status_input,
)
from components.tables import (
    display_amortization_table,
    display_payoff_strategy_table,
    display_monte_carlo_stats,
    display_refinance_summary,
)
from components.charts import (
    create_amortization_chart,
    create_payment_breakdown_chart,
    create_equity_chart,
    create_refinance_break_even_chart,
    create_payoff_comparison_chart,
    create_monte_carlo_fan_chart,
    create_monte_carlo_histogram,
    create_arm_vs_fixed_comparison_chart,
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
            "Data Management",
        ],
    )

    st.sidebar.divider()

    # Quick info
    st.sidebar.markdown("### Quick Reference")
    st.sidebar.markdown("""
    - **Amortization**: How loans are paid off over time
    - **ARM**: Adjustable Rate Mortgage - rate changes after initial period
    - **Break-even**: When refinance savings exceed costs
    - **Monte Carlo**: Probabilistic simulation of future scenarios
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
                    st.metric("Maximum Rate", f"{worst['max_rate']:.2%}")

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
                            f"${summary['interest_saved']:,.0f}",
                        )

                    with col_c:
                        st.metric(
                            "ROI on Extra Payments",
                            f"{summary['roi_on_extra']:.0f}%",
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

            tab1, tab2, tab3 = st.tabs(["Rate Paths", "Cost Distribution", "ARM vs Fixed"])

            with tab1:
                st.subheader("Projected Interest Rate Paths")
                fan_data = generate_fan_chart_data(rate_paths)
                fig = create_monte_carlo_fan_chart(fan_data)
                st.plotly_chart(fig, use_container_width=True)

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
                        f"{comparison['probability_arm_saves_money']:.1%}",
                    )
                    st.metric(
                        "Expected Savings with ARM",
                        f"${comparison['expected_savings_with_arm']:,.0f}",
                    )

                with col_b:
                    st.metric(
                        "ARM Max Payment (95th percentile)",
                        f"${comparison['arm_max_payment_p95']:,.0f}",
                    )
                    st.metric(
                        "Fixed Payment",
                        f"${comparison['fixed_payment']:,.0f}",
                    )

                st.markdown(f"""
                **Analysis Summary:**
                - There is a **{comparison['probability_arm_saves_money']:.1%}** chance the ARM will cost less than the {fixed_rate}% fixed rate.
                - In the best 5% of scenarios, you save at least **${comparison['savings_p95']:,.0f}** with the ARM.
                - In the worst 5% of scenarios, the ARM costs **${-comparison['savings_p5']:,.0f}** more than fixed.
                """)


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
