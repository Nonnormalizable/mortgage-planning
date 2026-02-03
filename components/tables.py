"""Streamlit table display components."""

import streamlit as st
import pandas as pd
from typing import Optional


def display_amortization_table(
    schedule: pd.DataFrame,
    title: str = "Amortization Schedule",
    show_yearly: bool = True,
    max_rows: int = 50,
) -> None:
    """Display amortization schedule with formatting.

    Args:
        schedule: DataFrame with amortization data
        title: Table title
        show_yearly: If True, show yearly summary instead of monthly
        max_rows: Maximum rows to display at once
    """
    st.subheader(title)

    # View toggle
    view_type = st.radio(
        "View",
        options=["Yearly Summary", "Monthly Detail"],
        horizontal=True,
        key=f"table_view_{title}",
    )

    if view_type == "Yearly Summary":
        # Aggregate to yearly
        schedule_copy = schedule.copy()
        schedule_copy['year'] = ((schedule_copy['month'] - 1) // 12) + 1

        yearly = schedule_copy.groupby('year').agg({
            'payment': 'sum',
            'principal': 'sum',
            'interest': 'sum',
            'balance': 'last',
        }).reset_index()

        yearly.columns = ['Year', 'Total Payments', 'Principal Paid', 'Interest Paid', 'End Balance']

        # Format currency columns
        display_df = yearly.copy()
        for col in ['Total Payments', 'Principal Paid', 'Interest Paid', 'End Balance']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        # Show monthly with pagination
        total_rows = len(schedule)

        if total_rows > max_rows:
            col1, col2 = st.columns([3, 1])
            with col1:
                start_month = st.slider(
                    "Start from month",
                    min_value=1,
                    max_value=total_rows - max_rows + 1,
                    value=1,
                    key=f"month_slider_{title}",
                )
            with col2:
                st.write(f"Showing {max_rows} of {total_rows} months")

            display_slice = schedule.iloc[start_month - 1:start_month - 1 + max_rows].copy()
        else:
            display_slice = schedule.copy()

        # Rename columns for display
        display_df = display_slice.rename(columns={
            'month': 'Month',
            'payment': 'Payment',
            'principal': 'Principal',
            'interest': 'Interest',
            'balance': 'Balance',
            'cumulative_interest': 'Total Interest',
        })

        # Select columns to show
        cols_to_show = ['Month', 'Payment', 'Principal', 'Interest', 'Balance']
        if 'Total Interest' in display_df.columns:
            cols_to_show.append('Total Interest')

        display_df = display_df[cols_to_show]

        # Format currency
        for col in cols_to_show[1:]:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


def display_comparison_table(
    comparisons: pd.DataFrame,
    title: str = "Comparison",
    highlight_best: Optional[str] = None,
) -> None:
    """Display comparison table with optional highlighting.

    Args:
        comparisons: DataFrame with comparison data
        title: Table title
        highlight_best: Column name to highlight best value (lowest)
    """
    st.subheader(title)

    display_df = comparisons.copy()

    # Format numeric columns
    for col in display_df.columns:
        if display_df[col].dtype in ['float64', 'int64']:
            if 'rate' in col.lower() or 'ratio' in col.lower():
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
            elif any(word in col.lower() for word in ['cost', 'payment', 'savings', 'interest', 'principal']):
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
            elif 'month' in col.lower():
                display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


def display_payoff_strategy_table(strategies: pd.DataFrame) -> None:
    """Display payoff strategy comparison table."""
    st.subheader("Payoff Strategy Comparison")

    display_df = strategies.copy()

    # Rename columns
    display_df = display_df.rename(columns={
        'strategy': 'Strategy',
        'extra_monthly': 'Extra/Month',
        'term_months': 'Payoff (Months)',
        'total_interest': 'Total Interest',
        'interest_saved': 'Interest Saved',
        'months_saved': 'Months Saved',
    })

    # Format
    display_df['Extra/Month'] = display_df['Extra/Month'].apply(lambda x: f"${x:,.0f}")
    display_df['Total Interest'] = display_df['Total Interest'].apply(lambda x: f"${x:,.0f}")
    display_df['Interest Saved'] = display_df['Interest Saved'].apply(lambda x: f"${x:,.0f}")

    # Add years column
    display_df['Years Saved'] = (display_df['Months Saved'] / 12).round(1)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


def display_monte_carlo_stats(stats: pd.DataFrame) -> None:
    """Display Monte Carlo simulation statistics."""
    st.subheader("Simulation Statistics")

    display_df = stats.copy()

    # Format based on metric type
    formatted_rows = []
    for _, row in display_df.iterrows():
        metric = row['metric']
        formatted_row = {'Metric': metric.replace('_', ' ').title()}

        for col in ['mean', 'median', 'std', 'p5', 'p25', 'p75', 'p95', 'min', 'max']:
            if col in row:
                val = row[col]
                if 'rate' in metric:
                    formatted_row[col.upper()] = f"{val:.2%}"
                else:
                    formatted_row[col.upper()] = f"${val:,.0f}"

        formatted_rows.append(formatted_row)

    result_df = pd.DataFrame(formatted_rows)

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
    )


def display_refinance_summary(comparison: dict) -> None:
    """Display refinance comparison summary as a formatted table."""
    st.subheader("Refinance Analysis Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current Loan**")
        st.metric("Monthly Payment", f"${comparison['current_payment']:,.2f}")
        st.metric("Remaining Months", comparison['current_remaining_months'])
        st.metric("Remaining Interest", f"${comparison['current_remaining_interest']:,.2f}")

    with col2:
        st.markdown("**New Loan**")
        st.metric("Monthly Payment", f"${comparison['new_payment']:,.2f}",
                  delta=f"${-comparison['monthly_savings']:,.2f}/mo")
        st.metric("Term", f"{comparison['new_term_months']} months")
        st.metric("Total Interest", f"${comparison['new_total_interest']:,.2f}")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Upfront Costs", f"${comparison['total_upfront_costs']:,.2f}")

    with col2:
        break_even = comparison['break_even_months']
        if break_even:
            st.metric("Break-even", f"{break_even:.0f} months ({break_even/12:.1f} years)")
        else:
            st.metric("Break-even", "Never")

    with col3:
        st.metric("Net Savings", f"${comparison['net_savings']:,.2f}",
                  delta="savings" if comparison['net_savings'] > 0 else "loss")

    # Recommendation
    st.info(comparison['recommendation'])
