"""Plotly chart components for mortgage visualization."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Optional


def create_amortization_chart(schedule: pd.DataFrame) -> go.Figure:
    """Create interactive amortization chart showing balance, principal, and interest over time."""
    fig = go.Figure()

    # Add balance line
    fig.add_trace(go.Scatter(
        x=schedule['month'],
        y=schedule['balance'],
        name='Remaining Balance',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Month %{x}<br>Balance: $%{y:,.0f}<extra></extra>',
    ))

    # Add cumulative principal area
    fig.add_trace(go.Scatter(
        x=schedule['month'],
        y=schedule['cumulative_principal'],
        name='Principal Paid',
        fill='tozeroy',
        line=dict(color='#2ca02c', width=1),
        fillcolor='rgba(44, 160, 44, 0.3)',
        hovertemplate='Month %{x}<br>Principal Paid: $%{y:,.0f}<extra></extra>',
    ))

    # Add cumulative interest
    fig.add_trace(go.Scatter(
        x=schedule['month'],
        y=schedule['cumulative_interest'],
        name='Interest Paid',
        line=dict(color='#d62728', width=2, dash='dash'),
        hovertemplate='Month %{x}<br>Interest Paid: $%{y:,.0f}<extra></extra>',
    ))

    fig.update_layout(
        title='Loan Amortization Over Time',
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        hovermode='x unified',
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
        ),
        yaxis=dict(tickformat='$,.0f'),
    )

    return fig


def create_payment_breakdown_chart(schedule: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart showing principal vs interest in each payment."""
    # Sample for performance (show every 12th month for 30-year loan)
    step = max(1, len(schedule) // 30)
    sampled = schedule.iloc[::step].copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=sampled['month'],
        y=sampled['principal'],
        name='Principal',
        marker_color='#2ca02c',
        hovertemplate='Month %{x}<br>Principal: $%{y:,.2f}<extra></extra>',
    ))

    fig.add_trace(go.Bar(
        x=sampled['month'],
        y=sampled['interest'],
        name='Interest',
        marker_color='#d62728',
        hovertemplate='Month %{x}<br>Interest: $%{y:,.2f}<extra></extra>',
    ))

    fig.update_layout(
        title='Payment Breakdown: Principal vs Interest',
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        barmode='stack',
        hovermode='x unified',
        yaxis=dict(tickformat='$,.0f'),
    )

    return fig


def create_equity_chart(schedule: pd.DataFrame, home_value: Optional[float] = None) -> go.Figure:
    """Create equity buildup chart.

    If home_value is provided, shows equity as percentage of home value.
    """
    fig = go.Figure()

    equity = schedule['cumulative_principal'].values
    if home_value:
        equity_pct = (equity / home_value) * 100
        fig.add_trace(go.Scatter(
            x=schedule['month'],
            y=equity_pct,
            name='Equity %',
            fill='tozeroy',
            line=dict(color='#17becf', width=2),
            fillcolor='rgba(23, 190, 207, 0.3)',
            hovertemplate='Month %{x}<br>Equity: %{y:.1f}%<extra></extra>',
        ))
        fig.update_layout(yaxis_title='Equity (% of Home Value)')
    else:
        fig.add_trace(go.Scatter(
            x=schedule['month'],
            y=equity,
            name='Equity',
            fill='tozeroy',
            line=dict(color='#17becf', width=2),
            fillcolor='rgba(23, 190, 207, 0.3)',
            hovertemplate='Month %{x}<br>Equity: $%{y:,.0f}<extra></extra>',
        ))
        fig.update_layout(
            yaxis_title='Equity ($)',
            yaxis=dict(tickformat='$,.0f'),
        )

    fig.update_layout(
        title='Equity Buildup Over Time',
        xaxis_title='Month',
        hovermode='x unified',
    )

    return fig


def create_refinance_break_even_chart(break_even_data: pd.DataFrame) -> go.Figure:
    """Create break-even analysis chart for refinancing."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=break_even_data['month'],
        y=break_even_data['current_cumulative'],
        name='Current Loan',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Month %{x}<br>Cumulative: $%{y:,.0f}<extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=break_even_data['month'],
        y=break_even_data['new_cumulative'],
        name='Refinanced Loan',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='Month %{x}<br>Cumulative: $%{y:,.0f}<extra></extra>',
    ))

    # Find break-even point
    for i, row in break_even_data.iterrows():
        if row['savings'] > 0:
            break_even_month = row['month']
            fig.add_vline(
                x=break_even_month,
                line_dash='dash',
                line_color='gray',
                annotation_text=f'Break-even: Month {break_even_month}',
            )
            break

    fig.update_layout(
        title='Cumulative Cost Comparison',
        xaxis_title='Month',
        yaxis_title='Cumulative Payments ($)',
        hovermode='x unified',
        yaxis=dict(tickformat='$,.0f'),
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
        ),
    )

    return fig


def create_payoff_comparison_chart(
    original_schedule: pd.DataFrame,
    modified_schedule: pd.DataFrame,
    labels: tuple = ('Original', 'With Extra Payments'),
) -> go.Figure:
    """Create comparison chart for payoff strategies."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=original_schedule['month'],
        y=original_schedule['balance'],
        name=labels[0],
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Month %{x}<br>Balance: $%{y:,.0f}<extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=modified_schedule['month'],
        y=modified_schedule['balance'],
        name=labels[1],
        line=dict(color='#2ca02c', width=2),
        hovertemplate='Month %{x}<br>Balance: $%{y:,.0f}<extra></extra>',
    ))

    # Add payoff markers
    orig_payoff = original_schedule['month'].iloc[-1]
    mod_payoff = modified_schedule['month'].iloc[-1]

    fig.add_vline(x=orig_payoff, line_dash='dot', line_color='#1f77b4', opacity=0.5)
    fig.add_vline(x=mod_payoff, line_dash='dot', line_color='#2ca02c', opacity=0.5)

    months_saved = orig_payoff - mod_payoff
    fig.add_annotation(
        x=(orig_payoff + mod_payoff) / 2,
        y=original_schedule['balance'].max() * 0.5,
        text=f'{months_saved} months saved',
        showarrow=False,
        font=dict(size=14),
    )

    fig.update_layout(
        title='Balance Payoff Comparison',
        xaxis_title='Month',
        yaxis_title='Remaining Balance ($)',
        hovermode='x unified',
        yaxis=dict(tickformat='$,.0f'),
    )

    return fig


def create_monte_carlo_fan_chart(fan_data: pd.DataFrame) -> go.Figure:
    """Create fan chart showing rate path confidence intervals."""
    fig = go.Figure()

    # Add confidence bands (outer to inner for proper layering)
    fig.add_trace(go.Scatter(
        x=list(fan_data['month']) + list(fan_data['month'][::-1]),
        y=list(fan_data['p95']) + list(fan_data['p5'][::-1]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(width=0),
        name='90% CI',
        hoverinfo='skip',
    ))

    fig.add_trace(go.Scatter(
        x=list(fan_data['month']) + list(fan_data['month'][::-1]),
        y=list(fan_data['p75']) + list(fan_data['p25'][::-1]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.4)',
        line=dict(width=0),
        name='50% CI',
        hoverinfo='skip',
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=fan_data['month'],
        y=fan_data['p50'],
        name='Median',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Month %{x}<br>Rate: %{y:.3%}<extra></extra>',
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=fan_data['month'],
        y=fan_data['mean'],
        name='Mean',
        line=dict(color='#d62728', width=2, dash='dash'),
        hovertemplate='Month %{x}<br>Rate: %{y:.3%}<extra></extra>',
    ))

    fig.update_layout(
        title='Projected Interest Rate Paths',
        xaxis_title='Month',
        yaxis_title='Interest Rate',
        hovermode='x unified',
        yaxis=dict(tickformat='.3%', range=[0, 0.10]),
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
        ),
    )

    return fig


def create_monte_carlo_histogram(
    values: np.ndarray,
    title: str = 'Distribution',
    xlabel: str = 'Value',
    show_percentiles: bool = True,
) -> go.Figure:
    """Create histogram of Monte Carlo outcomes."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=50,
        name='Distribution',
        marker_color='#1f77b4',
        opacity=0.7,
    ))

    if show_percentiles:
        p5 = np.percentile(values, 5)
        p50 = np.percentile(values, 50)
        p95 = np.percentile(values, 95)

        for pval, pname, color in [(p5, '5th', '#2ca02c'), (p50, '50th', '#ff7f0e'), (p95, '95th', '#d62728')]:
            fig.add_vline(
                x=pval,
                line_dash='dash',
                line_color=color,
                annotation_text=f'{pname}: ${pval:,.0f}',
                annotation_position='top',
            )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title='Count',
        showlegend=False,
        xaxis=dict(tickformat='$,.0f'),
    )

    return fig


def create_arm_vs_fixed_comparison_chart(
    arm_schedule: pd.DataFrame,
    fixed_schedule: pd.DataFrame,
) -> go.Figure:
    """Create comparison chart for ARM vs fixed rate mortgage."""
    fig = go.Figure()

    # Payment comparison
    fig.add_trace(go.Scatter(
        x=arm_schedule['month'],
        y=arm_schedule['payment'],
        name='ARM Payment',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='Month %{x}<br>Payment: $%{y:,.2f}<extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=fixed_schedule['month'],
        y=fixed_schedule['payment'],
        name='Fixed Payment',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Month %{x}<br>Payment: $%{y:,.2f}<extra></extra>',
    ))

    # Add rate on secondary y-axis
    fig.add_trace(go.Scatter(
        x=arm_schedule['month'],
        y=arm_schedule['rate'] * 100,
        name='ARM Rate',
        line=dict(color='#d62728', width=1, dash='dot'),
        yaxis='y2',
        hovertemplate='Month %{x}<br>Rate: %{y:.3f}%<extra></extra>',
    ))

    fig.update_layout(
        title='ARM vs Fixed Rate Comparison',
        xaxis_title='Month',
        yaxis=dict(
            title='Monthly Payment ($)',
            tickformat='$,.0f',
        ),
        yaxis2=dict(
            title='Interest Rate (%)',
            overlaying='y',
            side='right',
            tickformat='.3f',
            range=[0, 10],
        ),
        hovermode='x unified',
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
        ),
    )

    return fig


def create_cost_distribution_comparison_chart(
    arm_total_paid: np.ndarray,
    refi_total_paid: np.ndarray,
) -> go.Figure:
    """Create overlaid histograms showing total cost distributions for ARM vs Refinance.

    Args:
        arm_total_paid: Array of total ARM costs across simulations
        refi_total_paid: Array of total refinance costs across simulations

    Returns:
        Plotly figure with overlaid histograms
    """
    fig = go.Figure()

    # ARM distribution
    fig.add_trace(go.Histogram(
        x=arm_total_paid,
        name='ARM (Stay)',
        marker_color='rgba(31, 119, 180, 0.6)',
        nbinsx=50,
        hovertemplate='ARM Total: $%{x:,.0f}<br>Count: %{y}<extra></extra>',
    ))

    # Refinance distribution
    fig.add_trace(go.Histogram(
        x=refi_total_paid,
        name='Refinance to Fixed',
        marker_color='rgba(44, 160, 44, 0.6)',
        nbinsx=50,
        hovertemplate='Refi Total: $%{x:,.0f}<br>Count: %{y}<extra></extra>',
    ))

    # Add median lines
    arm_median = np.median(arm_total_paid)
    refi_median = np.median(refi_total_paid)

    fig.add_vline(
        x=arm_median,
        line_dash='dash',
        line_color='#1f77b4',
        annotation_text=f'ARM Median: ${arm_median:,.0f}',
        annotation_position='top left',
    )

    fig.add_vline(
        x=refi_median,
        line_dash='dash',
        line_color='#2ca02c',
        annotation_text=f'Refi Median: ${refi_median:,.0f}',
        annotation_position='top right',
    )

    fig.update_layout(
        title='Total Cost Distribution: ARM vs Refinance',
        xaxis_title='Total Payments ($)',
        yaxis_title='Count',
        barmode='overlay',
        xaxis=dict(tickformat='$,.0f'),
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
        ),
    )

    return fig


def create_cumulative_cost_fan_chart(
    arm_cumulative_by_month: np.ndarray,
    refi_cumulative_by_month: np.ndarray,
    refinance_month: int,
) -> go.Figure:
    """Create fan chart showing ARM cumulative costs with confidence bands vs refinance line.

    Args:
        arm_cumulative_by_month: Array of shape (n_simulations, n_months) with ARM cumulative costs
        refi_cumulative_by_month: Array of shape (n_simulations, n_months) with refi cumulative costs
        refinance_month: Month at which refinancing occurs

    Returns:
        Plotly figure with fan chart
    """
    n_months = arm_cumulative_by_month.shape[1]
    months = list(range(1, n_months + 1))

    fig = go.Figure()

    # Calculate ARM percentiles
    arm_p5 = np.percentile(arm_cumulative_by_month, 5, axis=0)
    arm_p25 = np.percentile(arm_cumulative_by_month, 25, axis=0)
    arm_p50 = np.percentile(arm_cumulative_by_month, 50, axis=0)
    arm_p75 = np.percentile(arm_cumulative_by_month, 75, axis=0)
    arm_p95 = np.percentile(arm_cumulative_by_month, 95, axis=0)

    # 90% CI band for ARM
    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=list(arm_p95) + list(arm_p5[::-1]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(width=0),
        name='ARM 90% CI',
        hoverinfo='skip',
    ))

    # 50% CI band for ARM
    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=list(arm_p75) + list(arm_p25[::-1]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.4)',
        line=dict(width=0),
        name='ARM 50% CI',
        hoverinfo='skip',
    ))

    # ARM median line
    fig.add_trace(go.Scatter(
        x=months,
        y=arm_p50,
        name='ARM Median',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Month %{x}<br>ARM Cumulative: $%{y:,.0f}<extra></extra>',
    ))

    # Refinance path - use median (deterministic given fixed rate)
    refi_median = np.median(refi_cumulative_by_month, axis=0)
    fig.add_trace(go.Scatter(
        x=months,
        y=refi_median,
        name='Refinance Path',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='Month %{x}<br>Refi Cumulative: $%{y:,.0f}<extra></extra>',
    ))

    # Vertical line at refinance point
    fig.add_vline(
        x=refinance_month,
        line_dash='dash',
        line_color='gray',
        annotation_text=f'Refinance (Month {refinance_month})',
        annotation_position='top',
    )

    fig.update_layout(
        title='Cumulative Payments Over Time',
        xaxis_title='Month',
        yaxis_title='Cumulative Payments ($)',
        hovermode='x unified',
        yaxis=dict(tickformat='$,.0f'),
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
        ),
    )

    return fig


def create_savings_waterfall_chart(
    original_interest: float,
    new_interest: float,
    closing_costs: float,
    points_cost: float,
) -> go.Figure:
    """Create waterfall chart showing refinance savings breakdown."""
    fig = go.Figure(go.Waterfall(
        name='Savings Analysis',
        orientation='v',
        measure=['absolute', 'relative', 'relative', 'relative', 'total'],
        x=['Original Interest', 'Interest Saved', 'Closing Costs', 'Points', 'Net Savings'],
        y=[
            original_interest,
            -(original_interest - new_interest),
            closing_costs,
            points_cost,
            0,  # Total calculated automatically
        ],
        connector=dict(line=dict(color='rgb(63, 63, 63)')),
        decreasing=dict(marker=dict(color='#2ca02c')),
        increasing=dict(marker=dict(color='#d62728')),
        totals=dict(marker=dict(color='#1f77b4')),
        textposition='outside',
        text=[
            f'${original_interest:,.0f}',
            f'-${original_interest - new_interest:,.0f}',
            f'+${closing_costs:,.0f}',
            f'+${points_cost:,.0f}',
            '',
        ],
    ))

    fig.update_layout(
        title='Refinance Savings Breakdown',
        showlegend=False,
        yaxis=dict(tickformat='$,.0f'),
    )

    return fig
