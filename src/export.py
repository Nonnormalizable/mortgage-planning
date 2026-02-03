"""JSON export/import functionality for scenarios."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from .mortgage import Mortgage
from .arm import ARMParameters
from .refinance import RefinanceScenario
from .payoff import ExtraPayment, LumpSumPayment, PaymentFrequency
from .monte_carlo import RateSimulationParams, RateModel


def mortgage_to_dict(mortgage: Mortgage) -> dict:
    """Convert Mortgage to serializable dictionary."""
    return {
        'type': 'fixed',
        'principal': mortgage.principal,
        'annual_rate': mortgage.annual_rate,
        'term_months': mortgage.term_months,
        'start_date': mortgage.start_date,
    }


def dict_to_mortgage(data: dict) -> Mortgage:
    """Convert dictionary to Mortgage."""
    return Mortgage(
        principal=data['principal'],
        annual_rate=data['annual_rate'],
        term_months=data['term_months'],
        start_date=data.get('start_date'),
    )


def arm_to_dict(arm: ARMParameters) -> dict:
    """Convert ARMParameters to serializable dictionary."""
    return {
        'type': 'arm',
        'principal': arm.principal,
        'initial_rate': arm.initial_rate,
        'term_months': arm.term_months,
        'initial_period_months': arm.initial_period_months,
        'adjustment_period_months': arm.adjustment_period_months,
        'initial_cap': arm.initial_cap,
        'periodic_cap': arm.periodic_cap,
        'lifetime_cap': arm.lifetime_cap,
        'lifetime_floor': arm.lifetime_floor,
        'margin': arm.margin,
        'start_date': arm.start_date,
    }


def dict_to_arm(data: dict) -> ARMParameters:
    """Convert dictionary to ARMParameters."""
    return ARMParameters(
        principal=data['principal'],
        initial_rate=data['initial_rate'],
        term_months=data['term_months'],
        initial_period_months=data['initial_period_months'],
        adjustment_period_months=data.get('adjustment_period_months', 12),
        initial_cap=data.get('initial_cap', 0.05),
        periodic_cap=data.get('periodic_cap', 0.01),
        lifetime_cap=data.get('lifetime_cap', 0.05),
        lifetime_floor=data.get('lifetime_floor', 0.0),
        margin=data.get('margin', 0.025),
        start_date=data.get('start_date'),
    )


def refinance_to_dict(refinance: RefinanceScenario) -> dict:
    """Convert RefinanceScenario to serializable dictionary."""
    return {
        'new_principal': refinance.new_principal,
        'new_rate': refinance.new_rate,
        'new_term_months': refinance.new_term_months,
        'closing_costs': refinance.closing_costs,
        'points': refinance.points,
        'cash_out': refinance.cash_out,
        'roll_costs_into_loan': refinance.roll_costs_into_loan,
    }


def dict_to_refinance(data: dict) -> RefinanceScenario:
    """Convert dictionary to RefinanceScenario."""
    return RefinanceScenario(
        new_principal=data['new_principal'],
        new_rate=data['new_rate'],
        new_term_months=data['new_term_months'],
        closing_costs=data.get('closing_costs', 0),
        points=data.get('points', 0),
        cash_out=data.get('cash_out', 0),
        roll_costs_into_loan=data.get('roll_costs_into_loan', False),
    )


def extra_payment_to_dict(ep: ExtraPayment) -> dict:
    """Convert ExtraPayment to serializable dictionary."""
    return {
        'amount': ep.amount,
        'frequency': ep.frequency.value,
        'start_month': ep.start_month,
        'end_month': ep.end_month,
    }


def dict_to_extra_payment(data: dict) -> ExtraPayment:
    """Convert dictionary to ExtraPayment."""
    return ExtraPayment(
        amount=data['amount'],
        frequency=PaymentFrequency(data.get('frequency', 'monthly')),
        start_month=data.get('start_month', 1),
        end_month=data.get('end_month'),
    )


def lump_sum_to_dict(ls: LumpSumPayment) -> dict:
    """Convert LumpSumPayment to serializable dictionary."""
    return {
        'amount': ls.amount,
        'month': ls.month,
    }


def dict_to_lump_sum(data: dict) -> LumpSumPayment:
    """Convert dictionary to LumpSumPayment."""
    return LumpSumPayment(
        amount=data['amount'],
        month=data['month'],
    )


def rate_sim_params_to_dict(params: RateSimulationParams) -> dict:
    """Convert RateSimulationParams to serializable dictionary."""
    return {
        'current_rate': params.current_rate,
        'model': params.model.value,
        'long_term_mean': params.long_term_mean,
        'mean_reversion_speed': params.mean_reversion_speed,
        'volatility': params.volatility,
        'drift': params.drift,
        'num_simulations': params.num_simulations,
        'time_horizon_months': params.time_horizon_months,
    }


def dict_to_rate_sim_params(data: dict) -> RateSimulationParams:
    """Convert dictionary to RateSimulationParams."""
    return RateSimulationParams(
        current_rate=data['current_rate'],
        model=RateModel(data.get('model', 'vasicek')),
        long_term_mean=data.get('long_term_mean', 0.04),
        mean_reversion_speed=data.get('mean_reversion_speed', 0.1),
        volatility=data.get('volatility', 0.01),
        drift=data.get('drift', 0.0),
        num_simulations=data.get('num_simulations', 1000),
        time_horizon_months=data.get('time_horizon_months', 360),
    )


@dataclass
class Scenario:
    """Complete scenario for saving/loading."""

    name: str
    description: str
    mortgage: Optional[Dict] = None
    arm: Optional[Dict] = None
    refinance_options: List[Dict] = None
    extra_payments: List[Dict] = None
    lump_sums: List[Dict] = None
    monte_carlo_params: Optional[Dict] = None
    current_month: int = 0
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if self.refinance_options is None:
            self.refinance_options = []
        if self.extra_payments is None:
            self.extra_payments = []
        if self.lump_sums is None:
            self.lump_sums = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


def export_scenario(scenario: Scenario, filepath: str) -> None:
    """Export scenario to JSON file."""
    data = {
        'name': scenario.name,
        'description': scenario.description,
        'mortgage': scenario.mortgage,
        'arm': scenario.arm,
        'refinance_options': scenario.refinance_options,
        'extra_payments': scenario.extra_payments,
        'lump_sums': scenario.lump_sums,
        'monte_carlo_params': scenario.monte_carlo_params,
        'current_month': scenario.current_month,
        'created_at': scenario.created_at,
        'updated_at': datetime.now().isoformat(),
        'version': '1.0',
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def import_scenario(filepath: str) -> Scenario:
    """Import scenario from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    return Scenario(
        name=data['name'],
        description=data.get('description', ''),
        mortgage=data.get('mortgage'),
        arm=data.get('arm'),
        refinance_options=data.get('refinance_options', []),
        extra_payments=data.get('extra_payments', []),
        lump_sums=data.get('lump_sums', []),
        monte_carlo_params=data.get('monte_carlo_params'),
        current_month=data.get('current_month', 0),
        created_at=data.get('created_at'),
        updated_at=data.get('updated_at'),
    )


def list_example_scenarios(examples_dir: str = 'data/examples') -> List[str]:
    """List available example scenario files."""
    path = Path(examples_dir)
    if not path.exists():
        return []
    return [f.stem for f in path.glob('*.json')]


def load_example_scenario(name: str, examples_dir: str = 'data/examples') -> Scenario:
    """Load an example scenario by name."""
    filepath = Path(examples_dir) / f"{name}.json"
    return import_scenario(str(filepath))
