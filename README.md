# Mortgage Planning Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/Nonnormalizable/mortgage-planning/actions/workflows/ci.yml/badge.svg)](https://github.com/Nonnormalizable/mortgage-planning/actions/workflows/ci.yml)

An interactive Streamlit application for mortgage planning, refinancing analysis, and financial decision-making. This educational tool helps users understand the complexities of mortgage decisions through visualization and Monte Carlo simulation.

## Features

The application includes seven integrated tools:

- **Mortgage Calculator** — Calculate monthly payments and view complete amortization schedules for fixed-rate mortgages with interactive charts
- **ARM Analysis** — Analyze adjustable-rate mortgages (ARMs) with worst/best case scenarios and comparison to fixed-rate alternatives
- **Refinance Comparison** — Compare current mortgages to refinancing options with break-even analysis
- **Payoff Strategies** — Explore how extra payments, bi-weekly payments, and lump sums can reduce loan terms and save interest
- **Monte Carlo Simulation** — Model uncertain future interest rates using stochastic models (Vasicek, CIR, GBM, jump-diffusion) to understand the range of possible ARM outcomes
- **Shotwell Refinance** — Specialized comparison of ARM vs refinance-to-fixed strategies with Monte Carlo simulation
- **Data Management** — Export/import scenarios as JSON for saving and sharing analyses

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Nonnormalizable/mortgage-planning.git
   cd mortgage-planning
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

Run the application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
mortgage-planning/
├── app.py                 # Main Streamlit application
├── src/                   # Core calculation modules
│   ├── mortgage.py        # Fixed-rate mortgage calculations
│   ├── arm.py             # Adjustable-rate mortgage logic
│   ├── refinance.py       # Refinance comparison calculations
│   ├── payoff.py          # Extra payment and payoff strategies
│   ├── monte_carlo.py     # Monte Carlo simulation engine
│   └── export.py          # Scenario import/export utilities
├── components/            # Streamlit UI components
│   ├── inputs.py          # Input forms and widgets
│   ├── tables.py          # Data display tables
│   └── charts.py          # Plotly visualization charts
├── tests/                 # Test suite
│   ├── test_mortgage.py
│   ├── test_arm.py
│   ├── test_refinance.py
│   └── test_monte_carlo.py
├── requirements.txt       # Python dependencies
└── pyproject.toml         # Project configuration
```

## Technology Stack

- **[Streamlit](https://streamlit.io/)** — Web application framework
- **[NumPy](https://numpy.org/)** — Numerical computing
- **[Pandas](https://pandas.pydata.org/)** — Data manipulation and analysis
- **[Plotly](https://plotly.com/python/)** — Interactive visualizations

## Running Tests

```bash
python -m pytest
```

For verbose output:

```bash
python -m pytest -v
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Disclaimer

This tool is for **educational purposes only**. It is not financial advice. Mortgage decisions involve many factors not modeled here. Always consult with qualified financial professionals before making mortgage decisions.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
