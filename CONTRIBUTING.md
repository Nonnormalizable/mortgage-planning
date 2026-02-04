# Contributing to Mortgage Planning Tool

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Development Environment Setup

### Prerequisites

- Python 3.11 or higher
- Git

### Setup Steps

1. Fork the repository on GitHub

2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mortgage-planning.git
   cd mortgage-planning
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies (including dev dependencies):
   ```bash
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

6. Verify setup by running tests:
   ```bash
   python -m pytest
   ```

## Running the Application

```bash
streamlit run app.py
```

## Running Tests

Run the full test suite:

```bash
python -m pytest
```

Run with verbose output:

```bash
python -m pytest -v
```

Run a specific test file:

```bash
python -m pytest tests/test_mortgage.py
```

## Code Style Guidelines

### Type Hints

All functions must include type hints for parameters and return values:

```python
def calculate_payment(principal: float, rate: float, months: int) -> float:
    """Calculate monthly mortgage payment."""
    ...
```

### Docstrings

Include docstrings for all public functions and classes:

```python
def calculate_refinance_savings(
    current_payment: float,
    new_payment: float,
    closing_costs: float,
) -> dict[str, float]:
    """Calculate savings from refinancing.

    Args:
        current_payment: Current monthly payment amount.
        new_payment: New monthly payment after refinancing.
        closing_costs: Total refinancing costs.

    Returns:
        Dictionary containing monthly_savings and break_even_months.
    """
    ...
```

### Formatting and Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Pre-commit hooks will automatically check your code.

To manually run linting:

```bash
ruff check .
```

To auto-fix issues:

```bash
ruff check --fix .
```

### General Guidelines

- Follow existing patterns and conventions in the codebase
- Keep functions focused and single-purpose
- Write self-documenting code with clear variable names
- Add tests for new functionality

## Pull Request Process

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines above

3. **Add tests** for any new functionality

4. **Run the test suite** to ensure everything passes:
   ```bash
   python -m pytest
   ```

5. **Commit your changes** with a clear, descriptive message:
   ```bash
   git commit -m "Add feature: brief description of changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub with:
   - A clear title describing the change
   - A description of what was changed and why
   - Reference to any related issues

### PR Requirements

- All tests must pass
- Code must pass linting checks
- New features should include tests
- Documentation should be updated if needed

## Reporting Issues

### Bug Reports

When reporting a bug, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Python version and operating system
- Any relevant error messages or screenshots

### Feature Requests

For feature requests, please include:

- A clear description of the proposed feature
- The problem it would solve or use case it enables
- Any ideas for implementation (optional)

## Commit Message Conventions

Use clear, descriptive commit messages:

- Start with a verb in present tense: "Add", "Fix", "Update", "Remove"
- Keep the first line under 72 characters
- Reference issue numbers when applicable: "Fix #123: description"

Examples:
- `Add Monte Carlo simulation for ARM analysis`
- `Fix break-even calculation when closing costs are zero`
- `Update documentation for refinance comparison`

## Questions?

If you have questions about contributing, feel free to open an issue for discussion.
