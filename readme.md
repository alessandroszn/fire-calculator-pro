# FIRE Calculator Pro

## Description
A Python-based financial calculator that helps you determine your path to Financial Independence/Retire Early (FIRE). This tool enables you to simulate different investment scenarios, calculate your FIRE number, and visualize your journey to financial freedom based on various parameters and assumptions.

## Features
- Core FIRE Calculations:
  - Determine your FIRE number (25x, 30x, or custom multiplier of annual expenses)
  - Calculate time to reach financial independence based on savings rate
  - Support for traditional FIRE, Fat FIRE, Coast Fire and Lean FIRE approaches
  
- Advanced Simulation Options:
  - Monte Carlo simulations to account for market volatility
  - Sequence of returns risk analysis
  - Various withdrawal strategies (4% rule, variable percentage, etc.)
  - Inflation-adjusted projections

- Multiple Income & Expense Scenarios:
  - Factor in salary increases, bonuses, and side hustles
  - Account for major life expenses (home purchase, education, etc.)
  - Tax optimization strategies

- Data Visualization:
  - Interactive growth charts showing portfolio value over time
  - Withdrawal sustainability graphs
  - Success probability heatmaps based on different parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/alessandroszn/fire-calculator-pro
cd fire-calculator-pro

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

### Basic FIRE Calculation

```python
from fire_calculator import FIRECalculator

# Initialize calculator with your financial information
calc = FIRECalculator(
    current_age=30,
    annual_expenses=40000,
    current_savings=50000,
    annual_income=80000,
    savings_rate=0.4,  # 40% savings rate
    expected_return=0.07  # 7% expected investment returns
)

# Calculate basic FIRE metrics
fire_number = calc.calculate_fire_number(multiplier=25)
years_to_fire = calc.calculate_years_to_fire()

print(f"FIRE Number: ${fire_number:,.2f}")
print(f"Years to FIRE: {years_to_fire:.1f}")
```

### Scenario Comparison

```python
# Compare different scenarios
scenario1 = calc.simulate_fire_journey(savings_rate=0.4, expected_return=0.07)
scenario2 = calc.simulate_fire_journey(savings_rate=0.5, expected_return=0.07)
scenario3 = calc.simulate_fire_journey(savings_rate=0.4, expected_return=0.08)

# Visualize the comparison
calc.plot_scenarios([scenario1, scenario2, scenario3], 
                    labels=["Baseline", "Higher Savings", "Higher Returns"])
```

## Configuration Options

The calculator supports various configuration parameters:

```python
# Example configuration
config = {
    "simulation": {
        "monte_carlo_runs": 10000,
        "time_horizon_years": 60,
        "confidence_interval": 0.95
    },
    "inflation": {
        "expected_rate": 0.025,  # 2.5% annual inflation
        "variable_mode": True,
        "historical_data": "us_cpi_1913_2023.csv"
    },
    "withdrawal": {
        "strategy": "variable_percentage",  # Options: fixed, variable_percentage, guyton_klinger
        "initial_rate": 0.04,  # 4% rule
        "floor_rate": 0.025,
        "ceiling_rate": 0.055
    },
    "visualization": {
        "theme": "dark",
        "save_charts": True,
        "output_directory": "./reports"
    }
}
```

## Data Requirements

The calculator works best with the following input data:

- Financial Data:
  - Current savings and investments
  - Income streams and growth projections
  - Expected annual expenses in retirement
  - Expected investment returns and asset allocation

- Optional Data:
  - Historical market returns (provided with the tool)
  - Inflation data (provided with the tool)
  - Social security/pension estimates

## Roadmap

Future enhancements planned:

- Portfolio optimization suggestions
- Tax-efficient withdrawal planning
- Geographic arbitrage calculator
- Integration with actual investment account data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.