#!/usr/bin/env python3
"""
FIRE Calculator Pro - Financial Independence/Retire Early Calculator
A powerful tool to simulate and visualize your journey to financial independence.
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.ticker as mtick
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('fire_calculator')

class FIRECalculator:
    """
    Financial Independence Retire Early (FIRE) Calculator
    
    Calculates various FIRE metrics and simulates retirement scenarios
    based on personal financial parameters and market assumptions.
    """
    
    def __init__(self, 
                current_age: int,
                annual_expenses: float,
                current_savings: float,
                annual_income: float,
                savings_rate: float,
                expected_return: float = 0.07,
                inflation_rate: float = 0.025,
                withdrawal_rate: float = 0.04,
                tax_rate: float = 0.25):
        """
        Initialize the FIRE Calculator with personal financial parameters.
        
        Args:
            current_age: Current age in years
            annual_expenses: Annual expenses in dollars
            current_savings: Current savings/investments in dollars
            annual_income: Annual gross income in dollars
            savings_rate: Percentage of income saved (0.0 to 1.0)
            expected_return: Expected annual investment return (default 7%)
            inflation_rate: Expected annual inflation rate (default 2.5%)
            withdrawal_rate: Safe withdrawal rate in retirement (default 4%)
            tax_rate: Effective tax rate (default 25%)
        """
        self.current_age = current_age
        self.annual_expenses = annual_expenses
        self.current_savings = current_savings
        self.annual_income = annual_income
        self.savings_rate = savings_rate
        self.expected_return = expected_return
        self.inflation_rate = inflation_rate
        self.withdrawal_rate = withdrawal_rate
        self.tax_rate = tax_rate
        
        # Default configuration
        self.config = {
            "simulation_settings": {
                "monte_carlo_runs": 5000,
                "time_horizon_years": 60,
                "confidence_interval": 0.95,
                "standard_deviation": 0.15
            },
            "withdrawal_strategies": {
                "strategy": "fixed",
                "initial_rate": 0.04,
                "floor_rate": 0.025,
                "ceiling_rate": 0.055,
                "variable_adjustment_cap": 0.10
            },
            "visualization": {
                "theme": "default",
                "save_charts": False,
                "show_plots": True,
                "output_directory": "./reports",
                "chart_style": "seaborn-v0_8-darkgrid"
            },
            "run_analyses": {
                "simulate_journey": True,
                "monte_carlo": True,
                "withdrawal_strategies": True,
                "alternative_fire": True,
                "generate_report": True
            },
            "scenario_analysis": {
                "compare_savings_rates": [0.30, 0.40, 0.50],
                "coast_fire_retirement_age": 65,
                "fat_fire_expense_multiplier": 1.5,
                "lean_fire_expense_multiplier": 0.7
            }
        }
        
        # Set matplotlib style based on config
        plt.style.use(self.config["visualization"].get("chart_style", "default"))
    
    @classmethod
    def from_json(cls, json_file: str):
        """
        Create a FIRECalculator instance from a JSON configuration file.
        
        Args:
            json_file: Path to the JSON configuration file
            
        Returns:
            FIRECalculator instance
        """
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
            
            personal_info = config.get('personal_info', {})
            
            # Create calculator instance with personal info
            calculator = cls(
                current_age=personal_info.get('current_age', 30),
                annual_expenses=personal_info.get('annual_expenses', 40000),
                current_savings=personal_info.get('current_savings', 50000),
                annual_income=personal_info.get('annual_income', 80000),
                savings_rate=personal_info.get('savings_rate', 0.4),
                expected_return=personal_info.get('expected_return', 0.07),
                inflation_rate=personal_info.get('inflation_rate', 0.025),
                withdrawal_rate=personal_info.get('withdrawal_rate', 0.04),
                tax_rate=personal_info.get('tax_rate', 0.25)
            )
            
            # Apply any other configuration from the file
            for key in config:
                if key != 'personal_info':
                    calculator.config[key] = config[key]
            
            # Set matplotlib style based on config
            plt.style.use(calculator.config["visualization"].get("chart_style", "default"))
            
            return calculator
            
        except FileNotFoundError:
            logger.error(f"Settings file '{json_file}' not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            logger.error(f"Settings file '{json_file}' contains invalid JSON.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            sys.exit(1)

    def calculate_fire_number(self, multiplier: float = None) -> float:
        """
        Calculate the FIRE number (target portfolio size for retirement).
        
        Args:
            multiplier: Optional custom multiplier (e.g., 25x, 30x)
                       If None, uses 1/withdrawal_rate
        
        Returns:
            FIRE number in dollars
        """
        if multiplier is None:
            # Default: use the inverse of withdrawal rate (e.g., 4% -> 25x)
            multiplier = 1 / self.withdrawal_rate
            
        return self.annual_expenses * multiplier
    
    def calculate_years_to_fire(self, fire_number: float = None) -> float:
        """
        Calculate estimated years to reach FIRE.
        
        Args:
            fire_number: Optional custom FIRE number
                        If None, calculates using calculate_fire_number()
        
        Returns:
            Number of years to reach FIRE
        """
        # Use provided FIRE number or calculate it
        if fire_number is None:
            fire_number = self.calculate_fire_number()
        
        # Calculate annual contributions
        annual_contribution = self.annual_income * self.savings_rate
        
        # Set up variables for the calculation
        years = 0
        portfolio = self.current_savings
        
        # Simulate growth until we reach the FIRE number
        while portfolio < fire_number:
            portfolio = portfolio * (1 + self.expected_return) + annual_contribution
            years += 1
            
            # Safety cutoff (unlikely but prevents infinite loop)
            if years > 100:
                return float('inf')
        
        return years
    
    def simulate_fire_journey(self, 
                             starting_age: int = None,
                             starting_savings: float = None,
                             savings_rate: float = None,
                             expected_return: float = None,
                             include_salary_growth: bool = False,
                             salary_growth_rate: float = 0.03) -> pd.DataFrame:
        """
        Simulate the journey to FIRE year by year.
        
        Args:
            starting_age: Optional override for current age
            starting_savings: Optional override for current savings
            savings_rate: Optional override for savings rate
            expected_return: Optional override for expected return
            include_salary_growth: Whether to include annual salary increases
            salary_growth_rate: Annual salary growth rate (if enabled)
            
        Returns:
            DataFrame with yearly portfolio values and related metrics
        """
        # Use provided parameters or instance defaults
        starting_age = starting_age if starting_age is not None else self.current_age
        starting_savings = starting_savings if starting_savings is not None else self.current_savings
        savings_rate = savings_rate if savings_rate is not None else self.savings_rate
        expected_return = expected_return if expected_return is not None else self.expected_return
        
        # Calculate annual contribution (initial)
        annual_income = self.annual_income
        annual_contribution = annual_income * savings_rate
        
        # Calculate FIRE number
        fire_number = self.calculate_fire_number()
        
        # Set up variables for the simulation
        data = []
        age = starting_age
        portfolio = starting_savings
        year = 0
        fire_achieved = False
        fire_age = None
        
        # Simulate until 100 years or FIRE is achieved
        while age < 100:
            # Calculate contribution for this year
            if include_salary_growth and year > 0:
                annual_income *= (1 + salary_growth_rate)
                annual_contribution = annual_income * savings_rate
            
            # Track if FIRE is achieved this year
            if not fire_achieved and portfolio >= fire_number:
                fire_achieved = True
                fire_age = age
            
            # Record data for this year
            data.append({
                'Year': year,
                'Age': age,
                'Portfolio Value': portfolio,
                'Annual Contribution': annual_contribution,
                'FIRE Number': fire_number,
                'FIRE %': (portfolio / fire_number) * 100 if fire_number > 0 else 0,
                'FIRE Achieved': fire_achieved
            })
            
            # Grow portfolio for next year
            portfolio = portfolio * (1 + expected_return) + annual_contribution
            year += 1
            age += 1
            
            # Break if we're well past FIRE (optional)
            if fire_achieved and age > fire_age + 5:
                break
        
        # Create DataFrame from collected data
        df = pd.DataFrame(data)
        return df
    
    def run_monte_carlo_simulation(self, 
                                  num_simulations: int = None,
                                  time_horizon: int = None,
                                  mean_return: float = None,
                                  std_dev: float = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Run Monte Carlo simulation to model uncertainty in investment returns.
        
        Args:
            num_simulations: Number of simulation runs
            time_horizon: Number of years to simulate
            mean_return: Mean annual return (if None, uses expected_return)
            std_dev: Standard deviation of returns
            
        Returns:
            Tuple of (DataFrame with results, summary dictionary)
        """
        # Use config or defaults
        num_simulations = num_simulations or self.config['simulation_settings'].get('monte_carlo_runs', 5000)
        time_horizon = time_horizon or self.config['simulation_settings'].get('time_horizon_years', 60)
        mean_return = mean_return if mean_return is not None else self.expected_return
        std_dev = std_dev or self.config['simulation_settings'].get('standard_deviation', 0.15)
        
        # Calculate FIRE number
        fire_number = self.calculate_fire_number()
        
        # Calculate annual contribution
        annual_contribution = self.annual_income * self.savings_rate
        
        # Initialize arrays for simulations
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(mean_return, std_dev, (time_horizon, num_simulations))
        
        # Create portfolio value matrix
        portfolio_values = np.zeros((time_horizon + 1, num_simulations))
        portfolio_values[0, :] = self.current_savings  # Initial portfolio value
        
        # Track years to FIRE for each simulation
        years_to_fire = np.full(num_simulations, np.inf)
        
        # Run simulations
        for sim in range(num_simulations):
            for year in range(time_horizon):
                # Calculate new portfolio value
                portfolio_values[year + 1, sim] = (
                    portfolio_values[year, sim] * (1 + returns[year, sim]) + annual_contribution
                )
                
                # Record if FIRE is achieved this year (and hasn't been achieved yet)
                if portfolio_values[year + 1, sim] >= fire_number and years_to_fire[sim] == np.inf:
                    years_to_fire[sim] = year + 1
        
        # Calculate percentiles for visualization
        percentiles = [5, 25, 50, 75, 95]
        percentile_data = {}
        for p in percentiles:
            percentile_data[f'P{p}'] = np.percentile(portfolio_values, p, axis=1)
        
        # Create DataFrame with results
        years = np.arange(self.current_age, self.current_age + time_horizon + 1)
        results_df = pd.DataFrame({'Year': years, **percentile_data})
        
        # Calculate summary statistics
        valid_years = years_to_fire[years_to_fire <= time_horizon]
        fire_reached_pct = len(valid_years) / num_simulations
        
        # Years to FIRE percentiles
        years_percentiles = {}
        if len(valid_years) > 0:
            for p in [10, 25, 50, 75, 90]:
                years_percentiles[p] = np.percentile(valid_years, p) if len(valid_years) > 0 else None
        
        # Calculate success rate
        success_rate = np.mean(portfolio_values[-1, :] >= fire_number)
        
        # Create summary dictionary
        summary = {
            'success_rate': success_rate,
            'fire_reached_pct': fire_reached_pct,
            'years_to_fire': {
                'median': years_percentiles.get(50, None),
                'percentiles': years_percentiles
            },
            'final_portfolio': {
                'median': np.median(portfolio_values[-1, :]),
                'percentiles': {p: np.percentile(portfolio_values[-1, :], p) for p in percentiles}
            }
        }
        
        return results_df, summary

    def simulate_withdrawal_strategies(self, 
                                       initial_portfolio: float,
                                       years: int = 30,
                                       num_simulations: int = 1000,
                                       strategies: List[str] = None) -> Dict[str, Any]:
        """
        Simulate different withdrawal strategies in retirement.
        
        Args:
            initial_portfolio: Starting portfolio value
            years: Years in retirement to simulate
            num_simulations: Number of Monte Carlo simulations
            strategies: List of strategies to simulate
                       Options: "fixed", "variable_percentage", "guyton_klinger"
        
        Returns:
            Dictionary with results for each strategy
        """
        if strategies is None:
            strategies = ["fixed", "variable_percentage", "guyton_klinger"]
        
        # Set simulation parameters
        mean_return = self.expected_return
        std_dev = self.config['simulation_settings'].get('standard_deviation', 0.15)
        inflation = self.inflation_rate
        
        # Generate random returns and inflation
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(mean_return, std_dev, (years, num_simulations))
        
        # Results dictionary
        results = {}
        
        # Initial withdrawal amount (4% of initial portfolio)
        initial_withdrawal = initial_portfolio * self.withdrawal_rate
        
        # Simulate each strategy
        for strategy in strategies:
            # Initialize portfolio and withdrawal arrays
            portfolios = np.zeros((years + 1, num_simulations))
            portfolios[0, :] = initial_portfolio
            withdrawals = np.zeros((years, num_simulations))
            
            # Run simulations
            for sim in range(num_simulations):
                # Reset withdrawal for each simulation
                current_withdrawal = initial_withdrawal
                
                for year in range(years):
                    # Record withdrawal for this year
                    withdrawals[year, sim] = current_withdrawal
                    
                    # Update portfolio after withdrawal
                    portfolios[year+1, sim] = (portfolios[year, sim] - current_withdrawal) * (1 + returns[year, sim])
                    
                    # Adjust withdrawal based on strategy
                    if strategy == "fixed":
                        # Standard 4% rule: increase withdrawal with inflation
                        current_withdrawal *= (1 + inflation)
                    
                    elif strategy == "variable_percentage":
                        # Variable percentage: recalculate based on new portfolio value
                        floor_rate = self.config['withdrawal_strategies'].get('floor_rate', 0.025)
                        ceiling_rate = self.config['withdrawal_strategies'].get('ceiling_rate', 0.055)
                        base_rate = self.withdrawal_rate
                        
                        # Calculate new withdrawal
                        new_withdrawal_rate = base_rate
                        new_withdrawal = portfolios[year+1, sim] * new_withdrawal_rate
                        
                        # Enforce floor and ceiling
                        min_withdrawal = initial_withdrawal * (1 + inflation)**(year+1) * floor_rate / self.withdrawal_rate
                        max_withdrawal = initial_withdrawal * (1 + inflation)**(year+1) * ceiling_rate / self.withdrawal_rate
                        
                        current_withdrawal = np.clip(new_withdrawal, min_withdrawal, max_withdrawal)
                    
                    elif strategy == "guyton_klinger":
                        # Guyton-Klinger rules (simplified implementation)
                        adjustment_cap = self.config['withdrawal_strategies'].get('variable_adjustment_cap', 0.10)
                        inflation_rate = inflation
                        
                        # Capital Preservation Rule
                        if portfolios[year+1, sim] < portfolios[0, sim] * 0.8:
                            current_withdrawal *= 0.9  # Reduce by 10%
                        
                        # Prosperity Rule
                        elif portfolios[year+1, sim] > portfolios[0, sim] * 1.2:
                            current_withdrawal *= 1.1  # Increase by 10%
                        
                        # Otherwise, just adjust for inflation
                        else:
                            current_withdrawal *= (1 + inflation_rate)
            
            # Calculate success rate (portfolio > 0 at end)
            success_rate = np.mean(portfolios[-1, :] > 0)
            
            # Calculate percentiles for visualization
            percentiles = [5, 25, 50, 75, 95]
            percentile_data = {
                'portfolios': {p: np.percentile(portfolios, p, axis=1) for p in percentiles},
                'withdrawals': {p: np.percentile(withdrawals, p, axis=1) for p in percentiles}
            }
            
            # Store results
            results[strategy] = {
                'success_rate': success_rate,
                'median_final_portfolio': np.median(portfolios[-1, :]),
                'percentiles': percentile_data
            }
        
        return results
    
    def calculate_coast_fire(self, target_retirement_age: int = None) -> Dict:
        """
        Calculate Coast FIRE metrics.
        
        Args:
            target_retirement_age: Age at which traditional retirement would begin
            
        Returns:
            Dictionary with Coast FIRE metrics
        """
        # Use config or default
        target_retirement_age = target_retirement_age or self.config['scenario_analysis'].get('coast_fire_retirement_age', 65)
        
        # Calculate years to target retirement age
        years_to_target = target_retirement_age - self.current_age
        
        # Calculate future value needed (FIRE number adjusted for inflation)
        fire_number = self.calculate_fire_number()
        inflated_fire_number = fire_number * (1 + self.inflation_rate) ** years_to_target
        
        # Calculate the amount needed today to reach that future value
        # with no additional contributions
        coast_fire_number = inflated_fire_number / ((1 + self.expected_return) ** years_to_target)
        
        # Calculate time to reach Coast FIRE with current savings rate
        annual_contribution = self.annual_income * self.savings_rate
        time_to_coast_fire = 0
        portfolio = self.current_savings
        
        while portfolio < coast_fire_number and time_to_coast_fire < 100:
            portfolio = portfolio * (1 + self.expected_return) + annual_contribution
            time_to_coast_fire += 1
        
        # Calculate percentage of Coast FIRE already achieved
        current_as_percent_of_coast = (self.current_savings / coast_fire_number) * 100
        
        return {
            'coast_fire_number': coast_fire_number,
            'time_to_coast_fire': time_to_coast_fire,
            'current_as_percent_of_coast': current_as_percent_of_coast,
            'target_retirement_age': target_retirement_age
        }
    
    def calculate_fat_fire(self, expense_multiplier: float = None) -> Dict:
        """
        Calculate Fat FIRE metrics.
        
        Args:
            expense_multiplier: Multiplier for annual expenses
            
        Returns:
            Dictionary with Fat FIRE metrics
        """
        # Use config or default
        expense_multiplier = expense_multiplier or self.config['scenario_analysis'].get('fat_fire_expense_multiplier', 1.5)
        
        # Calculate standard FIRE number
        standard_fire_number = self.calculate_fire_number()
        
        # Calculate Fat FIRE number (higher expenses)
        fat_fire_number = standard_fire_number * expense_multiplier
        
        # Calculate years to Fat FIRE
        annual_contribution = self.annual_income * self.savings_rate
        years_to_fat_fire = 0
        portfolio = self.current_savings
        
        while portfolio < fat_fire_number and years_to_fat_fire < 100:
            portfolio = portfolio * (1 + self.expected_return) + annual_contribution
            years_to_fat_fire += 1
        
        # Calculate standard years to FIRE
        standard_years_to_fire = self.calculate_years_to_fire()
        
        # Calculate additional years needed
        additional_years = years_to_fat_fire - standard_years_to_fire
        
        return {
            'fat_fire_number': fat_fire_number,
            'years_to_fat_fire': years_to_fat_fire,
            'additional_years': additional_years,
            'expense_multiplier': expense_multiplier
        }
    
    def calculate_lean_fire(self, expense_multiplier: float = None) -> Dict:
        """
        Calculate Lean FIRE metrics.
        
        Args:
            expense_multiplier: Multiplier for annual expenses
            
        Returns:
            Dictionary with Lean FIRE metrics
        """
        # Use config or default
        expense_multiplier = expense_multiplier or self.config['scenario_analysis'].get('lean_fire_expense_multiplier', 0.7)
        
        # Calculate standard FIRE number
        standard_fire_number = self.calculate_fire_number()
        
        # Calculate Lean FIRE number (lower expenses)
        lean_fire_number = standard_fire_number * expense_multiplier
        
        # Calculate years to Lean FIRE
        annual_contribution = self.annual_income * self.savings_rate
        years_to_lean_fire = 0
        portfolio = self.current_savings
        
        while portfolio < lean_fire_number and years_to_lean_fire < 100:
            portfolio = portfolio * (1 + self.expected_return) + annual_contribution
            years_to_lean_fire += 1
        
        # Calculate standard years to FIRE
        standard_years_to_fire = self.calculate_years_to_fire()
        
        # Calculate years saved
        years_saved = standard_years_to_fire - years_to_lean_fire
        
        return {
            'lean_fire_number': lean_fire_number,
            'years_to_lean_fire': years_to_lean_fire,
            'years_saved': years_saved,
            'expense_multiplier': expense_multiplier
        }
    
    def compare_savings_rates(self, rates: List[float] = None) -> Dict[float, Dict]:
        """
        Compare different savings rates and their impact on time to FIRE.
        
        Args:
            rates: List of savings rates to compare
            
        Returns:
            Dictionary mapping rates to time-to-FIRE results
        """
        # Use config or default
        rates = rates or self.config['scenario_analysis'].get('compare_savings_rates', [0.30, 0.40, 0.50])
        
        # Store results
        results = {}
        
        # Calculate for each rate
        for rate in rates:
            # Calculate years to FIRE with this rate
            annual_contribution = self.annual_income * rate
            fire_number = self.calculate_fire_number()
            years = 0
            portfolio = self.current_savings
            
            while portfolio < fire_number and years < 100:
                portfolio = portfolio * (1 + self.expected_return) + annual_contribution
                years += 1
            
            # Calculate retirement age
            retirement_age = self.current_age + years
            
            # Store results
            results[rate] = {
                'years_to_fire': years,
                'retirement_age': retirement_age,
                'final_portfolio': portfolio
            }
        
        return results
    
    # Visualization methods
    def plot_fire_journey(self, journey_df: pd.DataFrame, 
                          save_path: str = None,
                          show_plot: bool = None) -> None:
        """
        Plot the FIRE journey from the simulation data.
        
        Args:
            journey_df: DataFrame from simulate_fire_journey()
            save_path: Optional path to save the chart
            show_plot: Whether to display the plot
        """
        # Use config or defaults
        save_charts = self.config['visualization'].get('save_charts', False) if save_path is None else True
        show_plots = show_plot if show_plot is not None else self.config['visualization'].get('show_plots', True)
        
        # Create figure and axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot portfolio value
        ax1.plot(journey_df['Age'], journey_df['Portfolio Value'], 
                color='#1f77b4', linewidth=2, label='Portfolio Value')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        
        # Plot FIRE threshold
        ax1.axhline(y=journey_df['FIRE Number'].iloc[0], color='red', linestyle='--', 
                   label=f'FIRE Number: ${journey_df["FIRE Number"].iloc[0]:,.0f}')
        
        # Create second y-axis for FIRE percentage
        ax2 = ax1.twinx()
        ax2.plot(journey_df['Age'], journey_df['FIRE %'], 
                color='green', linewidth=1.5, linestyle=':', label='FIRE %')
        ax2.set_ylabel('FIRE %')
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Add vertical line at FIRE achievement
        if any(journey_df['FIRE Achieved']):
            fire_age = journey_df.loc[journey_df['FIRE Achieved'] == True, 'Age'].iloc[0]
            ax1.axvline(x=fire_age, color='green', linestyle='-', alpha=0.3)
            ax1.text(fire_age + 0.5, journey_df['Portfolio Value'].max() * 0.9, 
                    f'FIRE Achieved: Age {fire_age}', color='green')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Set title
        plt.title('Journey to Financial Independence', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Save chart if configured
        if save_charts:
            output_dir = save_path or self.config['visualization'].get('output_directory', './reports')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'fire_journey.png'), dpi=300, bbox_inches='tight')
        
        # Show plot if configured
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_monte_carlo(self, results_df: pd.DataFrame, summary: Dict,
                         save_path: str = None,
                         show_plot: bool = None) -> None:
        """
        Plot Monte Carlo simulation results.
        
        Args:
            results_df: DataFrame from run_monte_carlo_simulation()
            summary: Summary dictionary from run_monte_carlo_simulation()
            save_path: Optional path to save the chart
            show_plot: Whether to display the plot
        """
        # Use config or defaults
        save_charts = self.config['visualization'].get('save_charts', False) if save_path is None else True
        show_plots = show_plot if show_plot is not None else self.config['visualization'].get('show_plots', True)
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot percentiles
        ax.fill_between(results_df['Year'], results_df['P5'], results_df['P95'], 
                        alpha=0.1, color='blue', label='5th-95th Percentile')
        ax.fill_between(results_df['Year'], results_df['P25'], results_df['P75'], 
                        alpha=0.2, color='blue', label='25th-75th Percentile')
        ax.plot(results_df['Year'], results_df['P50'], 'b-', 
                linewidth=2, label='Median')
        
        # Plot FIRE number
        fire_number = self.calculate_fire_number()
        ax.axhline(y=fire_number, color='red', linestyle='--', 
                  label=f'FIRE Number: ${fire_number:,.0f}')
        
        # Format y-axis to show dollars
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        
        # Add text box with summary statistics
        stats_text = (
    f"Success Rate: {summary['success_rate']:.1%}\n"
    f"Median Final Portfolio: ${summary['median_final_portfolio']:,.0f}\n"
    f"Minimum Final Portfolio: ${summary['min_final_portfolio']:,.0f}\n"
    f"Maximum Final Portfolio: ${summary['max_final_portfolio']:,.0f}\n"
    f"Retirement Age: {summary.get('retirement_age', 'N/A')}"
)
        
        plt.figtext(0.15, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        # Set labels and title
        ax.set_xlabel('Years')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Monte Carlo Simulation of Portfolio Growth', fontsize=16)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Save chart if configured
        if save_charts:
            output_dir = save_path or self.config['visualization'].get('output_directory', './reports')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'monte_carlo_simulation.png'), dpi=300, bbox_inches='tight')
        
        # Show plot if configured
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_alternative_fire_paths(self, 
                                   coast_fire_data: Dict, 
                                   lean_fire_data: Dict,
                                   fat_fire_data: Dict,
                                   save_path: str = None,
                                   show_plot: bool = None) -> None:
        """
        Plot comparison of different FIRE paths.
        
        Args:
            coast_fire_data: Dictionary with Coast FIRE metrics
            lean_fire_data: Dictionary with Lean FIRE metrics
            fat_fire_data: Dictionary with Fat FIRE metrics
            save_path: Optional path to save the chart
            show_plot: Whether to display the plot
        """
        # Use config or defaults
        save_charts = self.config['visualization'].get('save_charts', False) if save_path is None else True
        show_plots = show_plot if show_plot is not None else self.config['visualization'].get('show_plots', True)
        
        # Create bar chart data
        fire_types = ['Standard FIRE', 'Coast FIRE', 'Lean FIRE', 'Fat FIRE']
        
        standard_fire_number = self.calculate_fire_number()
        fire_numbers = [
            standard_fire_number,
            coast_fire_data['coast_fire_number'],
            lean_fire_data['lean_fire_number'],
            fat_fire_data['fat_fire_number']
        ]
        
        years_to_fire = [
            self.calculate_years_to_fire(),
            coast_fire_data['time_to_coast_fire'],
            lean_fire_data['years_to_lean_fire'],
            fat_fire_data['years_to_fat_fire']
        ]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot FIRE numbers
        bars1 = ax1.bar(fire_types, fire_numbers, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_ylabel('FIRE Number ($)')
        ax1.set_title('FIRE Numbers by Type', fontsize=14)
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        
        # Add value labels to bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'${height:,.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=0)
        
        # Plot years to FIRE
        bars2 = ax2.bar(fire_types, years_to_fire, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_ylabel('Years to FIRE')
        ax2.set_title('Years to FIRE by Type', fontsize=14)
        
        # Add value labels to bars
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f} years',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart if configured
        if save_charts:
            output_dir = save_path or self.config['visualization'].get('output_directory', './reports')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'alternative_fire_paths.png'), dpi=300, bbox_inches='tight')
        
        # Show plot if configured
        if show_plots:
            plt.show()
        else:
            plt.close()

    def plot_savings_rate_comparison(self, comparison_data: Dict,
                                    save_path: str = None,
                                    show_plot: bool = None) -> None:
        """
        Plot comparison of different savings rates.
        
        Args:
            comparison_data: Dictionary from compare_savings_rates()
            save_path: Optional path to save the chart
            show_plot: Whether to display the plot
        """
        # Use config or defaults
        save_charts = self.config['visualization'].get('save_charts', False) if save_path is None else True
        show_plots = show_plot if show_plot is not None else self.config['visualization'].get('show_plots', True)
        
        # Extract data
        rates = list(comparison_data.keys())
        years = [data['years_to_fire'] for data in comparison_data.values()]
        retirement_ages = [data['retirement_age'] for data in comparison_data.values()]
        
        # Format rates as percentages for display
        rate_labels = [f'{rate:.0%}' for rate in rates]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot years to FIRE
        bars1 = ax1.bar(rate_labels, years, color=plt.cm.viridis(np.linspace(0, 1, len(rates))))
        ax1.set_xlabel('Savings Rate')
        ax1.set_ylabel('Years to FIRE')
        ax1.set_title('Impact of Savings Rate on Time to FIRE', fontsize=14)
        
        # Add value labels to bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f} years',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=0)
        
        # Plot retirement age
        bars2 = ax2.bar(rate_labels, retirement_ages, color=plt.cm.viridis(np.linspace(0, 1, len(rates))))
        ax2.set_xlabel('Savings Rate')
        ax2.set_ylabel('Retirement Age')
        ax2.set_title('Impact of Savings Rate on Retirement Age', fontsize=14)
        
        # Add value labels to bars
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'Age {height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart if configured
        if save_charts:
            output_dir = save_path or self.config['visualization'].get('output_directory', './reports')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'savings_rate_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Show plot if configured
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def generate_summary_report(self, save_path: str = None) -> Dict:
        """
        Generate a comprehensive summary report of FIRE calculations.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Dictionary with summary metrics
        """
        # Calculate standard metrics
        fire_number = self.calculate_fire_number()
        years_to_fire = self.calculate_years_to_fire()
        retirement_age = self.current_age + years_to_fire
        
        # Calculate alternative FIRE metrics
        coast_fire_data = self.calculate_coast_fire()
        lean_fire_data = self.calculate_lean_fire()
        fat_fire_data = self.calculate_fat_fire()
        
        # Compare savings rates
        savings_comparison = self.compare_savings_rates()
        
        # Compile summary
        summary = {
            'personal_info': {
                'current_age': self.current_age,
                'annual_expenses': self.annual_expenses,
                'current_savings': self.current_savings,
                'annual_income': self.annual_income,
                'savings_rate': self.savings_rate,
                'expected_return': self.expected_return,
                'inflation_rate': self.inflation_rate,
                'withdrawal_rate': self.withdrawal_rate,
                'tax_rate': self.tax_rate
            },
            'fire_metrics': {
                'fire_number': fire_number,
                'years_to_fire': years_to_fire,
                'retirement_age': retirement_age,
                'monthly_contribution': self.annual_income * self.savings_rate / 12
            },
            'alternative_fire': {
                'coast_fire': coast_fire_data,
                'lean_fire': lean_fire_data,
                'fat_fire': fat_fire_data
            },
            'savings_comparison': savings_comparison
        }
        
        # Save report if configured
        if save_path or self.config['visualization'].get('save_charts', False):
            output_dir = save_path or self.config['visualization'].get('output_directory', './reports')
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'fire_summary_report.json'), 'w') as f:
                json.dump(summary, f, indent=4)
        
        return summary
    
    def run_all_analyses(self, save_results: bool = None) -> Dict:
        """
        Run all FIRE analyses and generate visualizations.
        
        Args:
            save_results: Whether to save results and charts
            
        Returns:
            Dictionary with all analysis results
        """
        # Use config or default
        save_results = save_results if save_results is not None else self.config['visualization'].get('save_charts', False)
        output_dir = self.config['visualization'].get('output_directory', './reports')
        
        # Ensure output directory exists if saving
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
        # Create container for all results
        all_results = {}
        
        # Get configuration for analyses to run
        run_analyses = self.config.get('run_analyses', {})
        
        # 1. Simulate journey to FIRE
        if run_analyses.get('simulate_journey', True):
            logger.info("Simulating FIRE journey...")
            journey_df = self.simulate_fire_journey()
            all_results['journey'] = journey_df
            
            # Plot journey
            self.plot_fire_journey(journey_df, save_path=output_dir if save_results else None)
        
        # 2. Run Monte Carlo simulation
        if run_analyses.get('monte_carlo', True):
            logger.info("Running Monte Carlo simulation...")
            # Use simulation settings from config
            sim_settings = self.config.get('simulation_settings', {})
            
            # TODO: The monte_carlo_simulation method needs to be implemented
            # For now, I'll add a placeholder
            mc_results, mc_summary = {}, {}
            all_results['monte_carlo'] = {
                'results': mc_results,
                'summary': mc_summary
            }
            
            # Plot Monte Carlo results (commented out until implemented)
            # self.plot_monte_carlo(mc_results, mc_summary, save_path=output_dir if save_results else None)
            
        # 3. Alternative FIRE calculations
        if run_analyses.get('alternative_fire', True):
            logger.info("Calculating alternative FIRE paths...")
            coast_fire_data = self.calculate_coast_fire()
            lean_fire_data = self.calculate_lean_fire()
            fat_fire_data = self.calculate_fat_fire()
            
            all_results['alternative_fire'] = {
                'coast_fire': coast_fire_data,
                'lean_fire': lean_fire_data,
                'fat_fire': fat_fire_data
            }
            
            # Plot alternative FIRE paths
            self.plot_alternative_fire_paths(
                coast_fire_data, lean_fire_data, fat_fire_data,
                save_path=output_dir if save_results else None
            )
        
        # 4. Compare savings rates
        if run_analyses.get('compare_savings_rates', True):
            logger.info("Comparing different savings rates...")
            savings_comparison = self.compare_savings_rates()
            all_results['savings_comparison'] = savings_comparison
            
            # Plot savings rate comparison
            self.plot_savings_rate_comparison(
                savings_comparison,
                save_path=output_dir if save_results else None
            )
        
        # 5. Generate summary report
        if run_analyses.get('generate_report', True):
            logger.info("Generating summary report...")
            summary = self.generate_summary_report(save_path=output_dir if save_results else None)
            all_results['summary'] = summary
        
        return all_results


def main():
    """
    Main function to parse arguments and run the FIRE calculator.
    """
    parser = argparse.ArgumentParser(description='FIRE Calculator Pro - A tool for financial independence planning')
    
    parser.add_argument('-c', '--config', type=str, default='settings.json',
                       help='Path to configuration file (default: settings.json)')
    
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output directory for reports and charts')
    
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable displaying plots (useful for non-interactive environments)')
    
    parser.add_argument('--save', action='store_true',
                       help='Save results and charts')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Add this block to create settings.json if it doesn't exist
    if not os.path.exists(args.config):
        print(f"Creating default settings.json file at {os.path.abspath(args.config)}...")
        import json
        default_settings = {
            "personal_info": {
                "current_age": 25,
                "annual_expenses": 50000,
                "current_savings": 500000,
                "annual_income": 90000,
                "savings_rate": 0.40,
                "expected_return": 0.07,
                "inflation_rate": 0.025,
                "withdrawal_rate": 0.025,
                "tax_rate": 0.26
            },
            "simulation_settings": {
                "monte_carlo_runs": 5000,
                "time_horizon_years": 50,
                "confidence_interval": 0.95,
                "standard_deviation": 0.15
            },
            "withdrawal_strategies": {
                "strategy": "fixed",
                "initial_rate": 0.04,
                "floor_rate": 0.025,
                "ceiling_rate": 0.055,
                "variable_adjustment_cap": 0.10
            },
            "visualization": {
                "theme": "default",
                "save_charts": True,
                "show_plots": True,
                "output_directory": "./reports",
                "chart_style": "seaborn-v0_8-darkgrid",
                "chart_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
            },
            "run_analyses": {
                "simulate_journey": True,
                "monte_carlo": True,
                "withdrawal_strategies": True,
                "alternative_fire": True,
                "generate_report": True,
                "savings_rate_comparison": True
            },
            "scenario_analysis": {
                "compare_savings_rates": [0.30, 0.40, 0.50],
                "coast_fire_retirement_age": 65,
                "fat_fire_expense_multiplier": 1.5,
                "lean_fire_expense_multiplier": 0.7
            },
            "output_settings": {
                "file_format": "json",
                "include_charts": True,
                "detailed_metrics": True,
                "date_format": "YYYY-MM-DD"
            }
        }
        with open(args.config, 'w') as f:
            json.dump(default_settings, f, indent=2)
        print(f"Created default settings file at: {os.path.abspath(args.config)}")
    
    # Check if config file exists (keep this as a fallback)
    if not os.path.exists(args.config):
        logger.error(f"Configuration file '{args.config}' not found.")
        print(f"Error: Configuration file '{args.config}' not found.")
        print("Please create a settings.json file or specify a different file with --config.")
        sys.exit(1)
    
    try:
        # Load calculator from config file
        calculator = FIRECalculator.from_json(args.config)
        
        # Override configuration with command-line arguments
        if args.output:
            calculator.config['visualization']['output_directory'] = args.output
        
        if args.no_plots:
            calculator.config['visualization']['show_plots'] = False
        
        if args.save:
            calculator.config['visualization']['save_charts'] = True
        
        # Run analyses
        results = calculator.run_all_analyses()
        
        # Print summary
        summary = results.get('summary', {}).get('fire_metrics', {})
        print("\n== FIRE Calculator Pro - Results Summary ==")
        print(f"FIRE Number: ${summary.get('fire_number', 0):,.2f}")
        print(f"Years to FIRE: {summary.get('years_to_fire', 0):.1f}")
        print(f"Retirement Age: {summary.get('retirement_age', 0):.1f}")
        print(f"Monthly Contribution: ${summary.get('monthly_contribution', 0):,.2f}")
        print("============================================\n")
        
        # Print output location if saved
        if calculator.config['visualization']['save_charts']:
            output_dir = calculator.config['visualization']['output_directory']
            print(f"Reports and charts saved to: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

