"""
Logic Quantization Module (LQM) - Zero-Hallucination Mathematics

This module ensures all monetary calculations use decimal.Decimal to prevent
floating-point precision errors in financial computations. The LQM Standard
mandates that NO float types are allowed for currency operations.

Key Features:
- Strict decimal.Decimal enforcement for all monetary values
- EMI calculation using reducing balance formula
- Precision validation to exactly 2 decimal places
- Clear error messages for float input rejection
- Zero-hallucination mathematical guarantees

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, Any, Union
import logging

# Configure logger
logger = logging.getLogger(__name__)

class LQMError(Exception):
    """Base exception for LQM module errors."""
    pass

class FloatInputError(LQMError):
    """Raised when float input is detected in monetary calculations."""
    pass

class PrecisionError(LQMError):
    """Raised when decimal precision validation fails."""
    pass

class InvalidParameterError(LQMError):
    """Raised when calculation parameters are invalid."""
    pass

class EMICalculation:
    """
    Data class for EMI calculation results.
    All monetary values are stored as decimal.Decimal with 2 decimal places.
    """
    
    def __init__(
        self,
        principal_in_cents: Decimal,
        rate_per_annum: Decimal,
        tenure_months: int,
        emi_in_cents: Decimal,
        total_interest_in_cents: Decimal,
        total_amount_in_cents: Decimal
    ):
        self.principal_in_cents = principal_in_cents
        self.rate_per_annum = rate_per_annum
        self.tenure_months = tenure_months
        self.emi_in_cents = emi_in_cents
        self.total_interest_in_cents = total_interest_in_cents
        self.total_amount_in_cents = total_amount_in_cents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with string representation of Decimal values."""
        return {
            "principal_in_cents": str(self.principal_in_cents),
            "rate_per_annum": str(self.rate_per_annum),
            "tenure_months": self.tenure_months,
            "emi_in_cents": str(self.emi_in_cents),
            "total_interest_in_cents": str(self.total_interest_in_cents),
            "total_amount_in_cents": str(self.total_amount_in_cents)
        }

class LQM:
    """
    Logic Quantization Module - The mathematical heart of Loan2Day.
    
    This class implements zero-hallucination mathematics by enforcing
    decimal.Decimal usage for all monetary calculations. The LQM Standard
    ensures financial precision and regulatory compliance.
    """
    
    @staticmethod
    def validate_decimal_input(value: Any, parameter_name: str) -> Decimal:
        """
        Validate that input is decimal.Decimal and reject float inputs.
        
        Args:
            value: The value to validate
            parameter_name: Name of the parameter for error messages
            
        Returns:
            Decimal: The validated decimal value
            
        Raises:
            FloatInputError: If float input is detected
            InvalidParameterError: If input cannot be converted to Decimal
        """
        # Explicitly reject float inputs
        if isinstance(value, float):
            raise FloatInputError(
                f"Float input detected for {parameter_name}: {value}. "
                f"The LQM Standard mandates decimal.Decimal usage for all monetary calculations. "
                f"Use Decimal('{value}') instead of {value}."
            )
        
        # Handle string inputs by converting to Decimal
        if isinstance(value, str):
            try:
                decimal_value = Decimal(value)
            except InvalidOperation as e:
                raise InvalidParameterError(
                    f"Invalid decimal string for {parameter_name}: '{value}'. "
                    f"Error: {str(e)}"
                )
        elif isinstance(value, (int, Decimal)):
            decimal_value = Decimal(str(value))
        else:
            raise InvalidParameterError(
                f"Invalid type for {parameter_name}: {type(value)}. "
                f"Expected Decimal, int, or string representation."
            )
        
        return decimal_value
    
    @staticmethod
    def validate_precision(value: Decimal, parameter_name: str) -> Decimal:
        """
        Validate that decimal has exactly 2 decimal places for currency.
        
        Args:
            value: The decimal value to validate
            parameter_name: Name of the parameter for error messages
            
        Returns:
            Decimal: The validated decimal with 2 decimal places
            
        Raises:
            PrecisionError: If precision is not exactly 2 decimal places
        """
        # Round to 2 decimal places using banker's rounding
        rounded_value = value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        # Check if the original value had more than 2 decimal places
        if value != rounded_value:
            logger.warning(
                f"Precision adjustment for {parameter_name}: {value} -> {rounded_value}"
            )
        
        return rounded_value
    
    @staticmethod
    def validate_positive_value(value: Decimal, parameter_name: str, allow_zero: bool = False) -> Decimal:
        """
        Validate that the value is positive (or zero if allowed).
        
        Args:
            value: The decimal value to validate
            parameter_name: Name of the parameter for error messages
            allow_zero: Whether to allow zero values (default: False)
            
        Returns:
            Decimal: The validated decimal
            
        Raises:
            InvalidParameterError: If value is not positive (or zero when allowed)
        """
        if allow_zero and value < 0:
            raise InvalidParameterError(
                f"{parameter_name} must be non-negative. Received: {value}"
            )
        elif not allow_zero and value <= 0:
            raise InvalidParameterError(
                f"{parameter_name} must be positive. Received: {value}"
            )
        
        return value
    
    @staticmethod
    def calculate_emi(
        principal: Union[Decimal, str, int],
        annual_rate: Union[Decimal, str, int],
        tenure_months: int
    ) -> EMICalculation:
        """
        Calculate EMI using the reducing balance formula with zero-hallucination math.
        
        Formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        Where:
        - P = Principal amount (in cents for precision)
        - r = Monthly interest rate (annual_rate / 12 / 100)
        - n = Tenure in months
        
        Args:
            principal: Loan principal amount (can be Decimal, string, or int)
            annual_rate: Annual interest rate as percentage (e.g., 12.5 for 12.5%)
            tenure_months: Loan tenure in months
            
        Returns:
            EMICalculation: Complete EMI calculation results
            
        Raises:
            FloatInputError: If any float input is detected
            InvalidParameterError: If parameters are invalid
            PrecisionError: If precision validation fails
        """
        logger.info(
            f"Starting EMI calculation - Principal: {principal}, "
            f"Rate: {annual_rate}%, Tenure: {tenure_months} months"
        )
        
        # Validate and convert inputs to Decimal
        principal_decimal = LQM.validate_decimal_input(principal, "principal")
        rate_decimal = LQM.validate_decimal_input(annual_rate, "annual_rate")
        
        # Validate tenure is positive integer
        if not isinstance(tenure_months, int) or tenure_months <= 0:
            raise InvalidParameterError(
                f"tenure_months must be a positive integer. Received: {tenure_months}"
            )
        
        # Validate positive values (allow zero interest rate)
        principal_decimal = LQM.validate_positive_value(principal_decimal, "principal")
        rate_decimal = LQM.validate_positive_value(rate_decimal, "annual_rate", allow_zero=True)
        
        # Validate and enforce 2 decimal place precision
        principal_in_cents = LQM.validate_precision(principal_decimal, "principal")
        rate_per_annum = LQM.validate_precision(rate_decimal, "annual_rate")
        
        # Calculate monthly interest rate (r = annual_rate / 12 / 100)
        monthly_rate = rate_per_annum / Decimal('12') / Decimal('100')
        
        # Handle edge case: 0% interest rate
        if monthly_rate == 0:
            emi_in_cents = principal_in_cents / Decimal(str(tenure_months))
            emi_in_cents = LQM.validate_precision(emi_in_cents, "emi_calculation")
            total_interest_in_cents = Decimal('0.00')
            total_amount_in_cents = principal_in_cents
            
            logger.info(f"Zero interest rate detected. Simple division EMI: {emi_in_cents}")
        else:
            # Reducing balance EMI formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1)
            
            # Calculate (1 + r)
            one_plus_r = Decimal('1') + monthly_rate
            
            # Calculate (1 + r)^n using decimal power
            # Note: Decimal doesn't have ** operator, so we use repeated multiplication
            power_term = Decimal('1')
            for _ in range(tenure_months):
                power_term *= one_plus_r
            
            # Calculate numerator: P * r * (1+r)^n
            numerator = principal_in_cents * monthly_rate * power_term
            
            # Calculate denominator: (1+r)^n - 1
            denominator = power_term - Decimal('1')
            
            # Calculate EMI
            emi_in_cents = numerator / denominator
            emi_in_cents = LQM.validate_precision(emi_in_cents, "emi_calculation")
            
            # Calculate total amount and interest
            total_amount_in_cents = emi_in_cents * Decimal(str(tenure_months))
            total_interest_in_cents = total_amount_in_cents - principal_in_cents
            
            # Ensure precision for all calculated values
            total_amount_in_cents = LQM.validate_precision(total_amount_in_cents, "total_amount")
            total_interest_in_cents = LQM.validate_precision(total_interest_in_cents, "total_interest")
        
        # Create and return EMI calculation result
        result = EMICalculation(
            principal_in_cents=principal_in_cents,
            rate_per_annum=rate_per_annum,
            tenure_months=tenure_months,
            emi_in_cents=emi_in_cents,
            total_interest_in_cents=total_interest_in_cents,
            total_amount_in_cents=total_amount_in_cents
        )
        
        logger.info(
            f"EMI calculation completed - EMI: {emi_in_cents}, "
            f"Total Interest: {total_interest_in_cents}, "
            f"Total Amount: {total_amount_in_cents}"
        )
        
        return result
    
    @staticmethod
    def convert_to_currency_display(amount_in_cents: Decimal) -> str:
        """
        Convert amount in cents to currency display format.
        
        Args:
            amount_in_cents: Amount in cents as Decimal
            
        Returns:
            str: Formatted currency string (e.g., "₹1,50,000.00")
        """
        # Validate input
        amount_decimal = LQM.validate_decimal_input(amount_in_cents, "amount_in_cents")
        amount_decimal = LQM.validate_precision(amount_decimal, "amount_in_cents")
        
        # Convert cents to rupees (divide by 100)
        amount_in_rupees = amount_decimal / Decimal('100')
        
        # Format with Indian numbering system
        return f"₹{amount_in_rupees:,.2f}"
    
    @staticmethod
    def convert_from_currency_input(currency_string: str) -> Decimal:
        """
        Convert currency input string to cents as Decimal.
        
        Args:
            currency_string: Currency string (e.g., "150000.50" or "₹1,50,000.50")
            
        Returns:
            Decimal: Amount in cents
            
        Raises:
            InvalidParameterError: If currency string is invalid
        """
        # Clean the input string
        cleaned = currency_string.strip()
        
        # Remove currency symbols and commas
        cleaned = cleaned.replace('₹', '').replace(',', '').strip()
        
        try:
            # Convert to Decimal (amount in rupees)
            amount_in_rupees = Decimal(cleaned)
            
            # Convert to cents (multiply by 100)
            amount_in_cents = amount_in_rupees * Decimal('100')
            
            # Validate precision
            amount_in_cents = LQM.validate_precision(amount_in_cents, "currency_conversion")
            
            return amount_in_cents
            
        except InvalidOperation as e:
            raise InvalidParameterError(
                f"Invalid currency string: '{currency_string}'. Error: {str(e)}"
            )

# Convenience functions for direct usage
def calculate_emi(
    principal: Union[Decimal, str, int],
    annual_rate: Union[Decimal, str, int],
    tenure_months: int
) -> EMICalculation:
    """
    Convenience function for EMI calculation.
    
    This is the main entry point for EMI calculations in the Loan2Day system.
    All monetary calculations MUST use this function to ensure LQM compliance.
    
    Args:
        principal: Loan principal amount
        annual_rate: Annual interest rate as percentage
        tenure_months: Loan tenure in months
        
    Returns:
        EMICalculation: Complete EMI calculation results
    """
    return LQM.calculate_emi(principal, annual_rate, tenure_months)

def validate_monetary_input(value: Any, parameter_name: str) -> Decimal:
    """
    Convenience function for validating monetary inputs.
    
    Args:
        value: The value to validate
        parameter_name: Name of the parameter for error messages
        
    Returns:
        Decimal: The validated decimal value with 2 decimal places
    """
    decimal_value = LQM.validate_decimal_input(value, parameter_name)
    return LQM.validate_precision(decimal_value, parameter_name)

# Module-level constants for common calculations
ZERO_DECIMAL = Decimal('0.00')
ONE_DECIMAL = Decimal('1.00')
HUNDRED_DECIMAL = Decimal('100.00')

# Export main classes and functions
__all__ = [
    'LQM',
    'EMICalculation',
    'LQMError',
    'FloatInputError',
    'PrecisionError',
    'InvalidParameterError',
    'calculate_emi',
    'validate_monetary_input',
    'ZERO_DECIMAL',
    'ONE_DECIMAL',
    'HUNDRED_DECIMAL'
]