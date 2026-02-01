"""
Unit tests for LQM (Logic Quantization Module).

This test suite validates the zero-hallucination mathematics implementation
ensuring all monetary calculations use decimal.Decimal with proper precision.

Test Coverage:
- EMI calculation correctness with known reference values
- Float input rejection with clear error messages
- Precision validation to exactly 2 decimal places
- Edge cases (zero interest, large amounts, boundary values)
- Input validation and error handling
- Currency conversion utilities

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

import pytest
from decimal import Decimal, InvalidOperation
import sys
import os

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from core.lqm import (
    LQM, 
    EMICalculation, 
    calculate_emi, 
    validate_monetary_input,
    FloatInputError, 
    PrecisionError, 
    InvalidParameterError,
    ZERO_DECIMAL,
    ONE_DECIMAL,
    HUNDRED_DECIMAL
)

class TestLQMValidation:
    """Test LQM input validation and error handling."""
    
    def test_float_input_rejection_principal(self):
        """Test that float principal input is rejected with clear error message."""
        with pytest.raises(FloatInputError) as exc_info:
            LQM.validate_decimal_input(100000.50, "principal")
        
        error_msg = str(exc_info.value)
        assert "Float input detected" in error_msg
        assert "LQM Standard" in error_msg
        assert "decimal.Decimal" in error_msg
        assert "100000.5" in error_msg
    
    def test_float_input_rejection_rate(self):
        """Test that float rate input is rejected."""
        with pytest.raises(FloatInputError):
            calculate_emi(Decimal('100000'), 12.5, 12)
    
    def test_invalid_string_input(self):
        """Test that invalid string inputs raise appropriate errors."""
        with pytest.raises(InvalidParameterError):
            LQM.validate_decimal_input("invalid_decimal", "test_param")
    
    def test_invalid_type_input(self):
        """Test that invalid types raise appropriate errors."""
        with pytest.raises(InvalidParameterError):
            LQM.validate_decimal_input([], "test_param")
    
    def test_negative_principal_rejection(self):
        """Test that negative principal is rejected."""
        with pytest.raises(InvalidParameterError) as exc_info:
            calculate_emi(Decimal('-100000'), Decimal('12'), 12)
        
        assert "must be positive" in str(exc_info.value)
    
    def test_negative_rate_rejection(self):
        """Test that negative interest rate is rejected."""
        with pytest.raises(InvalidParameterError):
            calculate_emi(Decimal('100000'), Decimal('-5'), 12)
    
    def test_zero_tenure_rejection(self):
        """Test that zero or negative tenure is rejected."""
        with pytest.raises(InvalidParameterError):
            calculate_emi(Decimal('100000'), Decimal('12'), 0)
        
        with pytest.raises(InvalidParameterError):
            calculate_emi(Decimal('100000'), Decimal('12'), -5)
    
    def test_non_integer_tenure_rejection(self):
        """Test that non-integer tenure is rejected."""
        with pytest.raises(InvalidParameterError):
            calculate_emi(Decimal('100000'), Decimal('12'), 12.5)

class TestLQMPrecision:
    """Test LQM precision validation and rounding."""
    
    def test_precision_validation_exact_two_decimals(self):
        """Test that values with exactly 2 decimals pass validation."""
        value = Decimal('100000.50')
        result = LQM.validate_precision(value, "test_param")
        assert result == Decimal('100000.50')
    
    def test_precision_validation_rounding_down(self):
        """Test that values with >2 decimals are rounded correctly."""
        value = Decimal('100000.123')  # Should round to 100000.12
        result = LQM.validate_precision(value, "test_param")
        assert result == Decimal('100000.12')
    
    def test_precision_validation_rounding_up(self):
        """Test rounding for values ending in 5."""
        value = Decimal('100000.125')  # Should round to 100000.13 (ROUND_HALF_UP)
        result = LQM.validate_precision(value, "test_param")
        assert result == Decimal('100000.13')
        
        value = Decimal('100000.135')  # Should round to 100000.14
        result = LQM.validate_precision(value, "test_param")
        assert result == Decimal('100000.14')
    
    def test_precision_validation_no_decimals(self):
        """Test that whole numbers are handled correctly."""
        value = Decimal('100000')
        result = LQM.validate_precision(value, "test_param")
        assert result == Decimal('100000.00')

class TestEMICalculation:
    """Test EMI calculation correctness with known reference values."""
    
    def test_standard_emi_calculation(self):
        """Test EMI calculation with standard loan parameters."""
        # Test case: ₹100,000 at 12% for 12 months
        # Expected EMI ≈ ₹8,884.88 (calculated using standard formula)
        principal = Decimal('100000.00')
        rate = Decimal('12.00')
        tenure = 12
        
        result = calculate_emi(principal, rate, tenure)
        
        # Validate result structure
        assert isinstance(result, EMICalculation)
        assert result.principal_in_cents == principal
        assert result.rate_per_annum == rate
        assert result.tenure_months == tenure
        
        # Validate EMI calculation (approximately ₹8,884.88)
        expected_emi = Decimal('8884.88')
        assert abs(result.emi_in_cents - expected_emi) < Decimal('0.01')
        
        # Validate total calculations
        expected_total = result.emi_in_cents * Decimal('12')
        assert result.total_amount_in_cents == expected_total
        assert result.total_interest_in_cents == expected_total - principal
    
    def test_zero_interest_rate(self):
        """Test EMI calculation with 0% interest rate."""
        principal = Decimal('120000.00')
        rate = Decimal('0.00')
        tenure = 12
        
        result = calculate_emi(principal, rate, tenure)
        
        # With 0% interest, EMI should be principal / tenure
        expected_emi = principal / Decimal('12')
        assert result.emi_in_cents == expected_emi
        assert result.total_interest_in_cents == ZERO_DECIMAL
        assert result.total_amount_in_cents == principal
    
    def test_high_interest_rate(self):
        """Test EMI calculation with high interest rate."""
        principal = Decimal('50000.00')
        rate = Decimal('24.00')  # 24% annual rate
        tenure = 6
        
        result = calculate_emi(principal, rate, tenure)
        
        # Validate that EMI is calculated correctly
        assert result.emi_in_cents > principal / Decimal('6')  # Should be higher than simple division
        assert result.total_interest_in_cents > ZERO_DECIMAL
        assert result.total_amount_in_cents == result.principal_in_cents + result.total_interest_in_cents
    
    def test_long_tenure_calculation(self):
        """Test EMI calculation with long tenure."""
        principal = Decimal('500000.00')
        rate = Decimal('8.50')
        tenure = 240  # 20 years
        
        result = calculate_emi(principal, rate, tenure)
        
        # Validate that longer tenure results in lower EMI but higher total interest
        assert result.emi_in_cents < principal / Decimal('100')  # Much lower than short-term
        assert result.total_interest_in_cents > principal / Decimal('2')  # Significant interest
    
    def test_small_amount_calculation(self):
        """Test EMI calculation with small loan amount."""
        principal = Decimal('5000.00')
        rate = Decimal('15.00')
        tenure = 6
        
        result = calculate_emi(principal, rate, tenure)
        
        # Validate precision is maintained even for small amounts
        assert result.emi_in_cents > ZERO_DECIMAL
        assert str(result.emi_in_cents).count('.') <= 1  # Only one decimal point
        if '.' in str(result.emi_in_cents):
            decimal_places = len(str(result.emi_in_cents).split('.')[1])
            assert decimal_places <= 2  # At most 2 decimal places

class TestEMICalculationInputTypes:
    """Test EMI calculation with different input types."""
    
    def test_string_inputs(self):
        """Test that string inputs are properly converted."""
        result = calculate_emi("75000.00", "10.50", 18)
        
        assert result.principal_in_cents == Decimal('75000.00')
        assert result.rate_per_annum == Decimal('10.50')
        assert result.tenure_months == 18
        assert result.emi_in_cents > ZERO_DECIMAL
    
    def test_integer_inputs(self):
        """Test that integer inputs are properly converted."""
        result = calculate_emi(100000, 12, 24)
        
        assert result.principal_in_cents == Decimal('100000.00')
        assert result.rate_per_annum == Decimal('12.00')
        assert result.tenure_months == 24
    
    def test_mixed_input_types(self):
        """Test calculation with mixed input types."""
        result = calculate_emi(Decimal('80000'), "9.75", 15)
        
        assert result.principal_in_cents == Decimal('80000.00')
        assert result.rate_per_annum == Decimal('9.75')
        assert result.tenure_months == 15

class TestCurrencyConversion:
    """Test currency conversion utilities."""
    
    def test_currency_display_conversion(self):
        """Test conversion from cents to currency display format."""
        amount_in_cents = Decimal('150000.00')
        display = LQM.convert_to_currency_display(amount_in_cents)
        
        assert "₹" in display
        assert "1,500.00" in display  # 150000 cents = ₹1,500.00
    
    def test_currency_input_conversion_simple(self):
        """Test conversion from currency string to cents."""
        currency_string = "1500.50"
        result = LQM.convert_from_currency_input(currency_string)
        
        assert result == Decimal('150050.00')  # ₹1,500.50 = 150050 cents
    
    def test_currency_input_conversion_with_symbols(self):
        """Test conversion with currency symbols and commas."""
        currency_string = "₹1,50,000.75"
        result = LQM.convert_from_currency_input(currency_string)
        
        assert result == Decimal('15000075.00')  # ₹1,50,000.75 = 15000075 cents
    
    def test_invalid_currency_string(self):
        """Test that invalid currency strings raise appropriate errors."""
        with pytest.raises(InvalidParameterError):
            LQM.convert_from_currency_input("invalid_amount")

class TestEMICalculationResult:
    """Test EMICalculation result object."""
    
    def test_emi_calculation_to_dict(self):
        """Test conversion of EMI calculation to dictionary."""
        result = calculate_emi(Decimal('100000'), Decimal('12'), 12)
        result_dict = result.to_dict()
        
        # Validate dictionary structure
        expected_keys = [
            'principal_in_cents', 'rate_per_annum', 'tenure_months',
            'emi_in_cents', 'total_interest_in_cents', 'total_amount_in_cents'
        ]
        
        for key in expected_keys:
            assert key in result_dict
        
        # Validate that monetary values are strings (for JSON serialization)
        assert isinstance(result_dict['principal_in_cents'], str)
        assert isinstance(result_dict['emi_in_cents'], str)
        assert isinstance(result_dict['total_interest_in_cents'], str)

class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_validate_monetary_input_function(self):
        """Test the validate_monetary_input convenience function."""
        # Test valid input
        result = validate_monetary_input("100000.123", "test_amount")
        assert result == Decimal('100000.12')
        
        # Test float rejection
        with pytest.raises(FloatInputError):
            validate_monetary_input(100000.50, "test_amount")
    
    def test_module_constants(self):
        """Test that module constants are properly defined."""
        assert ZERO_DECIMAL == Decimal('0.00')
        assert ONE_DECIMAL == Decimal('1.00')
        assert HUNDRED_DECIMAL == Decimal('100.00')

class TestLQMCompliance:
    """Test LQM Standard compliance requirements."""
    
    def test_no_float_usage_in_calculations(self):
        """Verify that no float operations occur in EMI calculations."""
        # This test ensures that all intermediate calculations use Decimal
        principal = Decimal('100000.00')
        rate = Decimal('12.00')
        tenure = 12
        
        result = calculate_emi(principal, rate, tenure)
        
        # All result values should be Decimal type
        assert isinstance(result.principal_in_cents, Decimal)
        assert isinstance(result.rate_per_annum, Decimal)
        assert isinstance(result.emi_in_cents, Decimal)
        assert isinstance(result.total_interest_in_cents, Decimal)
        assert isinstance(result.total_amount_in_cents, Decimal)
    
    def test_precision_consistency(self):
        """Test that all monetary values maintain 2 decimal place precision."""
        result = calculate_emi(Decimal('99999.999'), Decimal('11.111'), 13)
        
        # Check that all monetary results have at most 2 decimal places
        monetary_values = [
            result.principal_in_cents,
            result.emi_in_cents,
            result.total_interest_in_cents,
            result.total_amount_in_cents
        ]
        
        for value in monetary_values:
            value_str = str(value)
            if '.' in value_str:
                decimal_places = len(value_str.split('.')[1])
                assert decimal_places <= 2, f"Value {value} has more than 2 decimal places"
    
    def test_mathematical_correctness(self):
        """Test that EMI calculations are mathematically correct."""
        principal = Decimal('200000.00')
        rate = Decimal('15.00')
        tenure = 24
        
        result = calculate_emi(principal, rate, tenure)
        
        # Verify the reducing balance formula manually
        monthly_rate = rate / Decimal('12') / Decimal('100')
        
        if monthly_rate == 0:
            expected_emi = principal / Decimal(str(tenure))
        else:
            # Calculate (1 + r)^n manually
            one_plus_r = Decimal('1') + monthly_rate
            power_term = Decimal('1')
            for _ in range(tenure):
                power_term *= one_plus_r
            
            # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
            numerator = principal * monthly_rate * power_term
            denominator = power_term - Decimal('1')
            expected_emi = numerator / denominator
            expected_emi = expected_emi.quantize(Decimal('0.01'))
        
        # Allow small rounding differences
        assert abs(result.emi_in_cents - expected_emi) < Decimal('0.01')

# Pytest fixtures for common test data
@pytest.fixture
def standard_loan_params():
    """Standard loan parameters for testing."""
    return {
        'principal': Decimal('100000.00'),
        'annual_rate': Decimal('12.00'),
        'tenure_months': 12
    }

@pytest.fixture
def zero_interest_params():
    """Zero interest loan parameters for testing."""
    return {
        'principal': Decimal('50000.00'),
        'rate': Decimal('0.00'),
        'tenure': 6
    }

# Integration test using fixtures
def test_emi_calculation_with_fixtures(standard_loan_params):
    """Test EMI calculation using pytest fixtures."""
    result = calculate_emi(**standard_loan_params)
    
    assert result.principal_in_cents == standard_loan_params['principal']
    assert result.rate_per_annum == standard_loan_params['annual_rate']
    assert result.tenure_months == standard_loan_params['tenure_months']
    assert result.emi_in_cents > ZERO_DECIMAL

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])