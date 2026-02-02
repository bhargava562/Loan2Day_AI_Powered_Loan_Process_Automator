# Property-Based Testing Suite for Loan2Day

This directory contains comprehensive property-based tests using Hypothesis to verify universal properties of the Loan2Day fintech platform across all possible input scenarios.

## 🧪 Test Philosophy

Property-based testing validates **universal properties** that must hold true for ALL possible inputs, not just specific test cases. This approach:

- **Discovers edge cases** automatically through randomized input generation
- **Validates business logic** across the entire input space
- **Ensures mathematical accuracy** with the LQM Standard (Decimal precision)
- **Verifies security constraints** against malicious inputs
- **Tests Plan B scenarios** for rejection recovery workflows

## 📁 Test Structure

### Core Test Files

- **`test_pdf_properties.py`** - PDF Generation Service property tests
- **`test_lqm_properties.py`** - Loan Quotation Math (LQM) property tests  
- **`test_sgs_properties.py`** - Secure Gateway Service (SGS) property tests
- **`test_error_handling_properties.py`** - Error handling and recovery property tests

## 🎯 PDF Generation Properties (test_pdf_properties.py)

### TestPDFGenerationCompleteness
**Feature:** loan2day, Property 17: PDF Generation Completeness

- **`test_pdf_data_field_completeness_property`** - Verifies ALL PDFs contain required data fields
- **`test_pdf_generation_consistency_property`** - Ensures deterministic PDF generation
- **`test_pdf_security_properties`** - Validates security requirements (no executable content)
- **`test_pdf_mathematical_accuracy_property`** - Verifies LQM Standard mathematical precision
- **`test_pdf_tanglish_support_property`** - Tests mixed-language (Tamil+English) support
- **`test_pdf_plan_b_scenario_property`** - Validates Plan B rejection recovery workflows
- **`test_pdf_concurrent_generation_property`** - Tests thread-safe concurrent operations

### TestPDFSecurityProperties
**Feature:** loan2day, Property 17: PDF Generation Security

- **`test_pdf_input_sanitization_property`** - Validates input sanitization against malicious content
- **`test_pdf_float_rejection_property`** - Enforces LQM Standard (rejects float inputs)
- **`test_pdf_negative_amount_rejection_property`** - Validates business logic constraints
- **`test_pdf_invalid_tenure_rejection_property`** - Tests tenure validation (1-360 months)
- **`test_pdf_empty_name_rejection_property`** - Validates required field enforcement

### TestPDFPerformanceProperties
**Feature:** loan2day, Property 17: PDF Generation Performance

- **`test_pdf_generation_speed_property`** - Ensures real-time generation (< 5 seconds)
- **`test_pdf_file_size_optimization_property`** - Validates optimal file sizes (50KB-500KB)
- **`test_pdf_memory_usage_property`** - Tests memory efficiency in batch operations

### TestPDFRegulatoryComplianceProperties
**Feature:** loan2day, Property 17: PDF Regulatory Compliance

- **`test_pdf_mandatory_disclosures_property`** - Verifies RBI compliance requirements
- **`test_pdf_high_interest_rate_warnings_property`** - Tests warnings for rates > 25%
- **`test_pdf_large_loan_compliance_property`** - Validates additional requirements for loans > ₹50L

### TestPDFAgentStateIntegrationProperties
**Feature:** loan2day, Property 17: PDF Agent Integration

- **`test_pdf_agent_state_data_extraction_property`** - Tests Master-Worker agent pattern compliance
- **`test_pdf_agent_state_validation_property`** - Validates AgentState data completeness
- **`test_pdf_plan_b_agent_state_property`** - Tests Plan B scenarios from AgentState

### TestPDFEdgeCaseProperties
**Feature:** loan2day, Property 17: PDF Edge Case Handling

- **`test_pdf_unicode_handling_property`** - Tests Unicode/Tamil character support
- **`test_pdf_micro_loan_property`** - Validates micro-finance scenarios
- **`test_pdf_file_system_edge_cases_property`** - Tests filesystem limitations
- **`test_pdf_concurrent_file_access_property`** - Validates concurrent directory access

## 🔧 Running Property Tests

### Individual Test Classes
```bash
# Run PDF generation completeness tests
python -m pytest tests/property/test_pdf_properties.py::TestPDFGenerationCompleteness -v

# Run security property tests
python -m pytest tests/property/test_pdf_properties.py::TestPDFSecurityProperties -v

# Run performance property tests  
python -m pytest tests/property/test_pdf_properties.py::TestPDFPerformanceProperties -v
```

### Specific Properties
```bash
# Test PDF data field completeness (100 iterations)
python -m pytest tests/property/test_pdf_properties.py::TestPDFGenerationCompleteness::test_pdf_data_field_completeness_property -v

# Test LQM Standard enforcement
python -m pytest tests/property/test_pdf_properties.py::TestPDFSecurityProperties::test_pdf_float_rejection_property -v

# Test Plan B scenarios
python -m pytest tests/property/test_pdf_properties.py::TestPDFGenerationCompleteness::test_pdf_plan_b_scenario_property -v
```

### Full Property Test Suite
```bash
# Run all property tests (WARNING: Takes 10-30 minutes)
python -m pytest tests/property/ -v --tb=short

# Run with statistics
python -m pytest tests/property/ -v --hypothesis-show-statistics
```

## 📊 Test Coverage Metrics

### Input Space Coverage
- **Monetary Values:** ₹1,000 to ₹50 Lakh (using Decimal precision)
- **Interest Rates:** 0% to 50% per annum
- **Loan Tenure:** 6 to 360 months (0.5 to 30 years)
- **Names:** Unicode support including Tamil/Hindi characters
- **Edge Cases:** Micro-loans, large loans, concurrent operations

### Property Validation
- **Mathematical Accuracy:** 100% LQM Standard compliance
- **Security:** Input sanitization, float rejection, path traversal protection
- **Performance:** Real-time generation, memory efficiency, file size optimization
- **Regulatory:** RBI compliance, mandatory disclosures, audit trails
- **Integration:** Master-Worker agent pattern, AgentState validation

## 🚨 Critical Properties

### LQM Standard Enforcement
```python
# NEVER use float for currency - must use Decimal
with pytest.raises((FloatInputError, DataValidationError)):
    SanctionLetterData(
        loan_amount_in_cents=1000.50,  # Float - REJECTED
        # ... other params
    )
```

### Plan B Rejection Recovery
```python
# Plan B scenarios must generate valid PDFs
plan_b_data = SanctionLetterData(
    loan_amount_in_cents=reduced_amount,  # 70% of original
    interest_rate=higher_rate,            # +3% rate increase
    # ... Plan B terms
)
pdf_path = pdf_service.generate_sanction_letter(plan_b_data)
assert os.path.exists(pdf_path)  # Must succeed
```

### Security Validation
```python
# Malicious inputs must be sanitized or rejected
potentially_malicious_name = "../../../etc/passwd"
# PDF service must handle safely without path traversal
```

## 🎯 Property Test Benefits

1. **Comprehensive Coverage** - Tests ALL possible input combinations, not just happy paths
2. **Edge Case Discovery** - Automatically finds boundary conditions and corner cases  
3. **Mathematical Verification** - Ensures LQM Standard precision across all calculations
4. **Security Validation** - Tests against malicious inputs and injection attacks
5. **Performance Assurance** - Validates real-time constraints under all conditions
6. **Regulatory Compliance** - Ensures RBI requirements are met for all loan scenarios

## 📈 Hypothesis Configuration

### Test Settings
- **Max Examples:** 10-100 iterations per property (configurable)
- **Deadline:** Disabled for PDF generation tests (I/O intensive)
- **Database:** Persistent example storage for regression testing
- **Shrinking:** Automatic minimal failing case discovery

### Strategy Configuration
```python
# Monetary amounts (in cents for precision)
loan_amount_strategy = st.decimals(
    min_value=Decimal('100000'),     # ₹1,000
    max_value=Decimal('5000000000'), # ₹50 Lakh
    places=2
)

# Interest rates (percentage per annum)
interest_rate_strategy = st.decimals(
    min_value=Decimal('0.00'),
    max_value=Decimal('50.00'),
    places=2
)
```

## 🔍 Debugging Property Failures

### Hypothesis Shrinking
When a property test fails, Hypothesis automatically finds the **minimal failing example**:

```
Falsifying example: test_pdf_data_field_completeness_property(
    borrower_name='AA',
    loan_amount=Decimal('100000.00'),
    emi_amount=Decimal('50000.00'),
    interest_rate=Decimal('0.01'),
    tenure=6
)
```

### Root Cause Analysis
1. **Check LQM Standard** - Are all monetary values using Decimal?
2. **Validate Business Logic** - Do the inputs make mathematical sense?
3. **Review Agent State** - Is the Master-Worker pattern being followed?
4. **Examine Security** - Are inputs properly sanitized?

## 🚀 Continuous Integration

Property tests are integrated into the CI/CD pipeline:

```yaml
# .github/workflows/property-tests.yml
- name: Run Property Tests
  run: |
    python -m pytest tests/property/ \
      --hypothesis-show-statistics \
      --tb=short \
      --maxfail=5
```

## 📚 References

- **Hypothesis Documentation:** https://hypothesis.readthedocs.io/
- **Property-Based Testing:** https://increment.com/testing/in-praise-of-property-based-testing/
- **LQM Standard:** `app/core/lqm.py` - Loan Quotation Math precision requirements
- **Master-Worker Pattern:** `app/agents/` - Agent orchestration architecture

---

**Author:** Lead AI Architect, Loan2Day Fintech Platform  
**Last Updated:** February 2026  
**Test Framework:** Hypothesis 6.92.1, pytest 7.4.3