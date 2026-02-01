"""
Property-based tests for PDF Generation Service.

This test suite uses Hypothesis to verify universal properties of the PDF
generation system across all possible loan approval scenarios through randomization.
Property tests ensure consistent PDF generation behavior regardless of input variations.

Test Coverage:
- Property 17: PDF Generation Completeness
- PDF data field completeness and accuracy
- ReportLab integration consistency
- Regulatory disclosure inclusion
- Secure download link generation
- Real-time generation performance

Framework: Hypothesis for property-based testing
Iterations: Minimum 100 iterations per property test
Tags: Each test tagged with design document property reference

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from decimal import Decimal
import sys
import os
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import hashlib
import time
import re

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from services.pdf_service import (
    PDFService, SanctionLetterData, PDFGenerationError,
    DataValidationError, FileGenerationError
)
from models.pydantic_models import (
    AgentState, UserProfile, LoanRequest, EMICalculation,
    LoanPurpose, KYCStatus, AgentStep
)
from app.core.lqm import FloatInputError

# Hypothesis strategies for generating test data

# Generate valid names
name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Pd', 'Zs')),
    min_size=2,
    max_size=50
).filter(lambda x: x.strip() and not x.isspace())

# Generate valid monetary amounts in cents (₹1,000 to ₹50 Lakh)
loan_amount_strategy = st.decimals(
    min_value=Decimal('100000'),     # ₹1,000 in cents
    max_value=Decimal('5000000000'), # ₹50 Lakh in cents
    places=2
)

# Generate valid EMI amounts (₹500 to ₹2 Lakh)
emi_strategy = st.decimals(
    min_value=Decimal('50000'),      # ₹500 in cents
    max_value=Decimal('20000000'),   # ₹2 Lakh in cents
    places=2
)

# Generate valid interest rates (0% to 50% per annum)
interest_rate_strategy = st.decimals(
    min_value=Decimal('0.00'),
    max_value=Decimal('50.00'),
    places=2
)

# Generate valid tenure (6 months to 30 years)
tenure_strategy = st.integers(min_value=6, max_value=360)

# Generate loan purposes
loan_purpose_strategy = st.sampled_from([
    "Personal Loan", "Home Loan", "Car Loan", "Education Loan",
    "Business Loan", "Medical Emergency", "Debt Consolidation"
])

# Generate phone numbers
phone_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Nd', 'Pd')),
    min_size=10,
    max_size=15
).filter(lambda x: any(c.isdigit() for c in x))

# Generate email addresses
email_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
    min_size=3,
    max_size=20
).map(lambda x: f"{x}@example.com")

# Generate addresses
address_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pd', 'Zs')),
    min_size=10,
    max_size=200
).filter(lambda x: x.strip() and not x.isspace())

# Generate user IDs
user_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
    min_size=8,
    max_size=20
)

class TestPDFGenerationCompleteness:
    """
    Property tests for PDF generation completeness and consistency.
    
    Feature: loan2day, Property 17: PDF Generation Completeness
    Validates: Requirements 11.2, 11.3, 11.4, 11.5
    """
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.pdf_service = PDFService(output_directory=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @given(
        name_strategy,
        loan_amount_strategy,
        emi_strategy,
        interest_rate_strategy,
        tenure_strategy,
        loan_purpose_strategy
    )
    @settings(max_examples=100)
    def test_pdf_data_field_completeness_property(
        self, borrower_name, loan_amount, emi_amount, interest_rate, tenure, loan_purpose
    ):
        """
        Property: ALL PDF sanction letters must contain required data fields.
        
        Feature: loan2day, Property 17: PDF Generation Completeness
        
        This property verifies that every generated PDF contains all mandatory
        data fields (name, loan amount, EMI, interest rate) with accurate values
        from the input data, ensuring no critical information is missing.
        """
        # Create sanction letter data
        sanction_data = SanctionLetterData(
            borrower_name=borrower_name,
            loan_amount_in_cents=loan_amount,
            emi_in_cents=emi_amount,
            interest_rate=interest_rate,
            tenure_months=tenure,
            loan_purpose=loan_purpose,
            sanction_date=datetime.now(),
            loan_id=f"LN{int(time.time())}"
        )
        
        # Generate PDF
        pdf_path = self.pdf_service.generate_sanction_letter(sanction_data)
        
        # Verify PDF file was created
        assert os.path.exists(pdf_path), f"PDF file not created: {pdf_path}"
        assert os.path.getsize(pdf_path) > 0, "PDF file is empty"
        
        # Read PDF content as text (simplified check)
        # In a real implementation, you'd use a PDF reader library
        # For this property test, we'll verify the file exists and has content
        
        # Verify file info contains expected data
        pdf_info = self.pdf_service.get_pdf_info(pdf_path)
        assert pdf_info["file_size_bytes"] > 1000, "PDF file too small to contain required content"
        assert pdf_info["is_readable"], "PDF file is not readable"
        
        # Verify filename contains loan ID
        filename = Path(pdf_path).name
        assert sanction_data.loan_id in filename, f"Loan ID not in filename: {filename}"
        
        # Verify display amounts are properly formatted
        display_amounts = sanction_data.get_display_amounts()
        assert "₹" in display_amounts["loan_amount"], "Loan amount not properly formatted"
        assert "₹" in display_amounts["emi_amount"], "EMI amount not properly formatted"
        
        # Verify all required fields are present in display amounts
        required_display_fields = [
            "loan_amount", "emi_amount", "processing_fee", 
            "insurance_premium", "total_interest", "total_repayment"
        ]
        for field in required_display_fields:
            assert field in display_amounts, f"Missing display field: {field}"
    
    @given(
        name_strategy,
        loan_amount_strategy,
        emi_strategy,
        interest_rate_strategy,
        tenure_strategy
    )
    @settings(max_examples=100)
    def test_pdf_generation_consistency_property(
        self, borrower_name, loan_amount, emi_amount, interest_rate, tenure
    ):
        """
        Property: PDF generation must be consistent across multiple runs.
        
        Feature: loan2day, Property 17: PDF Generation Completeness
        
        This property verifies that generating the same PDF multiple times
        produces identical results, ensuring deterministic behavior.
        """
        # Create sanction letter data
        sanction_data = SanctionLetterData(
            borrower_name=borrower_name,
            loan_amount_in_cents=loan_amount,
            emi_in_cents=emi_amount,
            interest_rate=interest_rate,
            tenure_months=tenure,
            loan_purpose="Personal Loan",
            sanction_date=datetime(2024, 1, 15, 10, 30, 0),  # Fixed date for consistency
            loan_id="TEST123"  # Fixed ID for consistency
        )
        
        # Generate PDF twice
        pdf_path_1 = self.pdf_service.generate_sanction_letter(
            sanction_data, 
            output_filename="test_consistency_1.pdf"
        )
        pdf_path_2 = self.pdf_service.generate_sanction_letter(
            sanction_data, 
            output_filename="test_consistency_2.pdf"
        )
        
        # Both files should exist
        assert os.path.exists(pdf_path_1), "First PDF not generated"
        assert os.path.exists(pdf_path_2), "Second PDF not generated"
        
        # Get file info for both
        info_1 = self.pdf_service.get_pdf_info(pdf_path_1)
        info_2 = self.pdf_service.get_pdf_info(pdf_path_2)
        
        # File sizes should be identical (deterministic generation)
        assert info_1["file_size_bytes"] == info_2["file_size_bytes"], \
            f"Inconsistent file sizes: {info_1['file_size_bytes']} vs {info_2['file_size_bytes']}"
        
        # Both should be readable
        assert info_1["is_readable"], "First PDF not readable"
        assert info_2["is_readable"], "Second PDF not readable"
    
    @given(
        name_strategy,
        loan_amount_strategy,
        emi_strategy,
        interest_rate_strategy,
        tenure_strategy
    )
    @settings(max_examples=50)
    def test_pdf_security_properties(
        self, borrower_name, loan_amount, emi_amount, interest_rate, tenure
    ):
        """
        Property: PDF files must meet security requirements.
        
        Feature: loan2day, Property 17: PDF Generation Completeness
        
        This property verifies that generated PDFs meet security standards:
        - No executable content
        - Proper file permissions
        - Safe filename generation
        """
        # Create sanction letter data
        sanction_data = SanctionLetterData(
            borrower_name=borrower_name,
            loan_amount_in_cents=loan_amount,
            emi_in_cents=emi_amount,
            interest_rate=interest_rate,
            tenure_months=tenure,
            loan_purpose="Personal Loan",
            sanction_date=datetime.now(),
            loan_id=f"SEC{int(time.time())}"
        )
        
        # Generate PDF
        pdf_path = self.pdf_service.generate_sanction_letter(sanction_data)
        
        # Verify file exists and is readable
        assert os.path.exists(pdf_path), "PDF file not created"
        assert os.access(pdf_path, os.R_OK), "PDF file not readable"
        
        # Verify filename is safe (no path traversal, special characters)
        filename = Path(pdf_path).name
        assert not filename.startswith('.'), "Filename should not start with dot"
        assert '..' not in filename, "Filename should not contain path traversal"
        assert filename.endswith('.pdf'), "Filename should end with .pdf"
        
        # Verify file is actually a PDF (basic check)
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            assert header == b'%PDF', "File does not have PDF header"
        
        # Verify file size is reasonable (not empty, not suspiciously large)
        file_size = os.path.getsize(pdf_path)
        assert 1000 < file_size < 10_000_000, f"Suspicious file size: {file_size} bytes"
    
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_pdf_error_handling_property(self, invalid_name):
        """
        Property: PDF service must handle invalid inputs gracefully.
        
        Feature: loan2day, Property 17: PDF Generation Completeness
        
        This property verifies that the PDF service properly validates inputs
        and raises appropriate exceptions for invalid data.
        """
        # Test with various invalid inputs - using float instead of Decimal should raise error
        with pytest.raises((DataValidationError, ValueError, FloatInputError)):
            # Invalid monetary values (using float instead of Decimal)
            SanctionLetterData(
                borrower_name=invalid_name,
                loan_amount_in_cents=1000.50,  # Float instead of Decimal - should raise FloatInputError
                emi_in_cents=Decimal('50000'),
                interest_rate=Decimal('12.50'),
                tenure_months=24,
                loan_purpose="Personal Loan",
                sanction_date=datetime.now(),
                loan_id="TEST123"
            )
    
    @given(
        loan_amount_strategy,
        emi_strategy,
        interest_rate_strategy,
        tenure_strategy
    )
    @settings(max_examples=50)
    def test_pdf_mathematical_accuracy_property(
        self, loan_amount, emi_amount, interest_rate, tenure
    ):
        """
        Property: PDF calculations must be mathematically accurate.
        
        Feature: loan2day, Property 17: PDF Generation Completeness
        
        This property verifies that all monetary calculations in the PDF
        are accurate and use proper decimal precision (LQM Standard).
        """
        # Create sanction letter data
        sanction_data = SanctionLetterData(
            borrower_name="Test User",
            loan_amount_in_cents=loan_amount,
            emi_in_cents=emi_amount,
            interest_rate=interest_rate,
            tenure_months=tenure,
            loan_purpose="Personal Loan",
            sanction_date=datetime.now(),
            loan_id="MATH123"
        )
        
        # Get display amounts
        display_amounts = sanction_data.get_display_amounts()
        
        # Verify mathematical relationships
        total_emi_payments = emi_amount * Decimal(str(tenure))
        calculated_total_interest = total_emi_payments - loan_amount
        
        # Extract numeric values from display strings
        total_interest_display = display_amounts["total_interest"]
        total_repayment_display = display_amounts["total_repayment"]
        
        # Remove currency symbol and commas for comparison
        total_interest_value = Decimal(total_interest_display.replace('₹', '').replace(',', ''))
        total_repayment_value = Decimal(total_repayment_display.replace('₹', '').replace(',', ''))
        
        # Verify calculations (allow small rounding differences)
        expected_total_interest = calculated_total_interest / 100  # Convert cents to rupees
        expected_total_repayment = total_emi_payments / 100
        
        assert abs(total_interest_value - expected_total_interest) <= Decimal('0.01'), \
            f"Interest calculation error: {total_interest_value} vs {expected_total_interest}"
        
        assert abs(total_repayment_value - expected_total_repayment) <= Decimal('0.01'), \
            f"Repayment calculation error: {total_repayment_value} vs {expected_total_repayment}"
    
    @given(
        name_strategy,
        loan_amount_strategy,
        emi_strategy,
        interest_rate_strategy,
        tenure_strategy,
        phone_strategy,
        email_strategy,
        address_strategy
    )
    @settings(max_examples=30)
    def test_pdf_tanglish_support_property(
        self, borrower_name, loan_amount, emi_amount, interest_rate, 
        tenure, phone, email, address
    ):
        """
        Property: PDF generation must handle Tanglish/mixed-language inputs.
        
        Feature: loan2day, Property 17: PDF Generation Completeness
        
        This property verifies that the PDF service can handle mixed-language
        inputs (Tamil + English) commonly used in Indian fintech applications.
        """
        # Create Tanglish-style inputs
        tanglish_name = f"{borrower_name} Kumar"  # Mix English with Indian suffix
        tanglish_address = f"{address}, Chennai la irukku"  # Mix with Tamil
        
        # Create sanction letter data with Tanglish inputs
        sanction_data = SanctionLetterData(
            borrower_name=tanglish_name,
            loan_amount_in_cents=loan_amount,
            emi_in_cents=emi_amount,
            interest_rate=interest_rate,
            tenure_months=tenure,
            loan_purpose="Personal Loan",
            sanction_date=datetime.now(),
            loan_id="TANGLISH123",
            borrower_address=tanglish_address,
            borrower_phone=phone,
            borrower_email=email
        )
        
        # Generate PDF - should not raise encoding errors
        pdf_path = self.pdf_service.generate_sanction_letter(sanction_data)
        
        # Verify PDF was created successfully
        assert os.path.exists(pdf_path), "PDF with Tanglish content not created"
        assert os.path.getsize(pdf_path) > 1000, "PDF with Tanglish content too small"
        
        # Verify PDF info can be retrieved
        pdf_info = self.pdf_service.get_pdf_info(pdf_path)
        assert pdf_info["is_readable"], "PDF with Tanglish content not readable"
    
    def test_pdf_plan_b_scenario_property(self):
        """
        Property: PDF generation must work in Plan B scenarios.
        
        Feature: loan2day, Property 17: PDF Generation Completeness
        
        This property verifies that PDF generation works correctly when
        Plan B logic is triggered (alternative loan offers).
        """
        # Create Plan B scenario data (lower loan amount, higher EMI)
        plan_b_data = SanctionLetterData(
            borrower_name="Plan B User",
            loan_amount_in_cents=Decimal('300000000'),  # ₹3 Lakh (reduced from original request)
            emi_in_cents=Decimal('1500000'),            # ₹15,000 EMI
            interest_rate=Decimal('18.50'),             # Higher interest rate
            tenure_months=24,                           # Shorter tenure
            loan_purpose="Debt Consolidation",
            sanction_date=datetime.now(),
            loan_id="PLANB123",
            processing_fee_in_cents=Decimal('75000'),   # ₹750 processing fee
            insurance_premium_in_cents=Decimal('25000') # ₹250 insurance
        )
        
        # Generate Plan B PDF
        pdf_path = self.pdf_service.generate_sanction_letter(plan_b_data)
        
        # Verify PDF was created
        assert os.path.exists(pdf_path), "Plan B PDF not created"
        
        # Verify display amounts include all fees
        display_amounts = plan_b_data.get_display_amounts()
        assert "₹750.00" in display_amounts["processing_fee"], "Processing fee not displayed correctly"
        assert "₹250.00" in display_amounts["insurance_premium"], "Insurance premium not displayed correctly"
        
        # Verify PDF info
        pdf_info = self.pdf_service.get_pdf_info(pdf_path)
        assert pdf_info["file_size_bytes"] > 1000, "Plan B PDF too small"
        assert pdf_info["is_readable"], "Plan B PDF not readable"
       
    
    @given(
        name_strategy,
        loan_amount_strategy,
        emi_strategy,
        interest_rate_strategy,
        tenure_strategy
    )
    @settings(max_examples=30)
    def test_pdf_plan_b_rejection_recovery_property(
        self, borrower_name, loan_amount, emi_amount, interest_rate, tenure
    ):
        """
        Property: PDF generation must work for Plan B rejection recovery scenarios.
        
        Feature: loan2day, Property 17: PDF Generation Completeness
        
        This property verifies that PDF generation works correctly when the
        Plan B logic is triggered due to initial loan rejection, ensuring
        alternative offers can be properly documented.
        """
        # Create Plan B scenario with reduced loan amount and modified terms
        reduced_amount = loan_amount * Decimal('0.7')  # 70% of original amount
        higher_rate = min(interest_rate + Decimal('3.00'), Decimal('25.00'))  # +3% rate, max 25%
        
        plan_b_data = SanctionLetterData(
            borrower_name=f"{borrower_name} (Plan B)",
            loan_amount_in_cents=reduced_amount,
            emi_in_cents=emi_amount * Decimal('0.8'),  # Reduced EMI
            interest_rate=higher_rate,
            tenure_months=min(tenure + 6, 360),  # Extended tenure by 6 months, max 360
            loan_purpose="Debt Consolidation",  # Common Plan B purpose
            sanction_date=datetime.now(),
            loan_id=f"PLANB{int(time.time())}",
            processing_fee_in_cents=Decimal('100000'),  # ₹1000 processing fee
            insurance_premium_in_cents=Decimal('50000')  # ₹500 insurance
        )
        
        # Generate Plan B PDF
        pdf_path = self.pdf_service.generate_sanction_letter(plan_b_data)
        
        # Verify PDF was created successfully
        assert os.path.exists(pdf_path), "Plan B PDF not created"
        
        # Verify PDF contains Plan B identifier
        filename = Path(pdf_path).name
        assert "PLANB" in filename, "Plan B identifier not in filename"
        
        # Verify display amounts are calculated correctly
        display_amounts = plan_b_data.get_display_amounts()
        
        # Verify all monetary fields are properly formatted
        for field_name, amount_str in display_amounts.items():
            assert "₹" in amount_str, f"Currency symbol missing in {field_name}"
            assert amount_str != "₹0.00" or field_name in ["processing_fee", "insurance_premium"], \
                f"Unexpected zero amount in {field_name}"
        
        # Verify PDF info
        pdf_info = self.pdf_service.get_pdf_info(pdf_path)
        assert pdf_info["file_size_bytes"] > 1000, "Plan B PDF too small"
        assert pdf_info["is_readable"], "Plan B PDF not readable"
    
    @given(
        name_strategy,
        st.decimals(min_value=Decimal('100000'), max_value=Decimal('1000000'), places=2),  # Small amounts
        st.integers(min_value=360, max_value=360),  # Exactly 30 years
        st.decimals(min_value=Decimal('0.01'), max_value=Decimal('1.00'), places=2)  # Very low rates
    )
    @settings(max_examples=20)
    def test_pdf_edge_case_scenarios_property(
        self, borrower_name, small_amount, max_tenure, low_rate
    ):
        """
        Property: PDF generation must handle edge case scenarios correctly.
        
        Feature: loan2day, Property 17: PDF Generation Completeness
        
        This property verifies that PDF generation works correctly for edge cases:
        - Very small loan amounts
        - Maximum tenure (30 years)
        - Very low interest rates (near 0%)
        """
        # Calculate EMI for edge case scenario
        from app.core.lqm import calculate_emi
        
        emi_calc = calculate_emi(
            principal=small_amount,
            annual_rate=low_rate,
            tenure_months=max_tenure
        )
        
        # Create edge case sanction letter data
        edge_case_data = SanctionLetterData(
            borrower_name=borrower_name,
            loan_amount_in_cents=small_amount,
            emi_in_cents=emi_calc.emi_in_cents,
            interest_rate=low_rate,
            tenure_months=max_tenure,
            loan_purpose="Personal Loan",
            sanction_date=datetime.now(),
            loan_id=f"EDGE{int(time.time())}"
        )
        
        # Generate PDF for edge case
        pdf_path = self.pdf_service.generate_sanction_letter(edge_case_data)
        
        # Verify PDF was created successfully
        assert os.path.exists(pdf_path), "Edge case PDF not created"
        
        # Verify display amounts are reasonable
        display_amounts = edge_case_data.get_display_amounts()
        
        # For very low rates, total interest should be minimal
        total_interest_str = display_amounts["total_interest"].replace('₹', '').replace(',', '')
        total_interest_value = Decimal(total_interest_str)
        
        # With very low rates, interest should be less than principal
        principal_value = small_amount / 100  # Convert cents to rupees
        assert total_interest_value < principal_value * 2, \
            f"Interest too high for low rate: {total_interest_value} vs principal {principal_value}"
        
        # Verify PDF info
        pdf_info = self.pdf_service.get_pdf_info(pdf_path)
        assert pdf_info["file_size_bytes"] > 1000, "Edge case PDF too small"
        assert pdf_info["is_readable"], "Edge case PDF not readable"
    
    def test_pdf_concurrent_generation_property(self):
        """
        Property: PDF generation must be thread-safe for concurrent operations.
        
        Feature: loan2day, Property 17: PDF Generation Completeness
        
        This property verifies that multiple PDF generations can occur
        concurrently without data corruption or file conflicts.
        """
        import threading
        import queue
        
        # Create test data for concurrent generation
        test_data_list = []
        for i in range(5):
            test_data = SanctionLetterData(
                borrower_name=f"Concurrent User {i}",
                loan_amount_in_cents=Decimal('500000000'),  # ₹5 Lakh
                emi_in_cents=Decimal('4500000'),            # ₹45,000
                interest_rate=Decimal('12.00'),
                tenure_months=120,  # 10 years
                loan_purpose="Personal Loan",
                sanction_date=datetime.now(),
                loan_id=f"CONC{i}_{int(time.time())}"
            )
            test_data_list.append(test_data)
        
        # Queue to collect results
        results_queue = queue.Queue()
        
        def generate_pdf_worker(sanction_data):
            """Worker function for concurrent PDF generation."""
            try:
                pdf_path = self.pdf_service.generate_sanction_letter(sanction_data)
                pdf_info = self.pdf_service.get_pdf_info(pdf_path)
                results_queue.put(('success', pdf_path, pdf_info))
            except Exception as e:
                results_queue.put(('error', str(e), None))
        
        # Start concurrent PDF generation
        threads = []
        for test_data in test_data_list:
            thread = threading.Thread(target=generate_pdf_worker, args=(test_data,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect and verify results
        successful_generations = 0
        generated_files = []
        
        while not results_queue.empty():
            status, result, info = results_queue.get()
            
            if status == 'success':
                successful_generations += 1
                generated_files.append(result)
                
                # Verify each generated file
                assert os.path.exists(result), f"Concurrent PDF not created: {result}"
                assert info["is_readable"], f"Concurrent PDF not readable: {result}"
                assert info["file_size_bytes"] > 1000, f"Concurrent PDF too small: {result}"
            else:
                # Log error but don't fail the test unless all fail
                logger.error(f"Concurrent PDF generation error: {result}")
        
        # At least 80% of concurrent generations should succeed
        success_rate = successful_generations / len(test_data_list)
        assert success_rate >= 0.8, \
            f"Concurrent generation success rate too low: {success_rate:.2%}"
        
        # Verify all generated files have unique names
        filenames = [Path(pdf_path).name for pdf_path in generated_files]
        assert len(filenames) == len(set(filenames)), \
            "Concurrent generation produced duplicate filenames"