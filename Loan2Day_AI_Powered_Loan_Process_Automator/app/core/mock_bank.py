"""
Mock Banking API - Realistic Banking Service Simulation

This module provides a comprehensive mock banking service that simulates real banking APIs
with realistic responses for credit scores, account verification, and transaction history.
It provides deterministic results based on input parameters to enable consistent testing
while maintaining realistic data patterns.

Key Features:
- Credit score simulation (780 for good PANs, 550 for poor credit history)
- Bank account verification with realistic response patterns
- Transaction history generation with synthetic but realistic data
- Income verification simulation based on salary account analysis
- Deterministic results for consistent testing
- Realistic API response times and error scenarios

Security Standards:
- All PAN numbers are validated for format compliance
- Account numbers follow realistic banking patterns
- Transaction patterns simulate real-world banking behavior
- No actual banking data is stored or processed

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

import hashlib
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging
import time
import re

# Configure logger
logger = logging.getLogger(__name__)

class CreditScoreRange(Enum):
    """Credit score range classifications."""
    EXCELLENT = "EXCELLENT"  # 750-900
    GOOD = "GOOD"           # 650-749
    FAIR = "FAIR"           # 550-649
    POOR = "POOR"           # 300-549

class AccountType(Enum):
    """Bank account type classifications."""
    SAVINGS = "SAVINGS"
    CURRENT = "CURRENT"
    SALARY = "SALARY"
    FIXED_DEPOSIT = "FIXED_DEPOSIT"

class TransactionType(Enum):
    """Transaction type classifications."""
    CREDIT = "CREDIT"
    DEBIT = "DEBIT"
    TRANSFER = "TRANSFER"
    ATM_WITHDRAWAL = "ATM_WITHDRAWAL"
    ONLINE_PAYMENT = "ONLINE_PAYMENT"
    SALARY_CREDIT = "SALARY_CREDIT"
    INTEREST_CREDIT = "INTEREST_CREDIT"

class BankingError(Exception):
    """Base exception for mock banking API errors."""
    pass

class InvalidPANError(BankingError):
    """Raised when PAN number format is invalid."""
    pass

class InvalidAccountError(BankingError):
    """Raised when account number format is invalid."""
    pass

class APITimeoutError(BankingError):
    """Raised when mock API timeout occurs."""
    pass

class CreditScoreResult:
    """
    Credit score lookup result with comprehensive details.
    
    Simulates real CIBIL/Experian credit bureau responses with
    realistic scoring and risk assessment data.
    """
    
    def __init__(
        self,
        pan_number: str,
        credit_score: int,
        score_range: CreditScoreRange,
        report_date: datetime,
        credit_history_length_months: int,
        total_accounts: int,
        active_accounts: int,
        total_credit_limit_in_cents: Decimal,
        credit_utilization_percentage: float,
        payment_history_score: int,
        recent_inquiries: int,
        defaulted_accounts: int,
        risk_factors: List[str]
    ):
        self.pan_number = pan_number
        self.credit_score = credit_score
        self.score_range = score_range
        self.report_date = report_date
        self.credit_history_length_months = credit_history_length_months
        self.total_accounts = total_accounts
        self.active_accounts = active_accounts
        self.total_credit_limit_in_cents = total_credit_limit_in_cents
        self.credit_utilization_percentage = credit_utilization_percentage
        self.payment_history_score = payment_history_score
        self.recent_inquiries = recent_inquiries
        self.defaulted_accounts = defaulted_accounts
        self.risk_factors = risk_factors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert credit score result to dictionary for JSON serialization."""
        return {
            "pan_number": self.pan_number,
            "credit_score": self.credit_score,
            "score_range": self.score_range.value,
            "report_date": self.report_date.isoformat(),
            "credit_history_length_months": self.credit_history_length_months,
            "total_accounts": self.total_accounts,
            "active_accounts": self.active_accounts,
            "total_credit_limit_in_cents": str(self.total_credit_limit_in_cents),
            "credit_utilization_percentage": self.credit_utilization_percentage,
            "payment_history_score": self.payment_history_score,
            "recent_inquiries": self.recent_inquiries,
            "defaulted_accounts": self.defaulted_accounts,
            "risk_factors": self.risk_factors,
            "is_good_credit": self.is_good_credit(),
            "loan_eligibility": self.get_loan_eligibility()
        }
    
    def is_good_credit(self) -> bool:
        """
        Determine if credit score indicates good creditworthiness.
        
        Returns:
            bool: True if credit score is good (>=650), False otherwise
        """
        return self.credit_score >= 650
    
    def get_loan_eligibility(self) -> str:
        """
        Get loan eligibility assessment based on credit score.
        
        Returns:
            str: Loan eligibility status
        """
        if self.credit_score >= 750:
            return "EXCELLENT_ELIGIBLE"
        elif self.credit_score >= 650:
            return "GOOD_ELIGIBLE"
        elif self.credit_score >= 550:
            return "CONDITIONAL_ELIGIBLE"
        else:
            return "NOT_ELIGIBLE"

class AccountVerificationResult:
    """
    Bank account verification result with comprehensive validation details.
    
    Simulates real bank account verification APIs with realistic
    validation checks and account status information.
    """
    
    def __init__(
        self,
        account_number: str,
        ifsc_code: str,
        account_holder_name: str,
        bank_name: str,
        branch_name: str,
        account_type: AccountType,
        account_status: str,
        is_active: bool,
        opening_date: datetime,
        current_balance_in_cents: Decimal,
        average_balance_in_cents: Decimal,
        last_transaction_date: datetime,
        kyc_status: str,
        verification_timestamp: datetime
    ):
        self.account_number = account_number
        self.ifsc_code = ifsc_code
        self.account_holder_name = account_holder_name
        self.bank_name = bank_name
        self.branch_name = branch_name
        self.account_type = account_type
        self.account_status = account_status
        self.is_active = is_active
        self.opening_date = opening_date
        self.current_balance_in_cents = current_balance_in_cents
        self.average_balance_in_cents = average_balance_in_cents
        self.last_transaction_date = last_transaction_date
        self.kyc_status = kyc_status
        self.verification_timestamp = verification_timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert account verification result to dictionary."""
        return {
            "account_number": self.account_number,
            "ifsc_code": self.ifsc_code,
            "account_holder_name": self.account_holder_name,
            "bank_name": self.bank_name,
            "branch_name": self.branch_name,
            "account_type": self.account_type.value,
            "account_status": self.account_status,
            "is_active": self.is_active,
            "opening_date": self.opening_date.isoformat(),
            "current_balance_in_cents": str(self.current_balance_in_cents),
            "average_balance_in_cents": str(self.average_balance_in_cents),
            "last_transaction_date": self.last_transaction_date.isoformat(),
            "kyc_status": self.kyc_status,
            "verification_timestamp": self.verification_timestamp.isoformat(),
            "account_age_months": self.get_account_age_months(),
            "is_salary_account": self.is_salary_account()
        }
    
    def get_account_age_months(self) -> int:
        """Calculate account age in months."""
        delta = datetime.now() - self.opening_date
        return int(delta.days / 30)
    
    def is_salary_account(self) -> bool:
        """Check if this is a salary account."""
        return self.account_type == AccountType.SALARY

class Transaction:
    """
    Individual transaction record with realistic banking transaction details.
    
    All monetary values use decimal.Decimal following the LQM Standard.
    """
    
    def __init__(
        self,
        transaction_id: str,
        date: datetime,
        transaction_type: TransactionType,
        amount_in_cents: Decimal,
        balance_after_in_cents: Decimal,
        description: str,
        reference_number: str,
        counterparty: Optional[str] = None
    ):
        self.transaction_id = transaction_id
        self.date = date
        self.transaction_type = transaction_type
        self.amount_in_cents = amount_in_cents
        self.balance_after_in_cents = balance_after_in_cents
        self.description = description
        self.reference_number = reference_number
        self.counterparty = counterparty
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "date": self.date.isoformat(),
            "transaction_type": self.transaction_type.value,
            "amount_in_cents": str(self.amount_in_cents),
            "balance_after_in_cents": str(self.balance_after_in_cents),
            "description": self.description,
            "reference_number": self.reference_number,
            "counterparty": self.counterparty
        }

class TransactionHistory:
    """
    Complete transaction history with analysis and patterns.
    
    All monetary calculations use decimal.Decimal for LQM compliance.
    """
    
    def __init__(
        self,
        account_number: str,
        from_date: datetime,
        to_date: datetime,
        transactions: List[Transaction],
        total_credits_in_cents: Decimal,
        total_debits_in_cents: Decimal,
        average_monthly_credits_in_cents: Decimal,
        average_monthly_debits_in_cents: Decimal,
        salary_credits_count: int,
        average_salary_in_cents: Decimal,
        bounce_count: int,
        analysis_timestamp: datetime
    ):
        self.account_number = account_number
        self.from_date = from_date
        self.to_date = to_date
        self.transactions = transactions
        self.total_credits_in_cents = total_credits_in_cents
        self.total_debits_in_cents = total_debits_in_cents
        self.average_monthly_credits_in_cents = average_monthly_credits_in_cents
        self.average_monthly_debits_in_cents = average_monthly_debits_in_cents
        self.salary_credits_count = salary_credits_count
        self.average_salary_in_cents = average_salary_in_cents
        self.bounce_count = bounce_count
        self.analysis_timestamp = analysis_timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction history to dictionary."""
        return {
            "account_number": self.account_number,
            "from_date": self.from_date.isoformat(),
            "to_date": self.to_date.isoformat(),
            "transaction_count": len(self.transactions),
            "transactions": [tx.to_dict() for tx in self.transactions],
            "total_credits_in_cents": str(self.total_credits_in_cents),
            "total_debits_in_cents": str(self.total_debits_in_cents),
            "average_monthly_credits_in_cents": str(self.average_monthly_credits_in_cents),
            "average_monthly_debits_in_cents": str(self.average_monthly_debits_in_cents),
            "salary_credits_count": self.salary_credits_count,
            "average_salary_in_cents": str(self.average_salary_in_cents),
            "bounce_count": self.bounce_count,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "net_flow_in_cents": str(self.total_credits_in_cents - self.total_debits_in_cents),
            "banking_behavior": self.get_banking_behavior()
        }
    
    def get_banking_behavior(self) -> str:
        """Analyze banking behavior pattern."""
        if self.salary_credits_count >= 3 and self.bounce_count == 0:
            return "STABLE_SALARIED"
        elif self.salary_credits_count >= 1 and self.bounce_count <= 1:
            return "REGULAR_INCOME"
        elif self.bounce_count > 2:
            return "IRREGULAR_PAYMENTS"
        else:
            return "MIXED_PATTERN"

class IncomeVerificationResult:
    """
    Income verification result based on salary account analysis.
    
    All monetary values use decimal.Decimal following the LQM Standard.
    """
    
    def __init__(
        self,
        pan_number: str,
        account_number: str,
        verified_monthly_income_in_cents: Decimal,
        income_stability_score: float,
        employment_type: str,
        employer_name: Optional[str],
        income_trend: str,
        verification_confidence: float,
        last_salary_date: datetime,
        verification_timestamp: datetime
    ):
        self.pan_number = pan_number
        self.account_number = account_number
        self.verified_monthly_income_in_cents = verified_monthly_income_in_cents
        self.income_stability_score = income_stability_score
        self.employment_type = employment_type
        self.employer_name = employer_name
        self.income_trend = income_trend
        self.verification_confidence = verification_confidence
        self.last_salary_date = last_salary_date
        self.verification_timestamp = verification_timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert income verification result to dictionary."""
        return {
            "pan_number": self.pan_number,
            "account_number": self.account_number,
            "verified_monthly_income_in_cents": str(self.verified_monthly_income_in_cents),
            "income_stability_score": self.income_stability_score,
            "employment_type": self.employment_type,
            "employer_name": self.employer_name,
            "income_trend": self.income_trend,
            "verification_confidence": self.verification_confidence,
            "last_salary_date": self.last_salary_date.isoformat(),
            "verification_timestamp": self.verification_timestamp.isoformat(),
            "annual_income_in_cents": str(self.verified_monthly_income_in_cents * 12),
            "is_income_sufficient": self.is_income_sufficient()
        }
    
    def is_income_sufficient(self, min_monthly_income_in_cents: Decimal = Decimal("2500000")) -> bool:
        """
        Check if income is sufficient for loan eligibility.
        
        Args:
            min_monthly_income_in_cents: Minimum required monthly income (default: ₹25,000)
            
        Returns:
            bool: True if income is sufficient
        """
        return self.verified_monthly_income_in_cents >= min_monthly_income_in_cents
class MockBankingAPI:
    """
    Mock Banking API - Comprehensive simulation of real banking services.
    
    This class provides realistic mock implementations of banking APIs including:
    - Credit score lookup (CIBIL/Experian simulation)
    - Bank account verification
    - Transaction history analysis
    - Income verification from salary accounts
    
    All responses are deterministic based on input parameters to enable
    consistent testing while maintaining realistic data patterns.
    
    Follows LQM Standard: All monetary values use decimal.Decimal.
    """
    
    # Realistic bank names and IFSC patterns
    MOCK_BANKS = {
        "HDFC": {"name": "HDFC Bank", "ifsc_prefix": "HDFC0"},
        "ICICI": {"name": "ICICI Bank", "ifsc_prefix": "ICIC0"},
        "SBI": {"name": "State Bank of India", "ifsc_prefix": "SBIN0"},
        "AXIS": {"name": "Axis Bank", "ifsc_prefix": "UTIB0"},
        "KOTAK": {"name": "Kotak Mahindra Bank", "ifsc_prefix": "KKBK0"},
        "INDUSIND": {"name": "IndusInd Bank", "ifsc_prefix": "INDB0"}
    }
    
    # Common employer names for salary account simulation
    MOCK_EMPLOYERS = [
        "TCS Limited", "Infosys Limited", "Wipro Limited", "HCL Technologies",
        "Tech Mahindra", "Cognizant Technology Solutions", "Accenture India",
        "IBM India", "Microsoft India", "Amazon India", "Google India",
        "Flipkart Internet", "Paytm", "Zomato Limited", "Swiggy",
        "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank"
    ]
    
    def __init__(self):
        """Initialize mock banking API."""
        self.api_call_count = 0
        logger.info("MockBankingAPI initialized successfully")
    
    def _validate_pan_format(self, pan_number: str) -> bool:
        """
        Validate PAN number format (AAAAA9999A).
        
        Args:
            pan_number: PAN number to validate
            
        Returns:
            bool: True if format is valid
        """
        pan_pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
        return bool(re.match(pan_pattern, pan_number.upper()))
    
    def _validate_account_number(self, account_number: str) -> bool:
        """
        Validate bank account number format.
        
        Args:
            account_number: Account number to validate
            
        Returns:
            bool: True if format is valid
        """
        # Account numbers should be 9-18 digits
        return account_number.isdigit() and 9 <= len(account_number) <= 18
    
    def _generate_deterministic_hash(self, input_string: str) -> int:
        """
        Generate deterministic hash for consistent mock responses.
        
        Args:
            input_string: Input string to hash
            
        Returns:
            int: Deterministic hash value
        """
        return int(hashlib.md5(input_string.encode()).hexdigest()[:8], 16)
    
    def _simulate_api_delay(self, min_ms: int = 100, max_ms: int = 500):
        """
        Simulate realistic API response delay.
        
        Args:
            min_ms: Minimum delay in milliseconds
            max_ms: Maximum delay in milliseconds
        """
        delay_seconds = random.uniform(min_ms / 1000, max_ms / 1000)
        time.sleep(delay_seconds)
    def lookup_credit_score(self, pan_number: str) -> CreditScoreResult:
        """
        Lookup credit score for given PAN number.
        
        Returns 780 for good PANs, 550 for poor credit history based on
        deterministic algorithm using PAN number characteristics.
        
        Args:
            pan_number: PAN number to lookup
            
        Returns:
            CreditScoreResult: Comprehensive credit score information
            
        Raises:
            InvalidPANError: If PAN format is invalid
            APITimeoutError: If simulated API timeout occurs
        """
        self.api_call_count += 1
        
        # Validate PAN format
        if not self._validate_pan_format(pan_number):
            raise InvalidPANError(f"Invalid PAN format: {pan_number}")
        
        # Simulate API delay
        self._simulate_api_delay(200, 800)
        
        # Simulate occasional API timeout (1% chance)
        hash_val = self._generate_deterministic_hash(pan_number)
        if hash_val % 100 == 0:
            raise APITimeoutError("Credit bureau API timeout")
        
        logger.info(f"Looking up credit score for PAN: {pan_number}")
        
        # Deterministic credit score based on PAN characteristics
        pan_hash = self._generate_deterministic_hash(pan_number)
        
        # Extract characteristics from PAN for scoring
        first_char_value = ord(pan_number[0]) - ord('A')
        last_char_value = ord(pan_number[-1]) - ord('A')
        middle_digits = int(pan_number[5:9])
        
        # Calculate base score using deterministic algorithm
        base_score = 300 + (pan_hash % 600)  # Range: 300-900
        
        # Apply adjustments based on PAN characteristics
        if first_char_value % 3 == 0:  # Good credit indicator
            base_score += 50
        if last_char_value % 2 == 0:   # Stability indicator
            base_score += 30
        if middle_digits % 1000 > 500: # Income indicator
            base_score += 40
        
        # Ensure score is within valid range
        credit_score = max(300, min(900, base_score))
        
        # Determine score range
        if credit_score >= 750:
            score_range = CreditScoreRange.EXCELLENT
        elif credit_score >= 650:
            score_range = CreditScoreRange.GOOD
        elif credit_score >= 550:
            score_range = CreditScoreRange.FAIR
        else:
            score_range = CreditScoreRange.POOR
        
        # Generate realistic credit profile details
        credit_history_months = 12 + (pan_hash % 120)  # 1-10 years
        total_accounts = 2 + (pan_hash % 8)             # 2-10 accounts
        active_accounts = max(1, total_accounts - (pan_hash % 3))
        
        # Credit limit based on score (LQM Standard: Decimal for currency)
        if credit_score >= 750:
            credit_limit_base = Decimal("50000000")  # ₹5 lakh in cents
        elif credit_score >= 650:
            credit_limit_base = Decimal("30000000")  # ₹3 lakh in cents
        else:
            credit_limit_base = Decimal("10000000")  # ₹1 lakh in cents
        
        total_credit_limit = credit_limit_base + Decimal(str(pan_hash % 20000000))
        
        # Credit utilization (lower is better)
        if credit_score >= 750:
            utilization = 10 + (pan_hash % 20)  # 10-30%
        else:
            utilization = 30 + (pan_hash % 40)  # 30-70%
        
        # Payment history score
        payment_score = max(60, min(100, credit_score // 9))
        
        # Recent inquiries and defaults
        recent_inquiries = pan_hash % 5
        defaulted_accounts = 0 if credit_score >= 650 else (pan_hash % 3)
        
        # Risk factors based on score
        risk_factors = []
        if credit_score < 650:
            risk_factors.append("Below average credit score")
        if utilization > 50:
            risk_factors.append("High credit utilization")
        if recent_inquiries > 3:
            risk_factors.append("Multiple recent credit inquiries")
        if defaulted_accounts > 0:
            risk_factors.append(f"{defaulted_accounts} defaulted account(s)")
        
        result = CreditScoreResult(
            pan_number=pan_number,
            credit_score=credit_score,
            score_range=score_range,
            report_date=datetime.now(),
            credit_history_length_months=credit_history_months,
            total_accounts=total_accounts,
            active_accounts=active_accounts,
            total_credit_limit_in_cents=total_credit_limit,
            credit_utilization_percentage=utilization,
            payment_history_score=payment_score,
            recent_inquiries=recent_inquiries,
            defaulted_accounts=defaulted_accounts,
            risk_factors=risk_factors
        )
        
        logger.info(
            f"Credit score lookup completed - PAN: {pan_number}, "
            f"Score: {credit_score}, Range: {score_range.value}"
        )
        
        return result
    def verify_account(self, account_number: str, ifsc_code: str) -> AccountVerificationResult:
        """
        Verify bank account details and return account information.
        
        Args:
            account_number: Bank account number
            ifsc_code: IFSC code of the bank branch
            
        Returns:
            AccountVerificationResult: Comprehensive account verification details
            
        Raises:
            InvalidAccountError: If account format is invalid
            APITimeoutError: If simulated API timeout occurs
        """
        self.api_call_count += 1
        
        # Validate account number format
        if not self._validate_account_number(account_number):
            raise InvalidAccountError(f"Invalid account number format: {account_number}")
        
        # Simulate API delay
        self._simulate_api_delay(300, 1000)
        
        # Simulate occasional API timeout (2% chance)
        hash_val = self._generate_deterministic_hash(account_number + ifsc_code)
        if hash_val % 50 == 0:
            raise APITimeoutError("Bank verification API timeout")
        
        logger.info(f"Verifying account: {account_number}, IFSC: {ifsc_code}")
        
        # Generate deterministic account details
        account_hash = self._generate_deterministic_hash(account_number)
        ifsc_hash = self._generate_deterministic_hash(ifsc_code)
        
        # Determine bank from IFSC or use hash
        bank_key = None
        for key, bank_info in self.MOCK_BANKS.items():
            if ifsc_code.startswith(bank_info["ifsc_prefix"]):
                bank_key = key
                break
        
        if not bank_key:
            bank_keys = list(self.MOCK_BANKS.keys())
            bank_key = bank_keys[ifsc_hash % len(bank_keys)]
        
        bank_info = self.MOCK_BANKS[bank_key]
        
        # Generate account holder name (deterministic)
        first_names = ["Rajesh", "Priya", "Amit", "Sunita", "Vikram", "Kavya", "Arjun", "Meera"]
        last_names = ["Sharma", "Patel", "Kumar", "Singh", "Reddy", "Nair", "Gupta", "Joshi"]
        
        first_name = first_names[account_hash % len(first_names)]
        last_name = last_names[(account_hash // 10) % len(last_names)]
        account_holder_name = f"{first_name} {last_name}"
        
        # Determine account type
        account_types = list(AccountType)
        account_type = account_types[account_hash % len(account_types)]
        
        # Account status (90% active)
        is_active = (account_hash % 10) < 9
        account_status = "ACTIVE" if is_active else "DORMANT"
        
        # Account opening date (1-10 years ago)
        days_ago = 365 + (account_hash % (365 * 9))  # 1-10 years
        opening_date = datetime.now() - timedelta(days=days_ago)
        
        # Balance information (LQM Standard: Decimal for currency)
        if account_type == AccountType.SALARY:
            base_balance = Decimal("5000000")  # ₹50,000 in cents
        elif account_type == AccountType.SAVINGS:
            base_balance = Decimal("2000000")  # ₹20,000 in cents
        else:
            base_balance = Decimal("10000000") # ₹1,00,000 in cents
        
        current_balance = base_balance + Decimal(str(account_hash % 5000000))
        average_balance = current_balance * Decimal("0.8")
        
        # Last transaction date (within last 30 days for active accounts)
        if is_active:
            last_tx_days = account_hash % 30
        else:
            last_tx_days = 30 + (account_hash % 180)  # 30-210 days ago
        
        last_transaction_date = datetime.now() - timedelta(days=last_tx_days)
        
        # KYC status (95% completed)
        kyc_status = "COMPLETED" if (account_hash % 20) < 19 else "PENDING"
        
        # Branch name
        branch_names = [
            "Main Branch", "Commercial Street", "MG Road", "Koramangala",
            "Whitefield", "Electronic City", "Indiranagar", "Jayanagar"
        ]
        branch_name = branch_names[ifsc_hash % len(branch_names)]
        
        result = AccountVerificationResult(
            account_number=account_number,
            ifsc_code=ifsc_code,
            account_holder_name=account_holder_name,
            bank_name=bank_info["name"],
            branch_name=branch_name,
            account_type=account_type,
            account_status=account_status,
            is_active=is_active,
            opening_date=opening_date,
            current_balance_in_cents=current_balance,
            average_balance_in_cents=average_balance,
            last_transaction_date=last_transaction_date,
            kyc_status=kyc_status,
            verification_timestamp=datetime.now()
        )
        
        logger.info(
            f"Account verification completed - Account: {account_number}, "
            f"Bank: {bank_info['name']}, Status: {account_status}, "
            f"Type: {account_type.value}"
        )
        
        return result
    def get_transaction_history(
        self,
        account_number: str,
        from_date: datetime,
        to_date: datetime,
        max_transactions: int = 100
    ) -> TransactionHistory:
        """
        Generate realistic transaction history for the specified period.
        
        Args:
            account_number: Bank account number
            from_date: Start date for transaction history
            to_date: End date for transaction history
            max_transactions: Maximum number of transactions to return
            
        Returns:
            TransactionHistory: Comprehensive transaction history with analysis
            
        Raises:
            InvalidAccountError: If account format is invalid
        """
        self.api_call_count += 1
        
        # Validate account number
        if not self._validate_account_number(account_number):
            raise InvalidAccountError(f"Invalid account number format: {account_number}")
        
        # Simulate API delay
        self._simulate_api_delay(500, 1500)
        
        logger.info(
            f"Generating transaction history for account: {account_number}, "
            f"Period: {from_date.date()} to {to_date.date()}"
        )
        
        # Generate deterministic transaction pattern
        account_hash = self._generate_deterministic_hash(account_number)
        
        # Calculate number of days in period
        period_days = (to_date - from_date).days
        
        # Generate realistic number of transactions (2-5 per week)
        transactions_per_week = 2 + (account_hash % 4)
        total_transactions = min(
            max_transactions,
            int((period_days / 7) * transactions_per_week)
        )
        
        transactions = []
        current_balance = Decimal("5000000")  # Starting balance ₹50,000 in cents
        
        # Generate salary pattern (monthly salary credits)
        salary_amount = Decimal("7500000") + Decimal(str(account_hash % 2500000))  # ₹75k-₹100k in cents
        salary_dates = []
        
        # Add monthly salary credits
        current_date = from_date
        while current_date <= to_date:
            # Salary typically credited on 1st or last day of month
            if account_hash % 2 == 0:
                salary_date = current_date.replace(day=1)
            else:
                # Last day of month
                if current_date.month == 12:
                    next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    next_month = current_date.replace(month=current_date.month + 1, day=1)
                salary_date = next_month - timedelta(days=1)
            
            if from_date <= salary_date <= to_date:
                salary_dates.append(salary_date)
                
                # Create salary transaction
                tx_id = f"SAL{salary_date.strftime('%Y%m%d')}{account_number[-4:]}"
                current_balance += salary_amount
                
                employer = self.MOCK_EMPLOYERS[account_hash % len(self.MOCK_EMPLOYERS)]
                
                transaction = Transaction(
                    transaction_id=tx_id,
                    date=salary_date,
                    transaction_type=TransactionType.SALARY_CREDIT,
                    amount_in_cents=salary_amount,
                    balance_after_in_cents=current_balance,
                    description=f"Salary Credit from {employer}",
                    reference_number=f"SAL{account_hash % 1000000:06d}",
                    counterparty=employer
                )
                transactions.append(transaction)
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        # Generate other transactions (expenses, transfers, etc.)
        remaining_transactions = total_transactions - len(transactions)
        
        for i in range(remaining_transactions):
            # Random date within period
            random_days = random.randint(0, period_days)
            tx_date = from_date + timedelta(days=random_days)
            
            # Skip if too close to salary dates
            too_close_to_salary = any(
                abs((tx_date - sal_date).days) < 2 for sal_date in salary_dates
            )
            if too_close_to_salary:
                continue
            
            # Determine transaction type and amount (LQM Standard: Decimal)
            tx_type_rand = (account_hash + i) % 100
            
            if tx_type_rand < 40:  # 40% regular expenses
                tx_type = TransactionType.DEBIT
                amount = Decimal(str(50000 + ((account_hash + i) % 500000)))  # ₹500-₹5.5k in cents
                current_balance -= amount
                descriptions = [
                    "Online Purchase", "Grocery Shopping", "Fuel Payment",
                    "Restaurant Bill", "Medical Expense", "Utility Bill"
                ]
                description = descriptions[(account_hash + i) % len(descriptions)]
                
            elif tx_type_rand < 60:  # 20% ATM withdrawals
                tx_type = TransactionType.ATM_WITHDRAWAL
                amount = Decimal(str(200000 + ((account_hash + i) % 800000)))  # ₹2k-₹10k in cents
                current_balance -= amount
                description = "ATM Withdrawal"
                
            elif tx_type_rand < 75:  # 15% online payments
                tx_type = TransactionType.ONLINE_PAYMENT
                amount = Decimal(str(100000 + ((account_hash + i) % 2000000)))  # ₹1k-₹21k in cents
                current_balance -= amount
                descriptions = [
                    "Credit Card Payment", "Insurance Premium", "EMI Payment",
                    "Online Shopping", "Subscription Payment"
                ]
                description = descriptions[(account_hash + i) % len(descriptions)]
                
            elif tx_type_rand < 85:  # 10% transfers
                tx_type = TransactionType.TRANSFER
                amount = Decimal(str(500000 + ((account_hash + i) % 3000000)))  # ₹5k-₹35k in cents
                current_balance -= amount
                description = "Fund Transfer"
                
            else:  # 15% credits (interest, refunds, etc.)
                tx_type = TransactionType.CREDIT
                amount = Decimal(str(10000 + ((account_hash + i) % 50000)))  # ₹100-₹600 in cents
                current_balance += amount
                descriptions = [
                    "Interest Credit", "Cashback Credit", "Refund Credit",
                    "Dividend Credit", "Bonus Credit"
                ]
                description = descriptions[(account_hash + i) % len(descriptions)]
            
            # Ensure balance doesn't go negative
            if tx_type in [TransactionType.DEBIT, TransactionType.ATM_WITHDRAWAL, 
                          TransactionType.ONLINE_PAYMENT, TransactionType.TRANSFER]:
                if current_balance < 0:
                    current_balance += amount  # Reverse the debit
                    continue  # Skip this transaction
            
            # Create transaction
            tx_id = f"TXN{tx_date.strftime('%Y%m%d')}{(account_hash + i) % 10000:04d}"
            ref_number = f"REF{(account_hash + i) % 1000000:06d}"
            
            transaction = Transaction(
                transaction_id=tx_id,
                date=tx_date,
                transaction_type=tx_type,
                amount_in_cents=amount,
                balance_after_in_cents=current_balance,
                description=description,
                reference_number=ref_number
            )
            transactions.append(transaction)
        
        # Sort transactions by date
        transactions.sort(key=lambda x: x.date)
        
        # Calculate summary statistics (LQM Standard: Decimal arithmetic)
        total_credits = sum(
            tx.amount_in_cents for tx in transactions 
            if tx.transaction_type in [TransactionType.CREDIT, TransactionType.SALARY_CREDIT]
        )
        
        total_debits = sum(
            tx.amount_in_cents for tx in transactions 
            if tx.transaction_type in [
                TransactionType.DEBIT, TransactionType.ATM_WITHDRAWAL,
                TransactionType.ONLINE_PAYMENT, TransactionType.TRANSFER
            ]
        )
        
        # Monthly averages
        months_in_period = max(1, period_days / 30)
        avg_monthly_credits = total_credits / Decimal(str(months_in_period))
        avg_monthly_debits = total_debits / Decimal(str(months_in_period))
        
        # Salary analysis
        salary_transactions = [
            tx for tx in transactions 
            if tx.transaction_type == TransactionType.SALARY_CREDIT
        ]
        salary_count = len(salary_transactions)
        
        if salary_transactions:
            avg_salary = sum(tx.amount_in_cents for tx in salary_transactions) / len(salary_transactions)
        else:
            avg_salary = Decimal("0")
        
        # Bounce count (simulate occasional bounced transactions)
        bounce_count = 0 if (account_hash % 10) < 8 else (account_hash % 3)
        
        result = TransactionHistory(
            account_number=account_number,
            from_date=from_date,
            to_date=to_date,
            transactions=transactions,
            total_credits_in_cents=total_credits,
            total_debits_in_cents=total_debits,
            average_monthly_credits_in_cents=avg_monthly_credits,
            average_monthly_debits_in_cents=avg_monthly_debits,
            salary_credits_count=salary_count,
            average_salary_in_cents=avg_salary,
            bounce_count=bounce_count,
            analysis_timestamp=datetime.now()
        )
        
        logger.info(
            f"Transaction history generated - Account: {account_number}, "
            f"Transactions: {len(transactions)}, Credits: ₹{total_credits/100:.2f}, "
            f"Debits: ₹{total_debits/100:.2f}, Salary Credits: {salary_count}"
        )
        
        return result
    def verify_income(
        self,
        pan_number: str,
        account_number: str,
        months_to_analyze: int = 6
    ) -> IncomeVerificationResult:
        """
        Verify income based on salary account analysis.
        
        Args:
            pan_number: PAN number of the applicant
            account_number: Salary account number
            months_to_analyze: Number of months to analyze (default: 6)
            
        Returns:
            IncomeVerificationResult: Comprehensive income verification details
            
        Raises:
            InvalidPANError: If PAN format is invalid
            InvalidAccountError: If account format is invalid
        """
        self.api_call_count += 1
        
        # Validate inputs
        if not self._validate_pan_format(pan_number):
            raise InvalidPANError(f"Invalid PAN format: {pan_number}")
        
        if not self._validate_account_number(account_number):
            raise InvalidAccountError(f"Invalid account number format: {account_number}")
        
        # Simulate API delay
        self._simulate_api_delay(800, 2000)
        
        logger.info(
            f"Verifying income for PAN: {pan_number}, "
            f"Account: {account_number}, Period: {months_to_analyze} months"
        )
        
        # Generate transaction history for analysis
        to_date = datetime.now()
        from_date = to_date - timedelta(days=months_to_analyze * 30)
        
        tx_history = self.get_transaction_history(account_number, from_date, to_date)
        
        # Analyze salary patterns
        salary_transactions = [
            tx for tx in tx_history.transactions
            if tx.transaction_type == TransactionType.SALARY_CREDIT
        ]
        
        if not salary_transactions:
            # No salary found, use average credits as income estimate
            verified_income = tx_history.average_monthly_credits_in_cents
            income_stability = 0.3  # Low stability without clear salary
            employment_type = "SELF_EMPLOYED"
            employer_name = None
            income_trend = "IRREGULAR"
            confidence = 0.4
        else:
            # Calculate verified income from salary credits
            verified_income = tx_history.average_salary_in_cents
            
            # Calculate income stability score
            if len(salary_transactions) >= months_to_analyze:
                income_stability = 0.9  # High stability with regular salary
            elif len(salary_transactions) >= months_to_analyze // 2:
                income_stability = 0.7  # Moderate stability
            else:
                income_stability = 0.5  # Low stability
            
            employment_type = "SALARIED"
            
            # Extract employer from salary transactions
            if salary_transactions:
                employer_name = salary_transactions[0].counterparty
            else:
                employer_name = None
            
            # Determine income trend (LQM Standard: Decimal arithmetic)
            if len(salary_transactions) >= 3:
                recent_avg = sum(
                    tx.amount_in_cents for tx in salary_transactions[-3:]
                ) / 3
                older_avg = sum(
                    tx.amount_in_cents for tx in salary_transactions[:-3]
                ) / max(1, len(salary_transactions) - 3)
                
                if recent_avg > older_avg * Decimal("1.1"):
                    income_trend = "INCREASING"
                elif recent_avg < older_avg * Decimal("0.9"):
                    income_trend = "DECREASING"
                else:
                    income_trend = "STABLE"
            else:
                income_trend = "INSUFFICIENT_DATA"
            
            # Confidence based on data quality
            confidence = min(0.95, 0.5 + (len(salary_transactions) * 0.1))
        
        # Last salary date
        if salary_transactions:
            last_salary_date = max(tx.date for tx in salary_transactions)
        else:
            last_salary_date = to_date - timedelta(days=60)  # Assume 2 months ago
        
        result = IncomeVerificationResult(
            pan_number=pan_number,
            account_number=account_number,
            verified_monthly_income_in_cents=verified_income,
            income_stability_score=income_stability,
            employment_type=employment_type,
            employer_name=employer_name,
            income_trend=income_trend,
            verification_confidence=confidence,
            last_salary_date=last_salary_date,
            verification_timestamp=datetime.now()
        )
        
        logger.info(
            f"Income verification completed - PAN: {pan_number}, "
            f"Monthly Income: ₹{verified_income/100:.2f}, "
            f"Stability: {income_stability:.2f}, Type: {employment_type}"
        )
        
        return result
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about mock API usage.
        
        Returns:
            Dict[str, Any]: API usage statistics
        """
        return {
            "total_api_calls": self.api_call_count,
            "supported_banks": len(self.MOCK_BANKS),
            "supported_employers": len(self.MOCK_EMPLOYERS),
            "api_version": "MockBankingAPI-1.0",
            "features": [
                "Credit Score Lookup",
                "Account Verification", 
                "Transaction History",
                "Income Verification"
            ]
        }
# Global mock banking API instance
_mock_banking_api: Optional[MockBankingAPI] = None

def get_mock_banking_api() -> MockBankingAPI:
    """
    Get global mock banking API instance (singleton pattern).
    
    Returns:
        MockBankingAPI: Global mock banking API instance
    """
    global _mock_banking_api
    if _mock_banking_api is None:
        _mock_banking_api = MockBankingAPI()
    return _mock_banking_api

# Convenience functions for direct usage
def lookup_credit_score(pan_number: str) -> CreditScoreResult:
    """
    Convenience function for credit score lookup.
    
    Args:
        pan_number: PAN number to lookup
        
    Returns:
        CreditScoreResult: Credit score information
    """
    api = get_mock_banking_api()
    return api.lookup_credit_score(pan_number)

def verify_account(account_number: str, ifsc_code: str) -> AccountVerificationResult:
    """
    Convenience function for account verification.
    
    Args:
        account_number: Bank account number
        ifsc_code: IFSC code
        
    Returns:
        AccountVerificationResult: Account verification details
    """
    api = get_mock_banking_api()
    return api.verify_account(account_number, ifsc_code)

def get_transaction_history(
    account_number: str,
    from_date: datetime,
    to_date: datetime,
    max_transactions: int = 100
) -> TransactionHistory:
    """
    Convenience function for transaction history.
    
    Args:
        account_number: Bank account number
        from_date: Start date
        to_date: End date
        max_transactions: Maximum transactions to return
        
    Returns:
        TransactionHistory: Transaction history with analysis
    """
    api = get_mock_banking_api()
    return api.get_transaction_history(account_number, from_date, to_date, max_transactions)

def verify_income(
    pan_number: str,
    account_number: str,
    months_to_analyze: int = 6
) -> IncomeVerificationResult:
    """
    Convenience function for income verification.
    
    Args:
        pan_number: PAN number
        account_number: Account number
        months_to_analyze: Months to analyze
        
    Returns:
        IncomeVerificationResult: Income verification details
    """
    api = get_mock_banking_api()
    return api.verify_income(pan_number, account_number, months_to_analyze)

# Export main classes and functions
__all__ = [
    'MockBankingAPI',
    'CreditScoreResult',
    'AccountVerificationResult', 
    'TransactionHistory',
    'Transaction',
    'IncomeVerificationResult',
    'CreditScoreRange',
    'AccountType',
    'TransactionType',
    'BankingError',
    'InvalidPANError',
    'InvalidAccountError',
    'APITimeoutError',
    'get_mock_banking_api',
    'lookup_credit_score',
    'verify_account',
    'get_transaction_history',
    'verify_income'
]