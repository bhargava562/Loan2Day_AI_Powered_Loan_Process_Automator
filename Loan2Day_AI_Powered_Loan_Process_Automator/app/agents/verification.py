"""
Verification Agent - KYC Processing and Security Validation

This agent handles identity verification, fraud detection, and document processing
using the SGS (Spectral-Graph Sentinel) security module. It implements the SBEF
(Semantic-Bayesian Evidence Fusion) algorithm to resolve conflicts between user
input and OCR-extracted document data.

Key Responsibilities:
- KYC document processing through SGS security scanning
- OCR text extraction and validation
- Fraud score calculation and risk assessment
- SBEF algorithm for data conflict resolution
- Trust score calculation for conflicting information

Architecture: Worker Agent in Master-Worker pattern
Security: All uploads processed through SGS.scan_topology()
Data Resolution: SBEF algorithm prevents application rejection due to minor conflicts

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Optional, Any, Union, BinaryIO
from decimal import Decimal
from datetime import datetime
import logging
import re
import hashlib
from enum import Enum

# Import core modules
from app.core.sgs import scan_topology, SecurityScore, SecurityThreatLevel
from app.models.pydantic_models import (
    AgentState, KYCDocument, TrustScore, DocumentType, 
    VerificationStatus, UserProfile
)

# Configure logger
logger = logging.getLogger(__name__)

class DataConflictType(Enum):
    """Types of data conflicts between user input and documents."""
    NAME_MISMATCH = "NAME_MISMATCH"
    DATE_MISMATCH = "DATE_MISMATCH"
    ADDRESS_MISMATCH = "ADDRESS_MISMATCH"
    PHONE_MISMATCH = "PHONE_MISMATCH"
    INCOME_MISMATCH = "INCOME_MISMATCH"
    EMPLOYMENT_MISMATCH = "EMPLOYMENT_MISMATCH"

class VerificationError(Exception):
    """Base exception for verification agent errors."""
    pass

class DocumentProcessingError(VerificationError):
    """Raised when document processing fails."""
    pass

class OCRExtractionError(VerificationError):
    """Raised when OCR text extraction fails."""
    pass

class FraudDetectionError(VerificationError):
    """Raised when fraud detection processing fails."""
    pass

class KYCResult:
    """
    Comprehensive KYC processing result with security and verification details.
    """
    
    def __init__(
        self,
        document_id: str,
        document_type: DocumentType,
        security_score: SecurityScore,
        ocr_extracted_data: Dict[str, str],
        verification_status: VerificationStatus,
        fraud_score: float,
        trust_score: Optional[TrustScore],
        data_conflicts: List[Dict[str, Any]],
        verification_confidence: float,
        processing_timestamp: datetime
    ):
        self.document_id = document_id
        self.document_type = document_type
        self.security_score = security_score
        self.ocr_extracted_data = ocr_extracted_data
        self.verification_status = verification_status
        self.fraud_score = fraud_score
        self.trust_score = trust_score
        self.data_conflicts = data_conflicts
        self.verification_confidence = verification_confidence
        self.processing_timestamp = processing_timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert KYC result to dictionary for AgentState storage."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "security_score": self.security_score.to_dict(),
            "ocr_extracted_data": self.ocr_extracted_data,
            "verification_status": self.verification_status.value,
            "fraud_score": self.fraud_score,
            "trust_score": self.trust_score.to_dict() if self.trust_score else None,
            "data_conflicts": self.data_conflicts,
            "verification_confidence": self.verification_confidence,
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "is_verified": self.is_verified(),
            "risk_factors": self.get_risk_factors()
        }
    
    def is_verified(self) -> bool:
        """Check if document is successfully verified."""
        return (
            self.verification_status == VerificationStatus.VERIFIED and
            self.security_score.is_safe() and
            self.fraud_score < 0.5 and
            self.verification_confidence > 0.7
        )
    
    def get_risk_factors(self) -> List[str]:
        """Get list of identified risk factors."""
        risk_factors = []
        
        if not self.security_score.is_safe():
            risk_factors.extend(self.security_score.get_risk_factors())
        
        if self.fraud_score > 0.7:
            risk_factors.append(f"High fraud risk score: {self.fraud_score:.2f}")
        
        if self.verification_confidence < 0.5:
            risk_factors.append(f"Low verification confidence: {self.verification_confidence:.2f}")
        
        if len(self.data_conflicts) > 2:
            risk_factors.append(f"Multiple data conflicts: {len(self.data_conflicts)}")
        
        return risk_factors

class OCRProcessor:
    """
    OCR text extraction and processing engine.
    
    In production, this would integrate with advanced OCR services like
    Google Cloud Vision, AWS Textract, or Azure Cognitive Services.
    For development, it provides realistic mock OCR capabilities.
    """
    
    def __init__(self):
        """Initialize OCR processor."""
        logger.info("OCRProcessor initialized (mock mode)")
    
    def extract_text_from_document(
        self,
        file_data: bytes,
        document_type: DocumentType
    ) -> Dict[str, str]:
        """
        Extract structured text data from document using OCR.
        
        Args:
            file_data: Raw document file bytes
            document_type: Type of document being processed
            
        Returns:
            Dict[str, str]: Extracted structured data
        """
        logger.info(f"Extracting text from {document_type.value} document")
        
        # Generate deterministic OCR results based on file hash
        file_hash = hashlib.md5(file_data).hexdigest()
        hash_int = int(file_hash[:8], 16)
        
        # Mock OCR extraction based on document type
        if document_type == DocumentType.PAN_CARD:
            return self._extract_pan_card_data(hash_int)
        elif document_type == DocumentType.AADHAAR_CARD:
            return self._extract_aadhaar_data(hash_int)
        elif document_type == DocumentType.BANK_STATEMENT:
            return self._extract_bank_statement_data(hash_int)
        elif document_type == DocumentType.SALARY_SLIP:
            return self._extract_salary_slip_data(hash_int)
        else:
            return self._extract_generic_document_data(hash_int)
    
    def _extract_pan_card_data(self, hash_int: int) -> Dict[str, str]:
        """Extract data from PAN card."""
        names = ["RAJESH KUMAR SHARMA", "PRIYA PATEL", "AMIT SINGH", "SUNITA REDDY"]
        
        return {
            "document_type": "PAN Card",
            "pan_number": f"ABCDE{hash_int % 10000:04d}F",
            "name": names[hash_int % len(names)],
            "father_name": f"FATHER OF {names[hash_int % len(names)].split()[0]}",
            "date_of_birth": f"{(hash_int % 28) + 1:02d}/{(hash_int % 12) + 1:02d}/{1980 + (hash_int % 25)}",
            "signature_present": "Yes"
        }
    
    def _extract_aadhaar_data(self, hash_int: int) -> Dict[str, str]:
        """Extract data from Aadhaar card."""
        names = ["Rajesh Kumar Sharma", "Priya Patel", "Amit Singh", "Sunita Reddy"]
        addresses = [
            "123 MG Road, Bangalore, Karnataka 560001",
            "456 Park Street, Mumbai, Maharashtra 400001",
            "789 Anna Salai, Chennai, Tamil Nadu 600001"
        ]
        
        return {
            "document_type": "Aadhaar Card",
            "aadhaar_number": f"{hash_int % 10000:04d} {hash_int % 10000:04d} {hash_int % 10000:04d}",
            "name": names[hash_int % len(names)],
            "date_of_birth": f"{(hash_int % 28) + 1:02d}/{(hash_int % 12) + 1:02d}/{1980 + (hash_int % 25)}",
            "gender": "Male" if hash_int % 2 == 0 else "Female",
            "address": addresses[hash_int % len(addresses)],
            "phone": f"+91{hash_int % 10000000000:010d}"
        }
    
    def _extract_bank_statement_data(self, hash_int: int) -> Dict[str, str]:
        """Extract data from bank statement."""
        banks = ["HDFC Bank", "ICICI Bank", "SBI", "Axis Bank"]
        
        return {
            "document_type": "Bank Statement",
            "account_holder_name": f"Account Holder {hash_int % 1000}",
            "account_number": f"{hash_int % 100000000000000:015d}",
            "bank_name": banks[hash_int % len(banks)],
            "statement_period": "01/01/2024 to 31/03/2024",
            "average_balance": f"₹{(hash_int % 500000) + 50000:,}",
            "total_credits": f"₹{(hash_int % 1000000) + 100000:,}",
            "salary_credits": str((hash_int % 3) + 1)
        }
    
    def _extract_salary_slip_data(self, hash_int: int) -> Dict[str, str]:
        """Extract data from salary slip."""
        companies = ["TCS Limited", "Infosys Limited", "Wipro Limited", "HCL Technologies"]
        
        return {
            "document_type": "Salary Slip",
            "employee_name": f"Employee {hash_int % 1000}",
            "employee_id": f"EMP{hash_int % 100000:05d}",
            "company_name": companies[hash_int % len(companies)],
            "month_year": "March 2024",
            "gross_salary": f"₹{(hash_int % 500000) + 50000:,}",
            "net_salary": f"₹{int(((hash_int % 500000) + 50000) * 0.8):,}",
            "designation": "Software Engineer"
        }
    
    def _extract_generic_document_data(self, hash_int: int) -> Dict[str, str]:
        """Extract data from generic document."""
        return {
            "document_type": "Generic Document",
            "extracted_text": f"Sample extracted text content {hash_int}",
            "confidence": f"{(hash_int % 40) + 60}%",
            "language": "English"
        }

class SBEFProcessor:
    """
    Semantic-Bayesian Evidence Fusion (SBEF) Algorithm Processor.
    
    This class implements the SBEF algorithm to resolve conflicts between
    user-provided data and OCR-extracted document data. Instead of rejecting
    applications due to minor discrepancies, SBEF calculates trust scores
    and provides intelligent conflict resolution.
    """
    
    def __init__(self):
        """Initialize SBEF processor."""
        logger.info("SBEFProcessor initialized")
    
    def detect_data_conflicts(
        self,
        user_data: Dict[str, str],
        ocr_data: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between user input and OCR extracted data.
        
        Args:
            user_data: User-provided data
            ocr_data: OCR-extracted data
            
        Returns:
            List[Dict[str, Any]]: List of detected conflicts
        """
        logger.info("Detecting data conflicts between user input and OCR data")
        
        conflicts = []
        
        # Common field mappings for comparison
        field_mappings = {
            "name": ["name", "employee_name", "account_holder_name"],
            "phone": ["phone", "mobile", "contact"],
            "address": ["address", "permanent_address", "current_address"],
            "company": ["company_name", "employer", "organization"]
        }
        
        for user_field, user_value in user_data.items():
            if not user_value:
                continue
                
            # Find corresponding OCR fields
            ocr_fields = field_mappings.get(user_field, [user_field])
            
            for ocr_field in ocr_fields:
                if ocr_field in ocr_data and ocr_data[ocr_field]:
                    ocr_value = ocr_data[ocr_field]
                    
                    # Calculate similarity and detect conflicts
                    similarity = self._calculate_text_similarity(user_value, ocr_value)
                    
                    if similarity < 0.8:  # Threshold for conflict detection
                        conflict = {
                            "conflict_type": self._determine_conflict_type(user_field),
                            "user_field": user_field,
                            "ocr_field": ocr_field,
                            "user_value": user_value,
                            "ocr_value": ocr_value,
                            "similarity_score": similarity,
                            "severity": self._determine_conflict_severity(similarity)
                        }
                        conflicts.append(conflict)
        
        logger.info(f"Detected {len(conflicts)} data conflicts")
        return conflicts
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        # Normalize texts for comparison
        norm_text1 = self._normalize_text(text1)
        norm_text2 = self._normalize_text(text2)
        
        # Simple Jaccard similarity for demonstration
        # In production, use more sophisticated algorithms like Levenshtein distance
        words1 = set(norm_text1.split())
        words2 = set(norm_text2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase, remove extra spaces and special characters
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _determine_conflict_type(self, field_name: str) -> DataConflictType:
        """Determine the type of data conflict."""
        if "name" in field_name.lower():
            return DataConflictType.NAME_MISMATCH
        elif "phone" in field_name.lower() or "mobile" in field_name.lower():
            return DataConflictType.PHONE_MISMATCH
        elif "address" in field_name.lower():
            return DataConflictType.ADDRESS_MISMATCH
        elif "date" in field_name.lower() or "dob" in field_name.lower():
            return DataConflictType.DATE_MISMATCH
        elif "salary" in field_name.lower() or "income" in field_name.lower():
            return DataConflictType.INCOME_MISMATCH
        elif "company" in field_name.lower() or "employer" in field_name.lower():
            return DataConflictType.EMPLOYMENT_MISMATCH
        else:
            return DataConflictType.NAME_MISMATCH  # Default
    
    def _determine_conflict_severity(self, similarity_score: float) -> str:
        """Determine conflict severity based on similarity score."""
        if similarity_score > 0.6:
            return "LOW"
        elif similarity_score > 0.3:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def calculate_trust_score(
        self,
        conflicts: List[Dict[str, Any]],
        user_confidence: float = 0.7,
        ocr_confidence: float = 0.8
    ) -> TrustScore:
        """
        Calculate trust score using SBEF algorithm.
        
        Args:
            conflicts: List of detected conflicts
            user_confidence: Base confidence in user input (0.0-1.0)
            ocr_confidence: Base confidence in OCR data (0.0-1.0)
            
        Returns:
            TrustScore: Calculated trust score with resolution method
        """
        logger.info(f"Calculating trust score for {len(conflicts)} conflicts")
        
        if not conflicts:
            # No conflicts - high trust in both sources
            return TrustScore(
                user_input_confidence=user_confidence,
                ocr_confidence=ocr_confidence,
                final_trust_score=min(user_confidence, ocr_confidence),
                resolution_method="no_conflicts",
                conflict_details={"message": "No data conflicts detected"}
            )
        
        # Analyze conflict patterns
        high_severity_conflicts = [c for c in conflicts if c["severity"] == "HIGH"]
        medium_severity_conflicts = [c for c in conflicts if c["severity"] == "MEDIUM"]
        low_severity_conflicts = [c for c in conflicts if c["severity"] == "LOW"]
        
        # Calculate confidence adjustments based on conflicts
        user_adjustment = 1.0
        ocr_adjustment = 1.0
        
        # Penalize based on conflict severity
        user_adjustment -= len(high_severity_conflicts) * 0.3
        user_adjustment -= len(medium_severity_conflicts) * 0.15
        user_adjustment -= len(low_severity_conflicts) * 0.05
        
        ocr_adjustment -= len(high_severity_conflicts) * 0.2
        ocr_adjustment -= len(medium_severity_conflicts) * 0.1
        ocr_adjustment -= len(low_severity_conflicts) * 0.03
        
        # Ensure adjustments don't go below 0
        user_adjustment = max(0.1, user_adjustment)
        ocr_adjustment = max(0.1, ocr_adjustment)
        
        # Calculate adjusted confidences
        adjusted_user_confidence = user_confidence * user_adjustment
        adjusted_ocr_confidence = ocr_confidence * ocr_adjustment
        
        # Determine resolution method and final trust score
        if len(high_severity_conflicts) > 2:
            resolution_method = "manual_review"
            final_trust_score = 0.3  # Low trust, requires manual review
        elif adjusted_ocr_confidence > adjusted_user_confidence * 1.2:
            resolution_method = "ocr_priority"
            final_trust_score = adjusted_ocr_confidence
        elif adjusted_user_confidence > adjusted_ocr_confidence * 1.2:
            resolution_method = "user_priority"
            final_trust_score = adjusted_user_confidence
        else:
            resolution_method = "weighted_average"
            final_trust_score = (adjusted_user_confidence + adjusted_ocr_confidence) / 2
        
        trust_score = TrustScore(
            user_input_confidence=adjusted_user_confidence,
            ocr_confidence=adjusted_ocr_confidence,
            final_trust_score=final_trust_score,
            resolution_method=resolution_method,
            conflict_details={
                "total_conflicts": len(conflicts),
                "high_severity": len(high_severity_conflicts),
                "medium_severity": len(medium_severity_conflicts),
                "low_severity": len(low_severity_conflicts),
                "conflicts": conflicts
            }
        )
        
        logger.info(
            f"Trust score calculated: Final={final_trust_score:.2f}, "
            f"Method={resolution_method}, Conflicts={len(conflicts)}"
        )
        
        return trust_score

class VerificationAgent:
    """
    Verification Agent - The Detective of Loan2Day.
    
    This agent handles comprehensive identity verification and fraud detection
    using the SGS (Spectral-Graph Sentinel) security module and SBEF algorithm
    for intelligent data conflict resolution.
    
    All file uploads MUST pass through SGS.scan_topology() before processing.
    """
    
    def __init__(self):
        """Initialize the Verification Agent."""
        self.ocr_processor = OCRProcessor()
        self.sbef_processor = SBEFProcessor()
        logger.info("VerificationAgent initialized successfully")
    
    async def process_kyc_document(
        self,
        file_data: Union[bytes, BinaryIO],
        document_type: DocumentType,
        filename: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> KYCResult:
        """
        Process KYC document through complete security and verification pipeline.
        
        This method implements the complete KYC processing workflow:
        1. SGS security scanning (mandatory)
        2. OCR text extraction
        3. Fraud score calculation
        4. Verification status determination
        
        Args:
            file_data: Document file data
            document_type: Type of KYC document
            filename: Original filename (optional)
            user_id: User identifier for audit logging
            
        Returns:
            KYCResult: Comprehensive KYC processing result
            
        Raises:
            DocumentProcessingError: If document processing fails
            OCRExtractionError: If OCR extraction fails
        """
        logger.info(
            f"Processing KYC document - Type: {document_type.value}, "
            f"User: {user_id}, File: {filename}"
        )
        
        try:
            # Step 1: Mandatory SGS security scanning
            logger.info("Step 1: SGS security scanning")
            security_score = scan_topology(file_data, filename, user_id)
            
            if not security_score.is_safe():
                logger.warning(
                    f"Document failed SGS security scan - Score: {security_score.overall_score}, "
                    f"Threat Level: {security_score.threat_level.value}"
                )
                
                # Return early with security failure
                return KYCResult(
                    document_id=f"DOC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id or 'UNKNOWN'}",
                    document_type=document_type,
                    security_score=security_score,
                    ocr_extracted_data={},
                    verification_status=VerificationStatus.REJECTED,
                    fraud_score=0.9,  # High fraud score for security failures
                    trust_score=None,
                    data_conflicts=[],
                    verification_confidence=0.1,
                    processing_timestamp=datetime.now()
                )
            
            # Step 2: OCR text extraction
            logger.info("Step 2: OCR text extraction")
            if isinstance(file_data, bytes):
                file_bytes = file_data
            else:
                file_bytes = file_data.read()
                if hasattr(file_data, 'seek'):
                    file_data.seek(0)
            
            ocr_extracted_data = self.ocr_processor.extract_text_from_document(
                file_bytes, document_type
            )
            
            # Step 3: Calculate fraud score
            logger.info("Step 3: Fraud score calculation")
            fraud_score = await self.calculate_fraud_score(
                security_score, ocr_extracted_data, document_type
            )
            
            # Step 4: Determine verification status
            verification_status = self._determine_verification_status(
                security_score, fraud_score, ocr_extracted_data
            )
            
            # Step 5: Calculate verification confidence
            verification_confidence = self._calculate_verification_confidence(
                security_score, fraud_score, ocr_extracted_data
            )
            
            # Generate document ID
            document_id = f"DOC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id or 'UNKNOWN'}"
            
            kyc_result = KYCResult(
                document_id=document_id,
                document_type=document_type,
                security_score=security_score,
                ocr_extracted_data=ocr_extracted_data,
                verification_status=verification_status,
                fraud_score=fraud_score,
                trust_score=None,  # Will be set during conflict resolution
                data_conflicts=[],  # Will be populated during conflict resolution
                verification_confidence=verification_confidence,
                processing_timestamp=datetime.now()
            )
            
            logger.info(
                f"KYC document processing completed - ID: {document_id}, "
                f"Status: {verification_status.value}, Fraud Score: {fraud_score:.2f}"
            )
            
            return kyc_result
            
        except Exception as e:
            logger.error(f"KYC document processing failed: {str(e)}")
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")
    
    async def calculate_fraud_score(
        self,
        security_score: SecurityScore,
        ocr_data: Dict[str, str],
        document_type: DocumentType
    ) -> float:
        """
        Calculate fraud risk score based on security analysis and document content.
        
        Args:
            security_score: SGS security analysis result
            ocr_data: OCR extracted data
            document_type: Type of document
            
        Returns:
            float: Fraud score (0.0 = low risk, 1.0 = high risk)
        """
        logger.info(f"Calculating fraud score for {document_type.value}")
        
        # Base fraud score from security analysis (inverted - lower security = higher fraud risk)
        base_fraud_score = 1.0 - security_score.overall_score
        
        # Adjust based on document-specific factors
        document_fraud_adjustment = 0.0
        
        # Check for suspicious patterns in OCR data
        if document_type == DocumentType.PAN_CARD:
            document_fraud_adjustment = self._analyze_pan_card_fraud_indicators(ocr_data)
        elif document_type == DocumentType.AADHAAR_CARD:
            document_fraud_adjustment = self._analyze_aadhaar_fraud_indicators(ocr_data)
        elif document_type == DocumentType.BANK_STATEMENT:
            document_fraud_adjustment = self._analyze_bank_statement_fraud_indicators(ocr_data)
        elif document_type == DocumentType.SALARY_SLIP:
            document_fraud_adjustment = self._analyze_salary_slip_fraud_indicators(ocr_data)
        
        # Combine scores (weighted average)
        final_fraud_score = (base_fraud_score * 0.7) + (document_fraud_adjustment * 0.3)
        
        # Ensure score is within valid range
        final_fraud_score = max(0.0, min(1.0, final_fraud_score))
        
        logger.info(f"Fraud score calculated: {final_fraud_score:.2f}")
        return final_fraud_score
    
    def _analyze_pan_card_fraud_indicators(self, ocr_data: Dict[str, str]) -> float:
        """Analyze PAN card for fraud indicators."""
        fraud_score = 0.0
        
        # Check PAN number format
        pan_number = ocr_data.get("pan_number", "")
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan_number):
            fraud_score += 0.3
        
        # Check for missing critical fields
        critical_fields = ["name", "father_name", "date_of_birth"]
        missing_fields = [field for field in critical_fields if not ocr_data.get(field)]
        fraud_score += len(missing_fields) * 0.2
        
        return min(1.0, fraud_score)
    
    def _analyze_aadhaar_fraud_indicators(self, ocr_data: Dict[str, str]) -> float:
        """Analyze Aadhaar card for fraud indicators."""
        fraud_score = 0.0
        
        # Check Aadhaar number format (12 digits with spaces)
        aadhaar_number = ocr_data.get("aadhaar_number", "").replace(" ", "")
        if not aadhaar_number.isdigit() or len(aadhaar_number) != 12:
            fraud_score += 0.4
        
        # Check for missing critical fields
        critical_fields = ["name", "date_of_birth", "address"]
        missing_fields = [field for field in critical_fields if not ocr_data.get(field)]
        fraud_score += len(missing_fields) * 0.2
        
        return min(1.0, fraud_score)
    
    def _analyze_bank_statement_fraud_indicators(self, ocr_data: Dict[str, str]) -> float:
        """Analyze bank statement for fraud indicators."""
        fraud_score = 0.0
        
        # Check for missing critical fields
        critical_fields = ["account_holder_name", "account_number", "bank_name"]
        missing_fields = [field for field in critical_fields if not ocr_data.get(field)]
        fraud_score += len(missing_fields) * 0.25
        
        # Check for suspicious patterns
        if "average_balance" in ocr_data:
            try:
                balance_str = ocr_data["average_balance"].replace("₹", "").replace(",", "")
                balance = float(balance_str)
                if balance < 0:  # Negative balance is suspicious
                    fraud_score += 0.3
            except (ValueError, AttributeError):
                fraud_score += 0.2  # Invalid balance format
        
        return min(1.0, fraud_score)
    
    def _analyze_salary_slip_fraud_indicators(self, ocr_data: Dict[str, str]) -> float:
        """Analyze salary slip for fraud indicators."""
        fraud_score = 0.0
        
        # Check for missing critical fields
        critical_fields = ["employee_name", "company_name", "gross_salary"]
        missing_fields = [field for field in critical_fields if not ocr_data.get(field)]
        fraud_score += len(missing_fields) * 0.25
        
        # Check salary consistency
        try:
            gross_salary_str = ocr_data.get("gross_salary", "").replace("₹", "").replace(",", "")
            net_salary_str = ocr_data.get("net_salary", "").replace("₹", "").replace(",", "")
            
            if gross_salary_str and net_salary_str:
                gross_salary = float(gross_salary_str)
                net_salary = float(net_salary_str)
                
                if net_salary > gross_salary:  # Net > Gross is impossible
                    fraud_score += 0.5
                elif net_salary < gross_salary * 0.5:  # Very low net salary
                    fraud_score += 0.2
        except (ValueError, AttributeError):
            fraud_score += 0.1  # Invalid salary format
        
        return min(1.0, fraud_score)
    
    def _determine_verification_status(
        self,
        security_score: SecurityScore,
        fraud_score: float,
        ocr_data: Dict[str, str]
    ) -> VerificationStatus:
        """Determine verification status based on analysis results."""
        
        # Reject if security scan failed
        if not security_score.is_safe():
            return VerificationStatus.REJECTED
        
        # Reject if high fraud risk
        if fraud_score > 0.7:
            return VerificationStatus.REJECTED
        
        # Mark as suspicious if moderate fraud risk
        if fraud_score > 0.5:
            return VerificationStatus.SUSPICIOUS
        
        # Check if critical data is missing
        if not ocr_data or len(ocr_data) < 3:
            return VerificationStatus.PENDING
        
        # Verify if all conditions are met
        return VerificationStatus.VERIFIED
    
    def _calculate_verification_confidence(
        self,
        security_score: SecurityScore,
        fraud_score: float,
        ocr_data: Dict[str, str]
    ) -> float:
        """Calculate overall verification confidence."""
        
        # Base confidence from security score
        security_confidence = security_score.overall_score
        
        # Fraud confidence (inverted)
        fraud_confidence = 1.0 - fraud_score
        
        # OCR data completeness confidence
        expected_fields = 5  # Expected number of fields for complete document
        actual_fields = len([v for v in ocr_data.values() if v and v.strip()])
        ocr_confidence = min(1.0, actual_fields / expected_fields)
        
        # Weighted average
        overall_confidence = (
            security_confidence * 0.4 +
            fraud_confidence * 0.4 +
            ocr_confidence * 0.2
        )
        
        return max(0.0, min(1.0, overall_confidence))
    
    async def resolve_data_conflicts(
        self,
        user_data: Dict[str, str],
        kyc_result: KYCResult
    ) -> TrustScore:
        """
        Resolve data conflicts using SBEF algorithm.
        
        Args:
            user_data: User-provided data
            kyc_result: KYC processing result with OCR data
            
        Returns:
            TrustScore: SBEF trust score with conflict resolution
        """
        logger.info("Resolving data conflicts using SBEF algorithm")
        
        # Detect conflicts between user input and OCR data
        conflicts = self.sbef_processor.detect_data_conflicts(
            user_data, kyc_result.ocr_extracted_data
        )
        
        # Calculate base confidences
        user_confidence = 0.7  # Default user input confidence
        ocr_confidence = kyc_result.verification_confidence
        
        # Adjust OCR confidence based on security and fraud scores
        if kyc_result.security_score.is_safe():
            ocr_confidence *= 1.1  # Boost for secure documents
        
        if kyc_result.fraud_score < 0.3:
            ocr_confidence *= 1.1  # Boost for low fraud risk
        elif kyc_result.fraud_score > 0.6:
            ocr_confidence *= 0.8  # Reduce for high fraud risk
        
        # Calculate trust score using SBEF
        trust_score = self.sbef_processor.calculate_trust_score(
            conflicts, user_confidence, ocr_confidence
        )
        
        # Update KYC result with conflict information
        kyc_result.data_conflicts = conflicts
        kyc_result.trust_score = trust_score
        
        logger.info(
            f"Data conflict resolution completed - Trust Score: {trust_score.final_trust_score:.2f}, "
            f"Method: {trust_score.resolution_method}, Conflicts: {len(conflicts)}"
        )
        
        return trust_score
    
    async def process_verification_request(
        self,
        agent_state: AgentState,
        user_provided_data: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process complete verification request and update agent state.
        
        This is the main entry point for verification processing in the
        Master-Worker agent pattern. It processes all KYC documents and
        resolves data conflicts using SBEF algorithm.
        
        Args:
            agent_state: Current agent state with KYC documents
            user_provided_data: User-provided data for conflict resolution
            
        Returns:
            Dict[str, Any]: Verification result with updated KYC status
            
        Raises:
            VerificationError: If verification processing fails
        """
        logger.info(f"Processing verification request for session: {agent_state.session_id}")
        
        try:
            verification_results = []
            overall_verification_status = VerificationStatus.VERIFIED
            total_fraud_score = 0.0
            
            # Process each KYC document
            for kyc_doc in agent_state.kyc_documents:
                logger.info(f"Processing document: {kyc_doc.document_type.value}")
                
                # Note: In this implementation, we assume documents are already processed
                # In a real system, you would re-process or validate existing results
                
                # Create KYC result from existing document data
                kyc_result = KYCResult(
                    document_id=f"DOC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    document_type=kyc_doc.document_type,
                    security_score=None,  # Would be populated from actual SGS scan
                    ocr_extracted_data=kyc_doc.extracted_data or {},
                    verification_status=kyc_doc.verification_status,
                    fraud_score=0.3,  # Mock fraud score
                    trust_score=kyc_doc.trust_score,
                    data_conflicts=[],
                    verification_confidence=0.8,  # Mock confidence
                    processing_timestamp=datetime.now()
                )
                
                # Resolve data conflicts if user data is provided
                if user_provided_data:
                    trust_score = await self.resolve_data_conflicts(
                        user_provided_data, kyc_result
                    )
                    kyc_result.trust_score = trust_score
                
                verification_results.append(kyc_result.to_dict())
                
                # Update overall status (most restrictive wins)
                if kyc_result.verification_status == VerificationStatus.REJECTED:
                    overall_verification_status = VerificationStatus.REJECTED
                elif (kyc_result.verification_status == VerificationStatus.SUSPICIOUS and 
                      overall_verification_status != VerificationStatus.REJECTED):
                    overall_verification_status = VerificationStatus.SUSPICIOUS
                elif (kyc_result.verification_status == VerificationStatus.PENDING and 
                      overall_verification_status == VerificationStatus.VERIFIED):
                    overall_verification_status = VerificationStatus.PENDING
                
                total_fraud_score += kyc_result.fraud_score
            
            # Calculate average fraud score
            avg_fraud_score = total_fraud_score / len(agent_state.kyc_documents) if agent_state.kyc_documents else 0.0
            
            # Prepare verification result
            verification_result = {
                "overall_status": overall_verification_status.value,
                "average_fraud_score": avg_fraud_score,
                "document_results": verification_results,
                "total_documents_processed": len(agent_state.kyc_documents),
                "verified_documents": len([r for r in verification_results if r["is_verified"]]),
                "processing_timestamp": datetime.now().isoformat(),
                "sbef_conflicts_resolved": sum(len(r.get("data_conflicts", [])) for r in verification_results),
                "recommendation": self._get_verification_recommendation(overall_verification_status, avg_fraud_score)
            }
            
            logger.info(
                f"Verification processing completed - Session: {agent_state.session_id}, "
                f"Status: {overall_verification_status.value}, "
                f"Fraud Score: {avg_fraud_score:.2f}"
            )
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Verification processing failed: {str(e)}")
            raise VerificationError(f"Verification processing failed: {str(e)}")
    
    def _get_verification_recommendation(
        self,
        status: VerificationStatus,
        fraud_score: float
    ) -> str:
        """Get verification recommendation based on status and fraud score."""
        
        if status == VerificationStatus.VERIFIED and fraud_score < 0.3:
            return "Proceed with loan processing - Low risk profile"
        elif status == VerificationStatus.VERIFIED and fraud_score < 0.5:
            return "Proceed with caution - Moderate risk profile"
        elif status == VerificationStatus.SUSPICIOUS:
            return "Manual review recommended - Suspicious indicators detected"
        elif status == VerificationStatus.PENDING:
            return "Additional documentation required"
        else:
            return "Reject application - High fraud risk or security concerns"

# Export main classes and functions
__all__ = [
    'VerificationAgent',
    'KYCResult',
    'OCRProcessor',
    'SBEFProcessor',
    'DataConflictType',
    'VerificationError',
    'DocumentProcessingError',
    'OCRExtractionError',
    'FraudDetectionError'
]