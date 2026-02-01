"""
PDF Generation Service - Sanction Letter Generation

This service generates legally binding PDF sanction letters using ReportLab
upon loan approval. All monetary values are populated from verified AgentState
data following the LQM Standard with decimal.Decimal precision.

Key Features:
- ReportLab-based PDF generation with professional formatting
- Verified data population from AgentState (name, loan amount, EMI, rate)
- Regulatory disclosures and legal terms inclusion
- Secure file generation with unique identifiers
- Real-time generation upon loan approval

Architecture: Service layer in Routes -> Services -> Repositories pattern
Security: Data verification before PDF generation
Compliance: Regulatory disclosures and legal terms

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Optional, Any, BinaryIO
from decimal import Decimal
from datetime import datetime, timedelta
import logging
import os
import uuid
from pathlib import Path
import io

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import black, blue, red, grey
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib import colors

# Import core modules
from app.models.pydantic_models import AgentState, UserProfile, LoanRequest, EMICalculation
from app.core.lqm import validate_monetary_input
from app.core.config import settings

# Configure logger
logger = logging.getLogger(__name__)

class PDFGenerationError(Exception):
    """Base exception for PDF generation errors."""
    pass

class DataValidationError(PDFGenerationError):
    """Raised when required data is missing or invalid."""
    pass

class FileGenerationError(PDFGenerationError):
    """Raised when PDF file generation fails."""
    pass

class SanctionLetterData:
    """
    Data container for sanction letter generation with validation.
    
    All monetary values use decimal.Decimal following LQM Standard.
    """
    
    def __init__(
        self,
        borrower_name: str,
        loan_amount_in_cents: Decimal,
        emi_in_cents: Decimal,
        interest_rate: Decimal,
        tenure_months: int,
        loan_purpose: str,
        sanction_date: datetime,
        loan_id: str,
        borrower_address: Optional[str] = None,
        borrower_phone: Optional[str] = None,
        borrower_email: Optional[str] = None,
        processing_fee_in_cents: Optional[Decimal] = None,
        insurance_premium_in_cents: Optional[Decimal] = None
    ):
        # Validate required fields
        if not borrower_name or not borrower_name.strip():
            raise DataValidationError("Borrower name is required")
        
        if not loan_id or not loan_id.strip():
            raise DataValidationError("Loan ID is required")
        
        # Validate monetary fields using LQM
        self.borrower_name = borrower_name.strip()
        self.loan_amount_in_cents = validate_monetary_input(loan_amount_in_cents, "loan_amount")
        self.emi_in_cents = validate_monetary_input(emi_in_cents, "emi_amount")
        self.interest_rate = validate_monetary_input(interest_rate, "interest_rate")
        self.tenure_months = tenure_months
        self.loan_purpose = loan_purpose or "Personal Loan"
        self.sanction_date = sanction_date
        self.loan_id = loan_id.strip()
        
        # Optional fields
        self.borrower_address = borrower_address
        self.borrower_phone = borrower_phone
        self.borrower_email = borrower_email
        
        # Optional monetary fields
        self.processing_fee_in_cents = (
            validate_monetary_input(processing_fee_in_cents, "processing_fee") 
            if processing_fee_in_cents else Decimal('0.00')
        )
        self.insurance_premium_in_cents = (
            validate_monetary_input(insurance_premium_in_cents, "insurance_premium")
            if insurance_premium_in_cents else Decimal('0.00')
        )
        
        # Validate business logic
        if self.tenure_months <= 0 or self.tenure_months > 360:
            raise DataValidationError("Tenure must be between 1 and 360 months")
        
        if self.interest_rate < Decimal('0.00') or self.interest_rate > Decimal('50.00'):
            raise DataValidationError("Interest rate must be between 0% and 50%")
        
        if self.loan_amount_in_cents <= Decimal('0.00'):
            raise DataValidationError("Loan amount must be positive")
        
        if self.emi_in_cents <= Decimal('0.00'):
            raise DataValidationError("EMI amount must be positive")
    
    def get_display_amounts(self) -> Dict[str, str]:
        """Get formatted display amounts for PDF."""
        return {
            "loan_amount": f"₹{self.loan_amount_in_cents / 100:,.2f}",
            "emi_amount": f"₹{self.emi_in_cents / 100:,.2f}",
            "processing_fee": f"₹{self.processing_fee_in_cents / 100:,.2f}",
            "insurance_premium": f"₹{self.insurance_premium_in_cents / 100:,.2f}",
            "total_interest": f"₹{(self.emi_in_cents * self.tenure_months - self.loan_amount_in_cents) / 100:,.2f}",
            "total_repayment": f"₹{(self.emi_in_cents * self.tenure_months) / 100:,.2f}"
        }

class PDFService:
    """
    PDF Generation Service for Loan2Day sanction letters.
    
    This service generates professional, legally binding PDF sanction letters
    with verified data from AgentState and proper regulatory compliance.
    """
    
    def __init__(self, output_directory: Optional[str] = None):
        """
        Initialize PDF service.
        
        Args:
            output_directory: Directory for PDF output (optional)
        """
        self.output_directory = Path(output_directory or "generated_pdfs")
        self.output_directory.mkdir(exist_ok=True)
        
        # Company information
        self.company_name = "Loan2Day Fintech Private Limited"
        self.company_address = "123 Financial District, Bangalore, Karnataka 560001"
        self.company_phone = "+91-80-1234-5678"
        self.company_email = "support@loan2day.com"
        self.company_website = "www.loan2day.com"
        self.rbi_license = "NBFC-P2P-2024-001"
        
        logger.info("PDFService initialized successfully")
    
    def _create_header(self, canvas, doc):
        """Create PDF header with company branding."""
        canvas.saveState()
        
        # Company logo placeholder (would be actual logo in production)
        canvas.setFont("Helvetica-Bold", 16)
        canvas.setFillColor(blue)
        canvas.drawString(50, 750, self.company_name)
        
        # Company details
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(black)
        canvas.drawString(50, 735, self.company_address)
        canvas.drawString(50, 725, f"Phone: {self.company_phone} | Email: {self.company_email}")
        canvas.drawString(50, 715, f"Website: {self.company_website} | RBI License: {self.rbi_license}")
        
        # Header line
        canvas.setStrokeColor(blue)
        canvas.setLineWidth(2)
        canvas.line(50, 705, 550, 705)
        
        canvas.restoreState()
    
    def _create_footer(self, canvas, doc):
        """Create PDF footer with legal disclaimers."""
        canvas.saveState()
        
        # Footer line
        canvas.setStrokeColor(grey)
        canvas.setLineWidth(1)
        canvas.line(50, 80, 550, 80)
        
        # Footer text
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(grey)
        canvas.drawString(50, 70, "This is a computer-generated document and does not require a signature.")
        canvas.drawString(50, 60, f"Generated on {datetime.now().strftime('%d/%m/%Y at %H:%M:%S')} | Document ID: {doc.title}")
        canvas.drawString(50, 50, "Subject to terms and conditions. Please read all terms carefully before acceptance.")
        
        # Page number
        canvas.drawRightString(550, 70, f"Page {doc.page}")
        
        canvas.restoreState()
    
    def _get_styles(self):
        """Get custom paragraph styles for PDF."""
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=blue,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomBold',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
        
        return styles
    
    def generate_sanction_letter(
        self,
        sanction_data: SanctionLetterData,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate PDF sanction letter with verified data.
        
        Args:
            sanction_data: Validated sanction letter data
            output_filename: Custom output filename (optional)
            
        Returns:
            str: Path to generated PDF file
            
        Raises:
            FileGenerationError: If PDF generation fails
        """
        logger.info(f"Generating sanction letter for loan: {sanction_data.loan_id}")
        
        try:
            # Generate filename if not provided
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"sanction_letter_{sanction_data.loan_id}_{timestamp}.pdf"
            
            output_path = self.output_directory / output_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=50,
                leftMargin=50,
                topMargin=100,
                bottomMargin=100,
                title=f"Sanction Letter - {sanction_data.loan_id}"
            )
            
            # Get styles
            styles = self._get_styles()
            
            # Build PDF content
            story = []
            
            # Title
            story.append(Paragraph("LOAN SANCTION LETTER", styles['CustomTitle']))
            story.append(Spacer(1, 20))
            
            # Loan reference
            story.append(Paragraph(f"<b>Loan Reference Number:</b> {sanction_data.loan_id}", styles['CustomBold']))
            story.append(Paragraph(f"<b>Date:</b> {sanction_data.sanction_date.strftime('%d %B, %Y')}", styles['CustomBold']))
            story.append(Spacer(1, 20))
            
            # Borrower details
            story.append(Paragraph("BORROWER DETAILS", styles['CustomHeading']))
            
            borrower_details = [
                ["Name:", sanction_data.borrower_name],
                ["Loan Amount:", sanction_data.get_display_amounts()["loan_amount"]],
                ["Purpose:", sanction_data.loan_purpose],
            ]
            
            if sanction_data.borrower_address:
                borrower_details.append(["Address:", sanction_data.borrower_address])
            if sanction_data.borrower_phone:
                borrower_details.append(["Phone:", sanction_data.borrower_phone])
            if sanction_data.borrower_email:
                borrower_details.append(["Email:", sanction_data.borrower_email])
            
            borrower_table = Table(borrower_details, colWidths=[2*inch, 4*inch])
            borrower_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            story.append(borrower_table)
            story.append(Spacer(1, 20))
            
            # Loan terms
            story.append(Paragraph("LOAN TERMS & CONDITIONS", styles['CustomHeading']))
            
            display_amounts = sanction_data.get_display_amounts()
            
            loan_terms = [
                ["Sanctioned Amount:", display_amounts["loan_amount"]],
                ["Interest Rate:", f"{sanction_data.interest_rate}% per annum"],
                ["Loan Tenure:", f"{sanction_data.tenure_months} months"],
                ["EMI Amount:", display_amounts["emi_amount"]],
                ["Total Interest:", display_amounts["total_interest"]],
                ["Total Repayment:", display_amounts["total_repayment"]],
            ]
            
            if sanction_data.processing_fee_in_cents > 0:
                loan_terms.append(["Processing Fee:", display_amounts["processing_fee"]])
            
            if sanction_data.insurance_premium_in_cents > 0:
                loan_terms.append(["Insurance Premium:", display_amounts["insurance_premium"]])
            
            loan_table = Table(loan_terms, colWidths=[2*inch, 4*inch])
            loan_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ]))
            
            story.append(loan_table)
            story.append(Spacer(1, 20))
            
            # Congratulations message
            story.append(Paragraph("CONGRATULATIONS!", styles['CustomHeading']))
            story.append(Paragraph(
                f"We are pleased to inform you that your loan application has been approved. "
                f"The sanctioned amount of <b>{display_amounts['loan_amount']}</b> will be disbursed "
                f"upon completion of documentation and fulfillment of disbursement conditions.",
                styles['CustomBody']
            ))
            story.append(Spacer(1, 15))
            
            # Important terms
            story.append(Paragraph("IMPORTANT TERMS & CONDITIONS", styles['CustomHeading']))
            
            terms_list = [
                "This sanction is valid for 30 days from the date of this letter.",
                "Disbursement is subject to verification of documents and meeting disbursement conditions.",
                "EMI will be auto-debited from your registered bank account on the due date each month.",
                "Prepayment of the loan is allowed with applicable charges as per loan agreement.",
                "This loan is governed by the terms and conditions of the loan agreement.",
                "Any changes to personal or financial information must be immediately communicated.",
                "Default in payment may result in additional charges and legal action.",
                "This sanction letter constitutes a legal document and is binding upon acceptance."
            ]
            
            for i, term in enumerate(terms_list, 1):
                story.append(Paragraph(f"{i}. {term}", styles['CustomBody']))
            
            story.append(Spacer(1, 20))
            
            # Acceptance section
            story.append(Paragraph("ACCEPTANCE", styles['CustomHeading']))
            story.append(Paragraph(
                "By proceeding with the loan disbursement, you acknowledge that you have read, "
                "understood, and agree to all the terms and conditions mentioned in this sanction letter "
                "and the detailed loan agreement.",
                styles['CustomBody']
            ))
            story.append(Spacer(1, 15))
            
            # Contact information
            story.append(Paragraph("CONTACT INFORMATION", styles['CustomHeading']))
            story.append(Paragraph(
                f"For any queries or assistance, please contact us at {self.company_phone} "
                f"or email us at {self.company_email}. You can also visit our website at {self.company_website}.",
                styles['CustomBody']
            ))
            story.append(Spacer(1, 20))
            
            # Signature section
            story.append(Paragraph("FOR LOAN2DAY FINTECH PRIVATE LIMITED", styles['CustomBold']))
            story.append(Spacer(1, 30))
            story.append(Paragraph("_________________________", styles['CustomBody']))
            story.append(Paragraph("Authorized Signatory", styles['CustomBody']))
            story.append(Paragraph("Loan Approval Department", styles['CustomBody']))
            
            # Build PDF with custom header/footer
            doc.build(
                story,
                onFirstPage=self._create_header,
                onLaterPages=self._create_header
            )
            
            logger.info(f"Sanction letter generated successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            raise FileGenerationError(f"Failed to generate PDF: {str(e)}")
    
    def generate_from_agent_state(
        self,
        agent_state: AgentState,
        loan_id: Optional[str] = None
    ) -> str:
        """
        Generate sanction letter from AgentState data.
        
        Args:
            agent_state: AgentState with verified loan data
            loan_id: Custom loan ID (optional)
            
        Returns:
            str: Path to generated PDF file
            
        Raises:
            DataValidationError: If required data is missing
            FileGenerationError: If PDF generation fails
        """
        logger.info(f"Generating sanction letter from AgentState: {agent_state.session_id}")
        
        try:
            # Validate required data
            if not agent_state.user_profile:
                raise DataValidationError("User profile is required for sanction letter")
            
            if not agent_state.emi_calculation:
                raise DataValidationError("EMI calculation is required for sanction letter")
            
            if not agent_state.loan_request:
                raise DataValidationError("Loan request is required for sanction letter")
            
            # Generate loan ID if not provided
            if not loan_id:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                loan_id = f"LN{timestamp}{agent_state.user_id[-4:]}"
            
            # Extract data from AgentState
            user_profile = agent_state.user_profile
            loan_request = agent_state.loan_request
            emi_calculation = agent_state.emi_calculation
            
            # Create sanction letter data
            sanction_data = SanctionLetterData(
                borrower_name=user_profile.name,
                loan_amount_in_cents=emi_calculation.principal_in_cents,
                emi_in_cents=emi_calculation.emi_in_cents,
                interest_rate=emi_calculation.rate_per_annum,
                tenure_months=emi_calculation.tenure_months,
                loan_purpose=loan_request.purpose.value,
                sanction_date=datetime.now(),
                loan_id=loan_id,
                borrower_address=getattr(user_profile, 'city', None),
                borrower_phone=user_profile.phone,
                borrower_email=user_profile.email,
                processing_fee_in_cents=Decimal('50000'),  # ₹500 processing fee
                insurance_premium_in_cents=Decimal('0.00')  # No insurance for now
            )
            
            # Generate PDF
            pdf_path = self.generate_sanction_letter(sanction_data)
            
            logger.info(f"Sanction letter generated from AgentState: {pdf_path}")
            return pdf_path
            
        except (DataValidationError, FileGenerationError):
            raise
        except Exception as e:
            logger.error(f"Failed to generate sanction letter from AgentState: {str(e)}")
            raise DataValidationError(f"Invalid AgentState data: {str(e)}")
    
    def cleanup_old_files(self, days_old: int = 30) -> int:
        """
        Clean up old PDF files to manage storage.
        
        Args:
            days_old: Delete files older than this many days
            
        Returns:
            int: Number of files deleted
        """
        logger.info(f"Cleaning up PDF files older than {days_old} days")
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            for pdf_file in self.output_directory.glob("*.pdf"):
                if pdf_file.stat().st_mtime < cutoff_date.timestamp():
                    pdf_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old PDF file: {pdf_file}")
            
            logger.info(f"Cleanup completed: {deleted_count} files deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"PDF cleanup failed: {str(e)}")
            return 0
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get information about generated PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict[str, Any]: PDF file information
        """
        try:
            pdf_file = Path(pdf_path)
            
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            stat = pdf_file.stat()
            
            return {
                "filename": pdf_file.name,
                "file_path": str(pdf_file),
                "file_size_bytes": stat.st_size,
                "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_readable": os.access(pdf_file, os.R_OK)
            }
            
        except Exception as e:
            logger.error(f"Failed to get PDF info: {str(e)}")
            return {"error": str(e)}

# Export main classes
__all__ = [
    'PDFService',
    'SanctionLetterData',
    'PDFGenerationError',
    'DataValidationError',
    'FileGenerationError'
]