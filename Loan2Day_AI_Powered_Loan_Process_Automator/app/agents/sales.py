"""
Sales Agent - Empathetic Communication and Plan B Logic

This agent handles empathetic customer communication using real-time sentiment analysis
and implements Plan B logic for loan rejection recovery. It maximizes conversion rates
through adaptive responses based on user emotional state and provides alternative loan
products when primary applications are rejected.

Key Responsibilities:
- Real-time sentiment analysis on user input
- Empathetic response generation based on emotional context
- Plan B logic activation for loan rejection scenarios
- Alternative loan product recommendations
- Conversion optimization through emotional intelligence

Architecture: Worker Agent in Master-Worker pattern
Empathy: Sentiment-driven communication adaptation
Recovery: Plan B logic prevents customer loss on rejection

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime
import logging
import re
from enum import Enum

# Import core modules and models
from app.core.lqm import validate_monetary_input, calculate_emi
from app.models.pydantic_models import (
    AgentState, SentimentScore, UserProfile, LoanRequest, 
    EMICalculation, LoanPurpose
)

# Configure logger
logger = logging.getLogger(__name__)

class EmotionalState(Enum):
    """User emotional states for empathetic response generation."""
    EXCITED = "EXCITED"
    OPTIMISTIC = "OPTIMISTIC"
    NEUTRAL = "NEUTRAL"
    CONCERNED = "CONCERNED"
    FRUSTRATED = "FRUSTRATED"
    ANXIOUS = "ANXIOUS"
    ANGRY = "ANGRY"

class ConversationStage(Enum):
    """Stages of loan conversation for context-aware responses."""
    INITIAL_INQUIRY = "INITIAL_INQUIRY"
    REQUIREMENT_GATHERING = "REQUIREMENT_GATHERING"
    LOAN_PRESENTATION = "LOAN_PRESENTATION"
    NEGOTIATION = "NEGOTIATION"
    OBJECTION_HANDLING = "OBJECTION_HANDLING"
    CLOSING = "CLOSING"
    REJECTION_RECOVERY = "REJECTION_RECOVERY"

class SalesError(Exception):
    """Base exception for sales agent errors."""
    pass

class SentimentAnalysisError(SalesError):
    """Raised when sentiment analysis fails."""
    pass

class PlanBGenerationError(SalesError):
    """Raised when Plan B offer generation fails."""
    pass

class AlternativeOffer:
    """
    Alternative loan offer for Plan B logic.
    
    All monetary values use decimal.Decimal following LQM Standard.
    """
    
    def __init__(
        self,
        offer_id: str,
        product_name: str,
        amount_in_cents: Decimal,
        tenure_months: int,
        interest_rate: Decimal,
        emi_in_cents: Decimal,
        special_features: List[str],
        eligibility_criteria: List[str],
        approval_probability: float,
        offer_valid_until: datetime
    ):
        self.offer_id = offer_id
        self.product_name = product_name
        self.amount_in_cents = amount_in_cents
        self.tenure_months = tenure_months
        self.interest_rate = interest_rate
        self.emi_in_cents = emi_in_cents
        self.special_features = special_features
        self.eligibility_criteria = eligibility_criteria
        self.approval_probability = approval_probability
        self.offer_valid_until = offer_valid_until
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alternative offer to dictionary."""
        return {
            "offer_id": self.offer_id,
            "product_name": self.product_name,
            "amount_in_cents": str(self.amount_in_cents),
            "amount_display": f"₹{self.amount_in_cents / 100:,.2f}",
            "tenure_months": self.tenure_months,
            "interest_rate": str(self.interest_rate),
            "emi_in_cents": str(self.emi_in_cents),
            "emi_display": f"₹{self.emi_in_cents / 100:,.2f}",
            "special_features": self.special_features,
            "eligibility_criteria": self.eligibility_criteria,
            "approval_probability": self.approval_probability,
            "offer_valid_until": self.offer_valid_until.isoformat()
        }

class SentimentAnalyzer:
    """
    Real-time sentiment analysis engine for empathetic communication.
    
    In production, this would integrate with advanced NLP models like
    BERT, RoBERTa, or cloud services like AWS Comprehend, Google Cloud
    Natural Language API. For development, it provides realistic sentiment
    analysis based on text patterns and keywords.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.emotion_keywords = {
            EmotionalState.EXCITED: [
                "excited", "thrilled", "amazing", "fantastic", "wonderful",
                "great", "awesome", "perfect", "excellent", "love"
            ],
            EmotionalState.OPTIMISTIC: [
                "hopeful", "confident", "positive", "good", "nice",
                "looking forward", "optimistic", "pleased", "happy"
            ],
            EmotionalState.CONCERNED: [
                "worried", "concerned", "unsure", "doubt", "hesitant",
                "nervous", "uncertain", "cautious", "careful"
            ],
            EmotionalState.FRUSTRATED: [
                "frustrated", "annoyed", "irritated", "disappointed",
                "upset", "bothered", "fed up", "tired of"
            ],
            EmotionalState.ANXIOUS: [
                "anxious", "stressed", "panic", "overwhelmed", "scared",
                "afraid", "nervous", "tense", "worried sick"
            ],
            EmotionalState.ANGRY: [
                "angry", "furious", "mad", "outraged", "livid",
                "pissed", "rage", "hate", "disgusted"
            ]
        }
        
        logger.info("SentimentAnalyzer initialized")
    
    def analyze_sentiment(self, text: str, context: Optional[Dict[str, Any]] = None) -> SentimentScore:
        """
        Analyze sentiment of user input text.
        
        Args:
            text: User input text to analyze
            context: Optional conversation context
            
        Returns:
            SentimentScore: Comprehensive sentiment analysis result
            
        Raises:
            SentimentAnalysisError: If sentiment analysis fails
        """
        logger.info("Analyzing sentiment for user input")
        
        try:
            if not text or not text.strip():
                raise SentimentAnalysisError("Empty text provided for sentiment analysis")
            
            # Normalize text for analysis
            normalized_text = text.lower().strip()
            
            # Calculate polarity (-1 to 1)
            polarity = self._calculate_polarity(normalized_text)
            
            # Calculate subjectivity (0 to 1)
            subjectivity = self._calculate_subjectivity(normalized_text)
            
            # Detect primary emotion
            emotion = self._detect_emotion(normalized_text)
            
            # Calculate confidence based on keyword matches and text length
            confidence = self._calculate_confidence(normalized_text, emotion)
            
            sentiment_score = SentimentScore(
                polarity=polarity,
                subjectivity=subjectivity,
                emotion=emotion.value.lower(),
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            logger.info(
                f"Sentiment analysis completed - Emotion: {emotion.value}, "
                f"Polarity: {polarity:.2f}, Confidence: {confidence:.2f}"
            )
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise SentimentAnalysisError(f"Sentiment analysis failed: {str(e)}")
    
    def _calculate_polarity(self, text: str) -> float:
        """Calculate sentiment polarity (-1 negative to +1 positive)."""
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "like", "happy", "pleased", "satisfied", "perfect",
            "awesome", "brilliant", "outstanding", "superb", "yes", "sure"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "horrible", "hate", "dislike",
            "angry", "frustrated", "disappointed", "upset", "no", "never",
            "worst", "disgusting", "annoying", "irritating", "stupid"
        ]
        
        words = text.split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0  # Neutral
        
        polarity = (positive_count - negative_count) / total_sentiment_words
        return max(-1.0, min(1.0, polarity))
    
    def _calculate_subjectivity(self, text: str) -> float:
        """Calculate subjectivity (0 objective to 1 subjective)."""
        subjective_indicators = [
            "i think", "i feel", "i believe", "in my opinion", "personally",
            "i love", "i hate", "i like", "i dislike", "amazing", "terrible",
            "wonderful", "awful", "fantastic", "horrible", "perfect", "worst"
        ]
        
        subjective_count = sum(1 for indicator in subjective_indicators if indicator in text)
        
        # Normalize by text length (approximate)
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        subjectivity = min(1.0, subjective_count / (word_count / 10))
        return subjectivity
    
    def _detect_emotion(self, text: str) -> EmotionalState:
        """Detect primary emotion from text."""
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            emotion_scores[emotion] = score
        
        # Find emotion with highest score
        if not any(emotion_scores.values()):
            return EmotionalState.NEUTRAL
        
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        return primary_emotion
    
    def _calculate_confidence(self, text: str, emotion: EmotionalState) -> float:
        """Calculate confidence in sentiment analysis."""
        # Base confidence on keyword matches
        if emotion == EmotionalState.NEUTRAL:
            return 0.6  # Moderate confidence for neutral
        
        keywords = self.emotion_keywords.get(emotion, [])
        matches = sum(1 for keyword in keywords if keyword in text)
        
        # Confidence based on number of matches and text length
        word_count = len(text.split())
        if word_count == 0:
            return 0.1
        
        match_ratio = matches / word_count
        confidence = min(0.95, 0.5 + (match_ratio * 2))
        
        return confidence

class EmpathyEngine:
    """
    Empathetic response generation engine.
    
    This engine generates contextually appropriate responses based on
    user sentiment and conversation stage to maximize engagement and
    conversion rates through emotional intelligence.
    """
    
    def __init__(self):
        """Initialize empathy engine."""
        self.response_templates = self._load_response_templates()
        logger.info("EmpathyEngine initialized")
    
    def _load_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load empathetic response templates organized by emotion and stage."""
        return {
            EmotionalState.EXCITED.value: {
                ConversationStage.INITIAL_INQUIRY.value: [
                    "I can feel your excitement! Let's find you the perfect loan to make your dreams come true.",
                    "Your enthusiasm is wonderful! I'm here to help you get the best loan terms possible.",
                    "I love your positive energy! Let's explore loan options that match your excitement."
                ],
                ConversationStage.LOAN_PRESENTATION.value: [
                    "Given your enthusiasm, I think you'll love these loan options we've tailored for you!",
                    "Your excitement tells me you're ready to move forward - here are some great options!",
                    "I can see you're eager to proceed - these loan terms should be perfect for you!"
                ]
            },
            EmotionalState.CONCERNED.value: {
                ConversationStage.INITIAL_INQUIRY.value: [
                    "I understand your concerns, and that's completely normal. Let me address them one by one.",
                    "Your caution shows you're making a thoughtful decision. I'm here to help clarify everything.",
                    "It's wise to have questions. Let me provide all the information you need to feel confident."
                ],
                ConversationStage.OBJECTION_HANDLING.value: [
                    "I hear your concerns, and they're valid. Let me show you how we can address each one.",
                    "Your worries are understandable. Many customers had similar concerns, and here's how we resolved them.",
                    "I appreciate your honesty about your concerns. Let's work together to find solutions."
                ]
            },
            EmotionalState.FRUSTRATED.value: {
                ConversationStage.INITIAL_INQUIRY.value: [
                    "I can sense your frustration, and I'm sorry you've had difficulties. Let me make this process smooth for you.",
                    "I understand this has been frustrating. I'm here to turn this experience around for you.",
                    "Your frustration is completely understandable. Let me help make this much easier."
                ],
                ConversationStage.REJECTION_RECOVERY.value: [
                    "I know this rejection is disappointing, but I have some alternative options that might work better for you.",
                    "I understand your frustration with the rejection. Let me show you some other paths to get you approved.",
                    "Don't give up! I have several alternative loan products that could be perfect for your situation."
                ]
            },
            EmotionalState.ANXIOUS.value: {
                ConversationStage.INITIAL_INQUIRY.value: [
                    "I can see this is making you anxious, and that's okay. Let's take this step by step, at your pace.",
                    "I understand this feels overwhelming. I'm here to guide you through everything calmly and clearly.",
                    "Your anxiety is natural - this is a big decision. Let me help you feel more comfortable with the process."
                ],
                ConversationStage.LOAN_PRESENTATION.value: [
                    "I know this might feel overwhelming, so let me explain each option clearly and simply.",
                    "Take your time to review these options. I'm here to answer any questions that might ease your concerns.",
                    "I understand this is a lot to process. Let's go through each loan option slowly and thoroughly."
                ]
            },
            EmotionalState.NEUTRAL.value: {
                ConversationStage.INITIAL_INQUIRY.value: [
                    "Thank you for your interest in our loan products. I'm here to help you find the right solution.",
                    "I appreciate you considering us for your loan needs. Let's explore what options work best for you.",
                    "Welcome! I'm excited to help you find a loan that perfectly fits your requirements."
                ],
                ConversationStage.LOAN_PRESENTATION.value: [
                    "Based on your profile, here are the loan options I recommend for your consideration.",
                    "I've prepared several loan options that align with your needs and financial profile.",
                    "Here are the loan products that best match your requirements and eligibility."
                ]
            }
        }
    
    def generate_empathetic_response(
        self,
        sentiment: SentimentScore,
        conversation_stage: ConversationStage,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate empathetic response based on sentiment and conversation context.
        
        Args:
            sentiment: User sentiment analysis result
            conversation_stage: Current stage of conversation
            context: Additional context for response generation
            
        Returns:
            str: Empathetic response tailored to user's emotional state
        """
        logger.info(
            f"Generating empathetic response - Emotion: {sentiment.emotion}, "
            f"Stage: {conversation_stage.value}"
        )
        
        # Get emotion-specific templates
        emotion_key = sentiment.emotion.upper()
        if emotion_key not in self.response_templates:
            emotion_key = EmotionalState.NEUTRAL.value
        
        emotion_templates = self.response_templates[emotion_key]
        
        # Get stage-specific templates
        stage_key = conversation_stage.value
        if stage_key not in emotion_templates:
            # Fallback to initial inquiry templates
            stage_key = ConversationStage.INITIAL_INQUIRY.value
        
        templates = emotion_templates.get(stage_key, emotion_templates[ConversationStage.INITIAL_INQUIRY.value])
        
        # Select template based on sentiment polarity
        if sentiment.polarity > 0.3:
            # Positive sentiment - use more enthusiastic templates
            template_index = 0
        elif sentiment.polarity < -0.3:
            # Negative sentiment - use more supportive templates
            template_index = min(len(templates) - 1, 2)
        else:
            # Neutral sentiment - use balanced templates
            template_index = 1 if len(templates) > 1 else 0
        
        base_response = templates[template_index]
        
        # Enhance response with context if available
        if context:
            enhanced_response = self._enhance_response_with_context(base_response, context)
            return enhanced_response
        
        return base_response
    
    def _enhance_response_with_context(self, base_response: str, context: Dict[str, Any]) -> str:
        """Enhance response with contextual information."""
        
        # Add personalization if user name is available
        if context.get("user_name"):
            base_response = f"{context['user_name']}, {base_response.lower()}"
        
        # Add specific loan amount reference if available
        if context.get("loan_amount"):
            amount_display = f"₹{context['loan_amount']:,.2f}"
            base_response += f" For your {amount_display} loan requirement, "
        
        # Add urgency if time-sensitive
        if context.get("urgent"):
            base_response += " I understand this is time-sensitive, so let's move quickly."
        
        return base_response

class PlanBEngine:
    """
    Plan B logic engine for loan rejection recovery.
    
    This engine generates alternative loan offers when primary applications
    are rejected, maximizing conversion rates by providing suitable alternatives
    rather than losing customers entirely.
    """
    
    def __init__(self):
        """Initialize Plan B engine."""
        self.alternative_products = self._load_alternative_products()
        logger.info("PlanBEngine initialized")
    
    def _load_alternative_products(self) -> List[Dict[str, Any]]:
        """Load alternative loan product configurations."""
        return [
            {
                "product_name": "Secured Personal Loan",
                "max_amount_multiplier": 0.7,  # 70% of original request
                "rate_adjustment": Decimal("2.0"),  # +2% interest rate
                "tenure_adjustment": 6,  # +6 months tenure
                "special_features": [
                    "Lower interest with collateral",
                    "Flexible repayment options",
                    "Quick approval process"
                ],
                "eligibility_criteria": [
                    "Collateral required (property/FD)",
                    "Minimum income: ₹25,000",
                    "CIBIL score: 600+"
                ]
            },
            {
                "product_name": "Co-applicant Personal Loan",
                "max_amount_multiplier": 0.8,  # 80% of original request
                "rate_adjustment": Decimal("1.5"),  # +1.5% interest rate
                "tenure_adjustment": 0,  # Same tenure
                "special_features": [
                    "Higher loan amount with co-applicant",
                    "Shared responsibility",
                    "Better interest rates"
                ],
                "eligibility_criteria": [
                    "Co-applicant with good credit",
                    "Combined income consideration",
                    "Joint liability"
                ]
            },
            {
                "product_name": "Step-up EMI Loan",
                "max_amount_multiplier": 0.6,  # 60% of original request
                "rate_adjustment": Decimal("1.0"),  # +1% interest rate
                "tenure_adjustment": 12,  # +12 months tenure
                "special_features": [
                    "Lower initial EMIs",
                    "EMI increases annually",
                    "Suitable for growing income"
                ],
                "eligibility_criteria": [
                    "Stable employment",
                    "Expected income growth",
                    "CIBIL score: 650+"
                ]
            },
            {
                "product_name": "Micro Personal Loan",
                "max_amount_multiplier": 0.3,  # 30% of original request
                "rate_adjustment": Decimal("3.0"),  # +3% interest rate
                "tenure_adjustment": -6,  # -6 months tenure
                "special_features": [
                    "Quick approval",
                    "Minimal documentation",
                    "Build credit history"
                ],
                "eligibility_criteria": [
                    "Basic income proof",
                    "Any CIBIL score",
                    "Minimum ₹15,000 income"
                ]
            }
        ]
    
    def generate_plan_b_offers(
        self,
        original_request: LoanRequest,
        user_profile: UserProfile,
        rejection_reason: str,
        max_offers: int = 3
    ) -> List[AlternativeOffer]:
        """
        Generate Plan B alternative loan offers based on rejection analysis.
        
        Args:
            original_request: Original loan request that was rejected
            user_profile: User profile information
            rejection_reason: Reason for original loan rejection
            max_offers: Maximum number of alternative offers to generate
            
        Returns:
            List[AlternativeOffer]: List of alternative loan offers
            
        Raises:
            PlanBGenerationError: If Plan B generation fails
        """
        logger.info(
            f"Generating Plan B offers - Original Amount: {original_request.amount_in_cents}, "
            f"Rejection: {rejection_reason}"
        )
        
        try:
            alternative_offers = []
            
            # Analyze rejection reason to determine suitable alternatives
            suitable_products = self._select_suitable_products(rejection_reason, user_profile)
            
            for i, product in enumerate(suitable_products[:max_offers]):
                offer = self._create_alternative_offer(
                    product, original_request, user_profile, i + 1
                )
                alternative_offers.append(offer)
            
            # Sort offers by approval probability (highest first)
            alternative_offers.sort(key=lambda x: x.approval_probability, reverse=True)
            
            logger.info(f"Generated {len(alternative_offers)} Plan B offers")
            return alternative_offers
            
        except Exception as e:
            logger.error(f"Plan B generation failed: {str(e)}")
            raise PlanBGenerationError(f"Plan B generation failed: {str(e)}")
    
    def _select_suitable_products(
        self,
        rejection_reason: str,
        user_profile: UserProfile
    ) -> List[Dict[str, Any]]:
        """Select suitable alternative products based on rejection reason."""
        
        suitable_products = []
        rejection_lower = rejection_reason.lower()
        
        # Income-related rejections
        if any(keyword in rejection_lower for keyword in ["income", "affordability", "emi"]):
            suitable_products.extend([
                self.alternative_products[2],  # Step-up EMI
                self.alternative_products[3],  # Micro loan
                self.alternative_products[1]   # Co-applicant
            ])
        
        # Credit score related rejections
        elif any(keyword in rejection_lower for keyword in ["credit", "cibil", "score"]):
            suitable_products.extend([
                self.alternative_products[0],  # Secured loan
                self.alternative_products[1],  # Co-applicant
                self.alternative_products[3]   # Micro loan
            ])
        
        # Amount-related rejections
        elif any(keyword in rejection_lower for keyword in ["amount", "limit", "maximum"]):
            suitable_products.extend([
                self.alternative_products[0],  # Secured loan
                self.alternative_products[1],  # Co-applicant
                self.alternative_products[2]   # Step-up EMI
            ])
        
        # Default: offer all products
        else:
            suitable_products = self.alternative_products.copy()
        
        return suitable_products
    
    def _create_alternative_offer(
        self,
        product_config: Dict[str, Any],
        original_request: LoanRequest,
        user_profile: UserProfile,
        offer_number: int
    ) -> AlternativeOffer:
        """Create alternative offer from product configuration."""
        
        # Calculate adjusted loan amount (LQM Standard: Decimal arithmetic)
        original_amount = validate_monetary_input(original_request.amount_in_cents, "original_amount")
        adjusted_amount = original_amount * Decimal(str(product_config["max_amount_multiplier"]))
        
        # Calculate adjusted tenure
        original_tenure = original_request.tenure_months
        adjusted_tenure = original_tenure + product_config["tenure_adjustment"]
        adjusted_tenure = max(6, min(360, adjusted_tenure))  # Keep within reasonable bounds
        
        # Calculate adjusted interest rate
        base_rate = Decimal("12.00")  # Default base rate
        if original_request.requested_rate:
            base_rate = validate_monetary_input(original_request.requested_rate, "base_rate")
        
        adjusted_rate = base_rate + product_config["rate_adjustment"]
        
        # Calculate EMI using LQM
        emi_calculation = calculate_emi(adjusted_amount, adjusted_rate, adjusted_tenure)
        
        # Calculate approval probability based on user profile
        approval_probability = self._calculate_approval_probability(
            product_config, user_profile, adjusted_amount
        )
        
        # Generate offer ID
        offer_id = f"PLB_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{offer_number}"
        
        # Offer valid for 7 days
        valid_until = datetime.now().replace(hour=23, minute=59, second=59) + timedelta(days=7)
        
        return AlternativeOffer(
            offer_id=offer_id,
            product_name=product_config["product_name"],
            amount_in_cents=adjusted_amount,
            tenure_months=adjusted_tenure,
            interest_rate=adjusted_rate,
            emi_in_cents=emi_calculation.emi_in_cents,
            special_features=product_config["special_features"],
            eligibility_criteria=product_config["eligibility_criteria"],
            approval_probability=approval_probability,
            offer_valid_until=valid_until
        )
    
    def _calculate_approval_probability(
        self,
        product_config: Dict[str, Any],
        user_profile: UserProfile,
        loan_amount: Decimal
    ) -> float:
        """Calculate approval probability for alternative offer."""
        
        base_probability = 0.7  # Base 70% probability for alternatives
        
        # Adjust based on user income (LQM Standard: Decimal arithmetic)
        monthly_income = validate_monetary_input(user_profile.income_in_cents, "monthly_income")
        annual_income = monthly_income * Decimal('12')
        loan_to_income_ratio = loan_amount / annual_income
        
        if loan_to_income_ratio <= Decimal('2.0'):
            base_probability += 0.2  # +20% for conservative loan amount
        elif loan_to_income_ratio > Decimal('4.0'):
            base_probability -= 0.2  # -20% for high loan amount
        
        # Adjust based on credit score
        if user_profile.credit_score:
            if user_profile.credit_score >= 750:
                base_probability += 0.15
            elif user_profile.credit_score >= 650:
                base_probability += 0.05
            elif user_profile.credit_score < 550:
                base_probability -= 0.15
        
        # Adjust based on employment type
        if user_profile.employment_type.value == "SALARIED":
            base_probability += 0.1
        elif user_profile.employment_type.value in ["SELF_EMPLOYED", "BUSINESS_OWNER"]:
            base_probability -= 0.05
        
        # Ensure probability is within valid range
        return max(0.1, min(0.95, base_probability))

class SalesAgent:
    """
    Sales Agent - The Negotiator of Loan2Day.
    
    This agent handles empathetic customer communication using real-time
    sentiment analysis and implements Plan B logic for loan rejection recovery.
    It maximizes conversion rates through emotional intelligence and adaptive
    response generation.
    """
    
    def __init__(self):
        """Initialize the Sales Agent."""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.empathy_engine = EmpathyEngine()
        self.plan_b_engine = PlanBEngine()
        logger.info("SalesAgent initialized successfully")
    
    async def analyze_user_sentiment(
        self,
        user_input: str,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> SentimentScore:
        """
        Analyze user sentiment for empathetic response generation.
        
        Args:
            user_input: User's message text
            conversation_context: Optional conversation context
            
        Returns:
            SentimentScore: Comprehensive sentiment analysis result
        """
        logger.info("Analyzing user sentiment for empathetic response")
        
        try:
            sentiment_score = self.sentiment_analyzer.analyze_sentiment(
                user_input, conversation_context
            )
            
            logger.info(
                f"Sentiment analysis completed - Emotion: {sentiment_score.emotion}, "
                f"Polarity: {sentiment_score.polarity:.2f}"
            )
            
            return sentiment_score
            
        except SentimentAnalysisError as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            # Return neutral sentiment as fallback
            return SentimentScore(
                polarity=0.0,
                subjectivity=0.5,
                emotion=EmotionalState.NEUTRAL.value.lower(),
                confidence=0.5,
                timestamp=datetime.now()
            )
    
    async def generate_empathetic_response(
        self,
        sentiment: SentimentScore,
        conversation_stage: ConversationStage,
        agent_state: AgentState,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate empathetic response based on user sentiment and conversation context.
        
        Args:
            sentiment: User sentiment analysis result
            conversation_stage: Current conversation stage
            agent_state: Current agent state for context
            additional_context: Additional context for response generation
            
        Returns:
            str: Empathetic response tailored to user's emotional state
        """
        logger.info(f"Generating empathetic response for {sentiment.emotion} user")
        
        # Prepare context for response generation
        context = additional_context or {}
        
        # Add user profile context if available
        if agent_state.user_profile:
            context["user_name"] = agent_state.user_profile.name.split()[0]  # First name
        
        # Add loan request context if available
        if agent_state.loan_request:
            loan_amount_rupees = float(agent_state.loan_request.amount_in_cents) / 100
            context["loan_amount"] = loan_amount_rupees
        
        # Generate empathetic response
        empathetic_response = self.empathy_engine.generate_empathetic_response(
            sentiment, conversation_stage, context
        )
        
        logger.info("Empathetic response generated successfully")
        return empathetic_response
    
    async def trigger_plan_b_logic(
        self,
        agent_state: AgentState,
        rejection_reason: str,
        user_sentiment: Optional[SentimentScore] = None
    ) -> Dict[str, Any]:
        """
        Trigger Plan B logic for loan rejection recovery.
        
        This method generates alternative loan offers and empathetic messaging
        to recover from loan rejections and maximize conversion rates.
        
        Args:
            agent_state: Current agent state with user and loan information
            rejection_reason: Reason for loan rejection
            user_sentiment: Optional user sentiment for response customization
            
        Returns:
            Dict[str, Any]: Plan B result with alternative offers and messaging
            
        Raises:
            PlanBGenerationError: If Plan B generation fails
        """
        logger.info(f"Triggering Plan B logic for rejection: {rejection_reason}")
        
        try:
            # Validate required data
            if not agent_state.user_profile or not agent_state.loan_request:
                raise PlanBGenerationError("User profile and loan request required for Plan B")
            
            # Generate alternative offers
            alternative_offers = self.plan_b_engine.generate_plan_b_offers(
                agent_state.loan_request,
                agent_state.user_profile,
                rejection_reason,
                max_offers=3
            )
            
            # Generate empathetic messaging for rejection recovery
            conversation_stage = ConversationStage.REJECTION_RECOVERY
            
            if user_sentiment:
                empathetic_message = await self.generate_empathetic_response(
                    user_sentiment, conversation_stage, agent_state
                )
            else:
                # Default empathetic message for rejection
                empathetic_message = (
                    "I understand this rejection is disappointing, but I have some "
                    "excellent alternative options that could work perfectly for you."
                )
            
            # Prepare Plan B result
            plan_b_result = {
                "triggered": True,
                "rejection_reason": rejection_reason,
                "empathetic_message": empathetic_message,
                "alternative_offers": [offer.to_dict() for offer in alternative_offers],
                "total_alternatives": len(alternative_offers),
                "best_offer": alternative_offers[0].to_dict() if alternative_offers else None,
                "recovery_strategy": self._determine_recovery_strategy(rejection_reason),
                "next_steps": self._generate_next_steps(alternative_offers),
                "plan_b_timestamp": datetime.now().isoformat()
            }
            
            logger.info(
                f"Plan B logic completed - Generated {len(alternative_offers)} alternatives, "
                f"Best approval probability: {alternative_offers[0].approval_probability:.1%}"
            )
            
            return plan_b_result
            
        except Exception as e:
            logger.error(f"Plan B logic failed: {str(e)}")
            raise PlanBGenerationError(f"Plan B logic failed: {str(e)}")
    
    def _determine_recovery_strategy(self, rejection_reason: str) -> str:
        """Determine recovery strategy based on rejection reason."""
        rejection_lower = rejection_reason.lower()
        
        if "income" in rejection_lower or "affordability" in rejection_lower:
            return "INCOME_OPTIMIZATION"
        elif "credit" in rejection_lower or "cibil" in rejection_lower:
            return "CREDIT_ENHANCEMENT"
        elif "amount" in rejection_lower:
            return "AMOUNT_ADJUSTMENT"
        elif "documentation" in rejection_lower:
            return "DOCUMENTATION_SUPPORT"
        else:
            return "GENERAL_ALTERNATIVES"
    
    def _generate_next_steps(self, alternative_offers: List[AlternativeOffer]) -> List[str]:
        """Generate next steps for Plan B recovery."""
        if not alternative_offers:
            return ["Contact our loan specialist for personalized assistance"]
        
        next_steps = [
            f"Review the {len(alternative_offers)} alternative loan options presented",
            "Choose the option that best fits your needs and budget",
            "Provide any additional documentation if required"
        ]
        
        # Add specific steps based on offer types
        offer_types = [offer.product_name for offer in alternative_offers]
        
        if any("Secured" in offer_type for offer_type in offer_types):
            next_steps.append("Prepare collateral documentation for secured loan options")
        
        if any("Co-applicant" in offer_type for offer_type in offer_types):
            next_steps.append("Identify a suitable co-applicant with good credit history")
        
        next_steps.append("Schedule a call with our loan advisor for detailed discussion")
        
        return next_steps
    
    async def process_sales_interaction(
        self,
        agent_state: AgentState,
        user_input: str,
        interaction_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Process complete sales interaction with sentiment analysis and empathetic response.
        
        This is the main entry point for sales processing in the Master-Worker
        agent pattern. It analyzes sentiment, generates empathetic responses,
        and updates the agent state with interaction history.
        
        Args:
            agent_state: Current agent state
            user_input: User's message or input
            interaction_type: Type of interaction (general, objection, negotiation)
            
        Returns:
            Dict[str, Any]: Sales interaction result with response and updated state
        """
        logger.info(f"Processing sales interaction - Type: {interaction_type}")
        
        try:
            # Analyze user sentiment
            sentiment_score = await self.analyze_user_sentiment(
                user_input, agent_state.conversation_context
            )
            
            # Determine conversation stage based on agent state and interaction type
            conversation_stage = self._determine_conversation_stage(agent_state, interaction_type)
            
            # Generate empathetic response
            empathetic_response = await self.generate_empathetic_response(
                sentiment_score, conversation_stage, agent_state
            )
            
            # Prepare sales interaction result
            sales_result = {
                "sentiment_analysis": sentiment_score.to_dict() if hasattr(sentiment_score, 'to_dict') else {
                    "polarity": sentiment_score.polarity,
                    "subjectivity": sentiment_score.subjectivity,
                    "emotion": sentiment_score.emotion,
                    "confidence": sentiment_score.confidence,
                    "timestamp": sentiment_score.timestamp.isoformat()
                },
                "empathetic_response": empathetic_response,
                "conversation_stage": conversation_stage.value,
                "interaction_type": interaction_type,
                "engagement_score": self._calculate_engagement_score(sentiment_score),
                "conversion_indicators": self._identify_conversion_indicators(user_input, sentiment_score),
                "recommended_actions": self._recommend_next_actions(sentiment_score, conversation_stage),
                "processing_timestamp": datetime.now().isoformat()
            }
            
            logger.info(
                f"Sales interaction processed - Emotion: {sentiment_score.emotion}, "
                f"Stage: {conversation_stage.value}, Engagement: {sales_result['engagement_score']:.2f}"
            )
            
            return sales_result
            
        except Exception as e:
            logger.error(f"Sales interaction processing failed: {str(e)}")
            raise SalesError(f"Sales interaction processing failed: {str(e)}")
    
    def _determine_conversation_stage(self, agent_state: AgentState, interaction_type: str) -> ConversationStage:
        """Determine current conversation stage based on agent state."""
        
        # Check agent state progression
        if agent_state.current_step.value == "GREETING":
            return ConversationStage.INITIAL_INQUIRY
        elif agent_state.current_step.value == "KYC":
            return ConversationStage.REQUIREMENT_GATHERING
        elif agent_state.current_step.value == "NEGOTIATION":
            if interaction_type == "objection":
                return ConversationStage.OBJECTION_HANDLING
            else:
                return ConversationStage.NEGOTIATION
        elif agent_state.current_step.value == "SANCTION":
            return ConversationStage.CLOSING
        elif agent_state.current_step.value == "PLAN_B":
            return ConversationStage.REJECTION_RECOVERY
        else:
            return ConversationStage.INITIAL_INQUIRY
    
    def _calculate_engagement_score(self, sentiment: SentimentScore) -> float:
        """Calculate user engagement score based on sentiment."""
        
        # Base engagement from sentiment polarity and confidence
        base_engagement = (sentiment.polarity + 1) / 2  # Normalize to 0-1
        confidence_boost = sentiment.confidence * 0.3
        
        # Adjust based on emotion
        emotion_adjustments = {
            "excited": 0.3,
            "optimistic": 0.2,
            "neutral": 0.0,
            "concerned": -0.1,
            "frustrated": -0.2,
            "anxious": -0.15,
            "angry": -0.3
        }
        
        emotion_adjustment = emotion_adjustments.get(sentiment.emotion, 0.0)
        
        engagement_score = base_engagement + confidence_boost + emotion_adjustment
        return max(0.0, min(1.0, engagement_score))
    
    def _identify_conversion_indicators(self, user_input: str, sentiment: SentimentScore) -> List[str]:
        """Identify indicators of conversion readiness."""
        indicators = []
        user_input_lower = user_input.lower()
        
        # Positive conversion indicators
        positive_phrases = [
            "let's proceed", "i'm interested", "sounds good", "when can we start",
            "what's next", "i want to apply", "let's do it", "i'm ready"
        ]
        
        if any(phrase in user_input_lower for phrase in positive_phrases):
            indicators.append("READY_TO_PROCEED")
        
        # Question indicators (engagement)
        if "?" in user_input or any(word in user_input_lower for word in ["what", "how", "when", "where", "why"]):
            indicators.append("SEEKING_INFORMATION")
        
        # Sentiment-based indicators
        if sentiment.emotion in ["excited", "optimistic"] and sentiment.polarity > 0.3:
            indicators.append("POSITIVE_SENTIMENT")
        
        if sentiment.emotion in ["concerned", "anxious"] and "but" in user_input_lower:
            indicators.append("CONDITIONAL_INTEREST")
        
        # Objection indicators
        objection_phrases = ["too expensive", "too high", "can't afford", "not sure", "need to think"]
        if any(phrase in user_input_lower for phrase in objection_phrases):
            indicators.append("PRICE_OBJECTION")
        
        return indicators
    
    def activate_plan_b_logic(
        self,
        agent_state: AgentState,
        rejection_reason: str,
        original_loan_amount: Decimal
    ) -> Dict[str, Any]:
        """
        Activate Plan B logic for loan rejection recovery.
        
        This method is called when a loan application is rejected to provide
        alternative options and empathetic recovery messaging.
        
        Args:
            agent_state: Current agent state with user information
            rejection_reason: Specific reason for loan rejection
            original_loan_amount: Original requested loan amount
            
        Returns:
            Dict[str, Any]: Plan B result with alternatives and messaging
        """
        logger.info(f"Activating Plan B logic - Rejection: {rejection_reason}")
        
        try:
            # Generate alternative offers using the Plan B engine
            alternative_offers = self.generate_alternative_offers(
                original_loan_amount=original_loan_amount,
                monthly_income_in_cents=agent_state.user_profile.income_in_cents,
                credit_score=getattr(agent_state.user_profile, 'credit_score', 650),
                max_offers=3
            )
            
            # Generate empathetic message based on rejection reason
            empathetic_message = self._generate_rejection_empathy_message(rejection_reason)
            
            # Prepare Plan B result
            plan_b_result = {
                "alternative_offers": alternative_offers,
                "empathetic_message": empathetic_message,
                "rejection_reason": rejection_reason,
                "recovery_strategy": self._determine_recovery_strategy(rejection_reason),
                "next_steps": self._generate_next_steps_for_alternatives(alternative_offers),
                "plan_b_activated": True,
                "activation_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Plan B activated successfully with {len(alternative_offers)} alternatives")
            return plan_b_result
            
        except Exception as e:
            logger.error(f"Plan B activation failed: {str(e)}")
            return {
                "alternative_offers": [],
                "empathetic_message": "I understand this is disappointing. Let me connect you with our specialist for personalized assistance.",
                "rejection_reason": rejection_reason,
                "error": str(e),
                "plan_b_activated": False
            }
    
    def generate_alternative_offers(
        self,
        original_loan_amount: Decimal,
        monthly_income_in_cents: Decimal,
        credit_score: Optional[int],
        max_offers: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate alternative loan offers based on user profile.
        
        Args:
            original_loan_amount: Original requested amount in cents
            monthly_income_in_cents: User's monthly income in cents
            credit_score: User's credit score
            max_offers: Maximum number of offers to generate
            
        Returns:
            List[Dict[str, Any]]: List of alternative loan offers
        """
        logger.info("Generating alternative loan offers")
        
        offers = []
        
        # Calculate maximum affordable EMI (40% of monthly income)
        max_affordable_emi = monthly_income_in_cents * Decimal("0.40")
        
        # Generate progressive alternatives
        for i in range(max_offers):
            # Progressive reduction in loan amount
            reduction_factor = Decimal(str(0.8 - (i * 0.15)))  # 80%, 65%, 50%
            alternative_amount = original_loan_amount * reduction_factor
            
            # Progressive increase in tenure for affordability
            base_tenure = 60  # 5 years
            alternative_tenure = base_tenure + (i * 12)  # +1 year per alternative
            
            # Adjust interest rate based on credit score and alternative level
            base_rate = self._calculate_interest_rate(credit_score)
            rate_adjustment = Decimal(str(i * 0.5))  # +0.5% per alternative
            alternative_rate = base_rate + rate_adjustment
            
            # Calculate EMI using LQM
            try:
                emi_calc = calculate_emi(
                    principal=alternative_amount,
                    annual_rate=alternative_rate,
                    tenure_months=alternative_tenure
                )
                
                # Only include if EMI is affordable
                if emi_calc.emi_in_cents <= max_affordable_emi:
                    offer = {
                        "loan_amount_in_cents": alternative_amount,
                        "tenure_months": alternative_tenure,
                        "interest_rate_per_annum": alternative_rate,
                        "emi_in_cents": emi_calc.emi_in_cents,
                        "total_interest_in_cents": emi_calc.total_interest_in_cents,
                        "total_amount_in_cents": emi_calc.total_amount_in_cents,
                        "offer_type": f"Alternative {i + 1}",
                        "approval_probability": self._calculate_approval_probability_simple(
                            credit_score, alternative_amount, monthly_income_in_cents
                        )
                    }
                    offers.append(offer)
                
            except Exception as e:
                logger.warning(f"Failed to calculate EMI for alternative {i + 1}: {e}")
                continue
        
        logger.info(f"Generated {len(offers)} alternative offers")
        return offers
    
    def _calculate_interest_rate(self, credit_score: Optional[int]) -> Decimal:
        """Calculate interest rate based on credit score."""
        if credit_score is None:
            return Decimal("14.00")  # Default rate for unknown credit
        elif credit_score >= 750:
            return Decimal("10.50")  # Excellent credit
        elif credit_score >= 650:
            return Decimal("12.00")  # Good credit
        elif credit_score >= 550:
            return Decimal("14.50")  # Fair credit
        else:
            return Decimal("17.00")  # Poor credit
    
    def _calculate_approval_probability_simple(
        self,
        credit_score: Optional[int],
        loan_amount: Decimal,
        monthly_income: Decimal
    ) -> float:
        """Calculate simple approval probability for alternatives."""
        base_probability = 0.7
        
        # Credit score adjustment (handle None case)
        if credit_score is not None:
            if credit_score >= 750:
                base_probability += 0.2
            elif credit_score >= 650:
                base_probability += 0.1
            elif credit_score < 550:
                base_probability -= 0.2
        # If credit_score is None, use base probability
        
        # Income to loan ratio adjustment
        annual_income = monthly_income * 12
        if loan_amount <= annual_income * Decimal("2"):
            base_probability += 0.1
        elif loan_amount > annual_income * Decimal("4"):
            base_probability -= 0.1
        
        return max(0.1, min(0.95, base_probability))
    
    def _generate_rejection_empathy_message(self, rejection_reason: str) -> str:
        """Generate empathetic message for rejection scenarios."""
        rejection_lower = rejection_reason.lower()
        
        if "credit" in rejection_lower or "score" in rejection_lower:
            return "I understand that credit score requirements can be challenging. The good news is that I have some alternative options that work with your current credit profile."
        elif "income" in rejection_lower:
            return "I know it's frustrating when income requirements don't align perfectly. Let me show you some options that are designed to work with your current income level."
        elif "amount" in rejection_lower:
            return "I understand you were hoping for the full amount. While we can't approve the original request, I have some excellent alternatives that can still meet your needs."
        else:
            return "I understand this rejection is disappointing, but I have some excellent alternative options that could work perfectly for your situation."
    
    def _generate_next_steps_for_alternatives(self, alternatives: List[Dict[str, Any]]) -> List[str]:
        """Generate next steps for alternative offers."""
        if not alternatives:
            return ["Contact our loan specialist for personalized assistance"]
        
        return [
            f"Review the {len(alternatives)} alternative loan options",
            "Choose the option that best fits your budget",
            "Provide any additional documentation if required",
            "Schedule a call with our loan advisor for detailed discussion"
        ]
    
    def process_tanglish_plan_b_request(
        self,
        user_input: str,
        agent_state: AgentState,
        rejection_context: str
    ) -> Dict[str, Any]:
        """
        Process Tanglish (mixed language) Plan B requests.
        
        Args:
            user_input: User input in Tanglish
            agent_state: Current agent state
            rejection_context: Context of the rejection
            
        Returns:
            Dict[str, Any]: Response with alternatives or clarification
        """
        logger.info("Processing Tanglish Plan B request")
        
        # Simple Tanglish keyword detection
        tanglish_keywords = {
            "venum": "want",
            "kammi": "less/reduce",
            "jaasthi": "too much/high",
            "pannunga": "please do",
            "mudiyuma": "can you",
            "irukka": "is there"
        }
        
        # Translate key terms
        translated_input = user_input.lower()
        for tamil_word, english_word in tanglish_keywords.items():
            if tamil_word in translated_input:
                translated_input = translated_input.replace(tamil_word, english_word)
        
        # Determine user intent
        if any(word in translated_input for word in ["reduce", "less", "kammi"]):
            intent = "REDUCE_AMOUNT_OR_EMI"
        elif any(word in translated_input for word in ["want", "venum"]):
            intent = "WANT_LOAN"
        elif any(word in translated_input for word in ["can you", "mudiyuma"]):
            intent = "REQUEST_MODIFICATION"
        else:
            intent = "GENERAL_INQUIRY"
        
        # Generate appropriate response
        if intent == "REDUCE_AMOUNT_OR_EMI":
            alternatives = self.generate_alternative_offers(
                original_loan_amount=agent_state.loan_request.amount_in_cents,
                monthly_income_in_cents=agent_state.user_profile.income_in_cents,
                credit_score=getattr(agent_state.user_profile, 'credit_score', 650),
                max_offers=2
            )
            
            return {
                "alternative_offers": alternatives,
                "response_message": "I understand you need a smaller amount or EMI. Here are some options that might work better for you.",
                "intent_detected": intent,
                "original_input": user_input
            }
        
        else:
            return {
                "response_message": "I understand you're interested in loan options. Let me help you find something that works for your needs.",
                "clarification": "Could you please tell me more about what specific changes you'd like to see?",
                "intent_detected": intent,
                "original_input": user_input
            }
    
    def handle_plan_b_negotiation(
        self,
        user_input: str,
        agent_state: AgentState,
        negotiation_round: int
    ) -> Dict[str, Any]:
        """
        Handle Plan B negotiation flow.
        
        Args:
            user_input: User's negotiation input
            agent_state: Current agent state
            negotiation_round: Round number of negotiation
            
        Returns:
            Dict[str, Any]: Negotiation response with action
        """
        logger.info(f"Handling Plan B negotiation - Round {negotiation_round}")
        
        user_input_lower = user_input.lower()
        
        # Determine negotiation action
        if any(word in user_input_lower for word in ["emi", "monthly", "payment"]) and any(word in user_input_lower for word in ["max", "maximum"]):
            action = "CALCULATE_ALTERNATIVES"
        elif any(word in user_input_lower for word in ["interest", "rate"]) and any(word in user_input_lower for word in ["reduce", "lower", "kammi"]):
            action = "ADJUST_INTEREST"
        elif any(word in user_input_lower for word in ["okay", "final", "documents", "submit"]):
            action = "PROCEED_TO_DOCUMENTATION"
        else:
            action = "CLARIFY_REQUIREMENTS"
        
        return {
            "action": action,
            "session_id": agent_state.session_id,
            "negotiation_round": negotiation_round,
            "user_input": user_input,
            "response_message": f"I understand your requirements. Let me {action.lower().replace('_', ' ')} for you.",
            "updated_state": agent_state  # In real implementation, this would be modified
        }
    
    def generate_empathetic_plan_b_response(
        self,
        rejection_reason: str,
        user_emotion: str,
        cultural_context: str,
        user_concern: str
    ) -> Dict[str, str]:
        """
        Generate empathetic Plan B response considering cultural context.
        
        Args:
            rejection_reason: Reason for loan rejection
            user_emotion: User's emotional state
            cultural_context: Cultural context (e.g., TANGLISH_SPEAKING)
            user_concern: Specific user concern
            
        Returns:
            Dict[str, str]: Empathetic response message
        """
        logger.info(f"Generating empathetic Plan B response for {user_emotion} user")
        
        # Base empathetic response
        base_message = "I completely understand your situation and I'm here to help you find a solution."
        
        # Add emotion-specific empathy
        if user_emotion == "disappointed":
            base_message += " I know this rejection is disappointing, but please don't lose hope."
        elif user_emotion == "frustrated":
            base_message += " I can sense your frustration, and I want to make this right for you."
        elif user_emotion == "worried":
            base_message += " I understand your concerns about your family's financial situation."
        
        # Add cultural sensitivity for Tanglish speakers
        if cultural_context == "TANGLISH_SPEAKING":
            base_message += " Let me explain the options in a way that's clear and helpful."
        
        # Add specific concern addressing
        if "credit score" in user_concern.lower():
            base_message += " Your past financial challenges don't define your future - we have options that work with your current situation."
        elif "salary" in user_concern.lower() or "income" in user_concern.lower():
            base_message += " We have loan products specifically designed for people with your income level."
        elif "emi" in user_concern.lower() or "budget" in user_concern.lower():
            base_message += " We can definitely find EMI options that fit comfortably within your family budget."
        
        # Add hope and next steps
        base_message += " I have several alternative options that I believe will work well for you. Let me show you what's available."
        
        return {
            "message": base_message,
            "tone": "empathetic_supportive",
            "cultural_context": cultural_context,
            "emotion_addressed": user_emotion
        }

# Import required for timedelta
from datetime import timedelta

# Export main classes and functions
__all__ = [
    'SalesAgent',
    'SentimentAnalyzer',
    'EmpathyEngine',
    'PlanBEngine',
    'AlternativeOffer',
    'EmotionalState',
    'ConversationStage',
    'SalesError',
    'SentimentAnalysisError',
    'PlanBGenerationError'
]