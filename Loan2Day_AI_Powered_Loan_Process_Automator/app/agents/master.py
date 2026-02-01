"""
Master Agent - LangGraph Orchestrator for Loan2Day

This agent serves as the central orchestrator using LangGraph state machine
to coordinate all Worker Agents (Sales, Verification, Underwriting) in the
Master-Worker pattern. It manages session state, routes user requests, and
maintains coordination without executing business logic directly.

Key Responsibilities:
- LangGraph state machine orchestration
- Route traffic based on user intent and current AgentState
- Coordinate between Worker Agents
- Handle graceful degradation and recovery
- Update AgentState after each Worker Agent completion

Architecture: Master Agent in Master-Worker pattern
Orchestration: LangGraph for state machine management
State Management: Centralized AgentState with Redis persistence

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Optional, Any, Union, TypedDict, Annotated
from decimal import Decimal
from datetime import datetime
import logging
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import core modules and models
from app.models.pydantic_models import (
    AgentState, AgentStep, KYCStatus, UserProfile, 
    LoanRequest, EMICalculation, SentimentScore
)
from app.agents.sales import SalesAgent, ConversationStage
from app.agents.verification import VerificationAgent
from app.agents.underwriting import UnderwritingAgent

# Configure logger
logger = logging.getLogger(__name__)

class UserIntent(Enum):
    """User intent classifications for routing decisions."""
    LOAN_INQUIRY = "LOAN_INQUIRY"
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    LOAN_APPLICATION = "LOAN_APPLICATION"
    STATUS_CHECK = "STATUS_CHECK"
    COMPLAINT = "COMPLAINT"
    GENERAL_QUERY = "GENERAL_QUERY"
    PLAN_B_REQUEST = "PLAN_B_REQUEST"

class MasterAgentError(Exception):
    """Base exception for master agent errors."""
    pass

class StateTransitionError(MasterAgentError):
    """Raised when invalid state transition is attempted."""
    pass

class WorkerDelegationError(MasterAgentError):
    """Raised when worker agent delegation fails."""
    pass

# LangGraph State Definition
class LoanProcessingState(TypedDict):
    """
    LangGraph state definition for loan processing workflow.
    
    This state is passed between all nodes in the LangGraph and maintains
    the complete context of the loan processing session.
    """
    # Core session data
    session_id: str
    user_id: str
    current_step: str
    
    # User interaction
    user_input: str
    user_intent: str
    conversation_history: List[Dict[str, Any]]
    
    # Agent state data
    agent_state: Dict[str, Any]  # Serialized AgentState
    
    # Processing results
    sales_result: Optional[Dict[str, Any]]
    verification_result: Optional[Dict[str, Any]]
    underwriting_result: Optional[Dict[str, Any]]
    
    # Response generation
    agent_response: str
    next_actions: List[str]
    requires_input: bool
    
    # Error handling
    error_occurred: bool
    error_message: Optional[str]
    retry_count: int

class IntentClassifier:
    """
    User intent classification engine for routing decisions.
    
    This classifier analyzes user input to determine intent and route
    requests to appropriate Worker Agents or processing flows.
    """
    
    def __init__(self):
        """Initialize intent classifier."""
        self.intent_keywords = {
            UserIntent.LOAN_INQUIRY: [
                "loan", "borrow", "credit", "finance", "money", "amount",
                "interest", "emi", "personal loan", "need money"
            ],
            UserIntent.DOCUMENT_UPLOAD: [
                "upload", "document", "file", "pan", "aadhaar", "bank statement",
                "salary slip", "kyc", "verification", "proof"
            ],
            UserIntent.LOAN_APPLICATION: [
                "apply", "application", "submit", "proceed", "start process",
                "fill form", "complete application"
            ],
            UserIntent.STATUS_CHECK: [
                "status", "progress", "update", "where is my", "how long",
                "when will", "approved", "rejected", "pending"
            ],
            UserIntent.COMPLAINT: [
                "complaint", "problem", "issue", "error", "wrong", "mistake",
                "dissatisfied", "unhappy", "frustrated"
            ],
            UserIntent.PLAN_B_REQUEST: [
                "alternative", "other options", "different loan", "plan b",
                "rejected", "denied", "other products"
            ]
        }
        logger.info("IntentClassifier initialized")
    
    def classify_intent(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> UserIntent:
        """
        Classify user intent from input text.
        
        Args:
            user_input: User's message text
            context: Optional conversation context
            
        Returns:
            UserIntent: Classified user intent
        """
        logger.info("Classifying user intent")
        
        if not user_input or not user_input.strip():
            return UserIntent.GENERAL_QUERY
        
        user_input_lower = user_input.lower()
        intent_scores = {}
        
        # Calculate scores for each intent based on keyword matches
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            intent_scores[intent] = score
        
        # Find intent with highest score
        if not any(intent_scores.values()):
            return UserIntent.GENERAL_QUERY
        
        classified_intent = max(intent_scores, key=intent_scores.get)
        
        # Context-based adjustments
        if context:
            classified_intent = self._adjust_intent_with_context(classified_intent, context)
        
        logger.info(f"Intent classified: {classified_intent.value}")
        return classified_intent
    
    def _adjust_intent_with_context(self, intent: UserIntent, context: Dict[str, Any]) -> UserIntent:
        """Adjust intent classification based on conversation context."""
        
        # If user is in KYC step and mentions documents, it's likely document upload
        current_step = context.get("current_step")
        if current_step == "KYC" and intent == UserIntent.GENERAL_QUERY:
            return UserIntent.DOCUMENT_UPLOAD
        
        # If user is in negotiation and asks questions, it's likely loan inquiry
        if current_step == "NEGOTIATION" and intent == UserIntent.GENERAL_QUERY:
            return UserIntent.LOAN_INQUIRY
        
        return intent

class MasterAgent:
    """
    Master Agent - The Central Orchestrator of Loan2Day.
    
    This agent uses LangGraph to implement a sophisticated state machine
    that coordinates all Worker Agents while maintaining centralized state
    management and ensuring proper workflow progression.
    """
    
    def __init__(self):
        """Initialize the Master Agent with LangGraph workflow."""
        self.intent_classifier = IntentClassifier()
        
        # Initialize Worker Agents
        self.sales_agent = SalesAgent()
        self.verification_agent = VerificationAgent()
        self.underwriting_agent = UnderwritingAgent()
        
        # Build LangGraph workflow
        self.workflow = self._build_langgraph_workflow()
        
        logger.info("MasterAgent initialized with LangGraph orchestration")
    
    def _build_langgraph_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow for loan processing state machine.
        
        Returns:
            StateGraph: Configured LangGraph workflow
        """
        logger.info("Building LangGraph workflow")
        
        # Create state graph
        workflow = StateGraph(LoanProcessingState)
        
        # Add nodes for each processing step
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("route_request", self._route_request_node)
        workflow.add_node("sales_processing", self._sales_processing_node)
        workflow.add_node("verification_processing", self._verification_processing_node)
        workflow.add_node("underwriting_processing", self._underwriting_processing_node)
        workflow.add_node("plan_b_processing", self._plan_b_processing_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional edges for routing
        workflow.add_conditional_edges(
            "classify_intent",
            self._should_route_request,
            {
                "route": "route_request",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "route_request",
            self._determine_processing_node,
            {
                "sales": "sales_processing",
                "verification": "verification_processing", 
                "underwriting": "underwriting_processing",
                "plan_b": "plan_b_processing",
                "error": "handle_error"
            }
        )
        
        # Add edges from processing nodes to response generation
        workflow.add_edge("sales_processing", "generate_response")
        workflow.add_edge("verification_processing", "generate_response")
        workflow.add_edge("underwriting_processing", "generate_response")
        workflow.add_edge("plan_b_processing", "generate_response")
        
        # Add conditional edges for response generation
        workflow.add_conditional_edges(
            "generate_response",
            self._should_continue_processing,
            {
                "continue": "route_request",
                "end": END,
                "error": "handle_error"
            }
        )
        
        # Error handling always ends
        workflow.add_edge("handle_error", END)
        
        # Compile workflow
        compiled_workflow = workflow.compile()
        
        logger.info("LangGraph workflow built successfully")
        return compiled_workflow
    
    # LangGraph Node Functions
    
    async def _classify_intent_node(self, state: LoanProcessingState) -> LoanProcessingState:
        """LangGraph node: Classify user intent."""
        logger.info("Processing intent classification node")
        
        try:
            # Get context from agent state
            context = {
                "current_step": state.get("current_step"),
                "session_id": state.get("session_id")
            }
            
            # Classify intent
            user_intent = self.intent_classifier.classify_intent(
                state["user_input"], context
            )
            
            # Update state
            state["user_intent"] = user_intent.value
            state["error_occurred"] = False
            
            logger.info(f"Intent classified: {user_intent.value}")
            
        except Exception as e:
            logger.error(f"Intent classification failed: {str(e)}")
            state["error_occurred"] = True
            state["error_message"] = f"Intent classification failed: {str(e)}"
        
        return state
    
    async def _route_request_node(self, state: LoanProcessingState) -> LoanProcessingState:
        """LangGraph node: Route request based on intent and current state."""
        logger.info("Processing request routing node")
        
        try:
            # Validate state transition
            current_step = state.get("current_step", "GREETING")
            user_intent = state.get("user_intent")
            
            # Determine if state transition is valid
            valid_transition = self._validate_state_transition(current_step, user_intent)
            
            if not valid_transition:
                logger.warning(f"Invalid state transition: {current_step} -> {user_intent}")
                state["error_occurred"] = True
                state["error_message"] = f"Invalid request for current state: {current_step}"
                return state
            
            # Update conversation history
            if "conversation_history" not in state:
                state["conversation_history"] = []
            
            state["conversation_history"].append({
                "timestamp": datetime.now().isoformat(),
                "user_input": state["user_input"],
                "intent": user_intent,
                "current_step": current_step
            })
            
            state["error_occurred"] = False
            
            logger.info(f"Request routed successfully: {user_intent}")
            
        except Exception as e:
            logger.error(f"Request routing failed: {str(e)}")
            state["error_occurred"] = True
            state["error_message"] = f"Request routing failed: {str(e)}"
        
        return state
    
    async def _sales_processing_node(self, state: LoanProcessingState) -> LoanProcessingState:
        """LangGraph node: Process request through Sales Agent."""
        logger.info("Processing sales agent node")
        
        try:
            # Reconstruct AgentState from serialized data
            agent_state = self._deserialize_agent_state(state["agent_state"])
            
            # Process through Sales Agent
            sales_result = await self.sales_agent.process_sales_interaction(
                agent_state,
                state["user_input"],
                interaction_type="general"
            )
            
            # Update agent state with sentiment history
            if "sentiment_analysis" in sales_result:
                sentiment_data = sales_result["sentiment_analysis"]
                sentiment_score = SentimentScore(
                    polarity=sentiment_data["polarity"],
                    subjectivity=sentiment_data["subjectivity"],
                    emotion=sentiment_data["emotion"],
                    confidence=sentiment_data["confidence"],
                    timestamp=datetime.fromisoformat(sentiment_data["timestamp"])
                )
                agent_state.sentiment_history.append(sentiment_score)
            
            # Store results
            state["sales_result"] = sales_result
            state["agent_state"] = self._serialize_agent_state(agent_state)
            state["error_occurred"] = False
            
            logger.info("Sales processing completed successfully")
            
        except Exception as e:
            logger.error(f"Sales processing failed: {str(e)}")
            state["error_occurred"] = True
            state["error_message"] = f"Sales processing failed: {str(e)}"
        
        return state
    
    async def _verification_processing_node(self, state: LoanProcessingState) -> LoanProcessingState:
        """LangGraph node: Process request through Verification Agent."""
        logger.info("Processing verification agent node")
        
        try:
            # Reconstruct AgentState from serialized data
            agent_state = self._deserialize_agent_state(state["agent_state"])
            
            # Process through Verification Agent
            verification_result = await self.verification_agent.process_verification_request(
                agent_state
            )
            
            # Update agent state with verification results
            agent_state.kyc_status = KYCStatus(verification_result["overall_status"])
            agent_state.fraud_score = verification_result["average_fraud_score"]
            
            # Store results
            state["verification_result"] = verification_result
            state["agent_state"] = self._serialize_agent_state(agent_state)
            state["error_occurred"] = False
            
            logger.info("Verification processing completed successfully")
            
        except Exception as e:
            logger.error(f"Verification processing failed: {str(e)}")
            state["error_occurred"] = True
            state["error_message"] = f"Verification processing failed: {str(e)}"
        
        return state
    
    async def _underwriting_processing_node(self, state: LoanProcessingState) -> LoanProcessingState:
        """LangGraph node: Process request through Underwriting Agent."""
        logger.info("Processing underwriting agent node")
        
        try:
            # Reconstruct AgentState from serialized data
            agent_state = self._deserialize_agent_state(state["agent_state"])
            
            # Process through Underwriting Agent
            underwriting_result = await self.underwriting_agent.process_underwriting_request(
                agent_state
            )
            
            # Update agent state with underwriting results
            if "emi_calculation" in underwriting_result:
                emi_data = underwriting_result["emi_calculation"]
                agent_state.emi_calculation = EMICalculation(
                    principal_in_cents=Decimal(emi_data["principal_in_cents"]),
                    rate_per_annum=Decimal(emi_data["rate_per_annum"]),
                    tenure_months=emi_data["tenure_months"],
                    emi_in_cents=Decimal(emi_data["emi_in_cents"]),
                    total_interest_in_cents=Decimal(emi_data["total_interest_in_cents"]),
                    total_amount_in_cents=Decimal(emi_data["total_amount_in_cents"])
                )
            
            # Update loan details with underwriting results
            if "approved_amount_in_cents" in underwriting_result:
                agent_state.loan_details["approved_amount_in_cents"] = Decimal(
                    underwriting_result["approved_amount_in_cents"]
                )
                agent_state.loan_details["approved_rate"] = Decimal(
                    underwriting_result["approved_rate"]
                )
                agent_state.loan_details["approved_emi_in_cents"] = Decimal(
                    underwriting_result["approved_emi_in_cents"]
                )
            
            # Store results
            state["underwriting_result"] = underwriting_result
            state["agent_state"] = self._serialize_agent_state(agent_state)
            state["error_occurred"] = False
            
            logger.info("Underwriting processing completed successfully")
            
        except Exception as e:
            logger.error(f"Underwriting processing failed: {str(e)}")
            state["error_occurred"] = True
            state["error_message"] = f"Underwriting processing failed: {str(e)}"
        
        return state
    
    async def _plan_b_processing_node(self, state: LoanProcessingState) -> LoanProcessingState:
        """LangGraph node: Process Plan B logic through Sales Agent."""
        logger.info("Processing Plan B node")
        
        try:
            # Reconstruct AgentState from serialized data
            agent_state = self._deserialize_agent_state(state["agent_state"])
            
            # Determine rejection reason from underwriting result or user input
            rejection_reason = "Application did not meet standard criteria"
            if state.get("underwriting_result"):
                underwriting_result = state["underwriting_result"]
                if underwriting_result.get("decision") == "REJECTED":
                    risk_assessment = underwriting_result.get("risk_assessment", {})
                    risk_factors = risk_assessment.get("risk_factors", [])
                    if risk_factors:
                        rejection_reason = "; ".join(risk_factors)
            
            # Get user sentiment if available
            user_sentiment = None
            if agent_state.sentiment_history:
                user_sentiment = agent_state.sentiment_history[-1]
            
            # Trigger Plan B logic
            plan_b_result = await self.sales_agent.trigger_plan_b_logic(
                agent_state, rejection_reason, user_sentiment
            )
            
            # Update agent state
            agent_state.plan_b_offers = plan_b_result["alternative_offers"]
            agent_state.current_step = AgentStep.PLAN_B
            
            # Store results
            state["sales_result"] = plan_b_result
            state["agent_state"] = self._serialize_agent_state(agent_state)
            state["current_step"] = AgentStep.PLAN_B.value
            state["error_occurred"] = False
            
            logger.info("Plan B processing completed successfully")
            
        except Exception as e:
            logger.error(f"Plan B processing failed: {str(e)}")
            state["error_occurred"] = True
            state["error_message"] = f"Plan B processing failed: {str(e)}"
        
        return state
    
    async def _generate_response_node(self, state: LoanProcessingState) -> LoanProcessingState:
        """LangGraph node: Generate final response to user."""
        logger.info("Processing response generation node")
        
        try:
            # Determine which result to use for response generation
            if state.get("sales_result"):
                result = state["sales_result"]
                if "empathetic_response" in result:
                    agent_response = result["empathetic_response"]
                elif "empathetic_message" in result:
                    agent_response = result["empathetic_message"]
                else:
                    agent_response = "Thank you for your interest. How can I help you today?"
                
                # Add Plan B offers if available
                if "alternative_offers" in result and result["alternative_offers"]:
                    agent_response += "\n\nI have some excellent alternative loan options for you:"
                    for i, offer in enumerate(result["alternative_offers"][:2], 1):
                        agent_response += f"\n{i}. {offer['product_name']}: {offer['amount_display']} at {offer['interest_rate']}% for {offer['tenure_months']} months (EMI: {offer['emi_display']})"
                
            elif state.get("verification_result"):
                result = state["verification_result"]
                if result["overall_status"] == "VERIFIED":
                    agent_response = "Great! Your documents have been verified successfully. Let's proceed with your loan application."
                elif result["overall_status"] == "PENDING":
                    agent_response = "We need some additional documentation to complete your verification. Please upload the required documents."
                else:
                    agent_response = "There were some issues with your document verification. Please contact our support team for assistance."
                
            elif state.get("underwriting_result"):
                result = state["underwriting_result"]
                decision = result.get("decision", "PENDING")
                
                if decision == "APPROVED":
                    approved_amount = result.get("approved_amount_in_cents", "0")
                    approved_emi = result.get("approved_emi_in_cents", "0")
                    amount_display = f"₹{float(approved_amount) / 100:,.2f}"
                    emi_display = f"₹{float(approved_emi) / 100:,.2f}"
                    
                    agent_response = f"Congratulations! Your loan of {amount_display} has been approved with an EMI of {emi_display}. Let's proceed with the final documentation."
                elif decision == "CONDITIONAL":
                    agent_response = "Your loan application requires some additional conditions to be met. Let me explain the requirements."
                else:
                    agent_response = "I understand this is disappointing, but I have some excellent alternative options for you."
                    
            else:
                agent_response = "Thank you for your message. How can I assist you with your loan requirements today?"
            
            # Generate next actions
            next_actions = self._generate_next_actions(state)
            
            # Update state
            state["agent_response"] = agent_response
            state["next_actions"] = next_actions
            state["requires_input"] = True
            state["error_occurred"] = False
            
            logger.info("Response generated successfully")
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            state["error_occurred"] = True
            state["error_message"] = f"Response generation failed: {str(e)}"
            state["agent_response"] = "I apologize, but I encountered an issue processing your request. Please try again."
        
        return state
    
    async def _handle_error_node(self, state: LoanProcessingState) -> LoanProcessingState:
        """LangGraph node: Handle errors and provide graceful degradation."""
        logger.info("Processing error handling node")
        
        error_message = state.get("error_message", "An unexpected error occurred")
        retry_count = state.get("retry_count", 0)
        
        # Increment retry count
        state["retry_count"] = retry_count + 1
        
        # Generate error response based on retry count
        if retry_count < 2:
            state["agent_response"] = (
                "I encountered a temporary issue processing your request. "
                "Let me try a different approach. Could you please rephrase your request?"
            )
            state["requires_input"] = True
        else:
            state["agent_response"] = (
                "I'm experiencing technical difficulties at the moment. "
                "Please contact our support team for immediate assistance, "
                "or try again in a few minutes."
            )
            state["requires_input"] = False
        
        state["next_actions"] = ["Contact support if issue persists"]
        
        logger.warning(f"Error handled: {error_message} (Retry: {retry_count})")
        
        return state
    
    # LangGraph Conditional Functions
    
    def _should_route_request(self, state: LoanProcessingState) -> str:
        """Determine if request should be routed or handled as error."""
        if state.get("error_occurred"):
            return "error"
        return "route"
    
    def _determine_processing_node(self, state: LoanProcessingState) -> str:
        """Determine which processing node to route to based on intent and state."""
        if state.get("error_occurred"):
            return "error"
        
        user_intent = state.get("user_intent")
        current_step = state.get("current_step", "GREETING")
        
        # Route based on intent and current step
        if user_intent == UserIntent.PLAN_B_REQUEST.value:
            return "plan_b"
        elif current_step == "GREETING" or user_intent == UserIntent.LOAN_INQUIRY.value:
            return "sales"
        elif current_step == "KYC" or user_intent == UserIntent.DOCUMENT_UPLOAD.value:
            return "verification"
        elif current_step == "NEGOTIATION" or user_intent == UserIntent.LOAN_APPLICATION.value:
            return "underwriting"
        else:
            return "sales"  # Default to sales for general queries
    
    def _should_continue_processing(self, state: LoanProcessingState) -> str:
        """Determine if processing should continue or end."""
        if state.get("error_occurred"):
            return "error"
        
        # Check if state transition is needed
        current_step = state.get("current_step")
        
        # Continue processing if state needs to advance
        if self._should_advance_state(state):
            return "continue"
        
        return "end"
    
    # Helper Functions
    
    def _validate_state_transition(self, current_step: str, user_intent: str) -> bool:
        """Validate if state transition is allowed."""
        
        # Define valid transitions
        valid_transitions = {
            "GREETING": [
                UserIntent.LOAN_INQUIRY.value,
                UserIntent.GENERAL_QUERY.value,
                UserIntent.STATUS_CHECK.value
            ],
            "KYC": [
                UserIntent.DOCUMENT_UPLOAD.value,
                UserIntent.STATUS_CHECK.value,
                UserIntent.GENERAL_QUERY.value
            ],
            "NEGOTIATION": [
                UserIntent.LOAN_APPLICATION.value,
                UserIntent.LOAN_INQUIRY.value,
                UserIntent.STATUS_CHECK.value,
                UserIntent.GENERAL_QUERY.value
            ],
            "SANCTION": [
                UserIntent.STATUS_CHECK.value,
                UserIntent.GENERAL_QUERY.value
            ],
            "PLAN_B": [
                UserIntent.PLAN_B_REQUEST.value,
                UserIntent.LOAN_INQUIRY.value,
                UserIntent.STATUS_CHECK.value,
                UserIntent.GENERAL_QUERY.value
            ]
        }
        
        allowed_intents = valid_transitions.get(current_step, [])
        return user_intent in allowed_intents or user_intent == UserIntent.COMPLAINT.value
    
    def _should_advance_state(self, state: LoanProcessingState) -> bool:
        """Determine if agent state should advance to next step."""
        
        current_step = state.get("current_step")
        
        # Check if conditions are met for state advancement
        if current_step == "GREETING" and state.get("sales_result"):
            return True
        elif current_step == "KYC" and state.get("verification_result"):
            verification_result = state["verification_result"]
            return verification_result.get("overall_status") == "VERIFIED"
        elif current_step == "NEGOTIATION" and state.get("underwriting_result"):
            underwriting_result = state["underwriting_result"]
            return underwriting_result.get("decision") in ["APPROVED", "REJECTED"]
        
        return False
    
    def _generate_next_actions(self, state: LoanProcessingState) -> List[str]:
        """Generate next actions based on current state and results."""
        
        current_step = state.get("current_step", "GREETING")
        next_actions = []
        
        if current_step == "GREETING":
            next_actions = [
                "Provide loan amount and tenure requirements",
                "Share your monthly income details",
                "Ask any questions about our loan products"
            ]
        elif current_step == "KYC":
            next_actions = [
                "Upload PAN card document",
                "Upload Aadhaar card document", 
                "Upload bank statement (last 3 months)",
                "Upload salary slips (last 3 months)"
            ]
        elif current_step == "NEGOTIATION":
            next_actions = [
                "Review the loan terms presented",
                "Ask questions about interest rates or EMI",
                "Proceed with the loan application"
            ]
        elif current_step == "SANCTION":
            next_actions = [
                "Download your sanction letter",
                "Complete final documentation",
                "Schedule loan disbursement"
            ]
        elif current_step == "PLAN_B":
            next_actions = [
                "Review alternative loan options",
                "Select your preferred alternative",
                "Provide additional documentation if required"
            ]
        
        return next_actions
    
    def _serialize_agent_state(self, agent_state: AgentState) -> Dict[str, Any]:
        """
        Serialize AgentState for LangGraph state storage with proper Decimal handling.
        
        Args:
            agent_state: AgentState object to serialize
            
        Returns:
            Dict[str, Any]: Serialized state dictionary
        """
        logger.debug(f"Serializing AgentState for session: {agent_state.session_id}")
        
        # Convert AgentState to dictionary with proper Decimal handling
        state_dict = {
            "session_id": agent_state.session_id,
            "user_id": agent_state.user_id,
            "current_step": agent_state.current_step.value if hasattr(agent_state.current_step, 'value') else agent_state.current_step,
            "loan_details": {k: str(v) for k, v in agent_state.loan_details.items()},
            "kyc_status": agent_state.kyc_status.value if hasattr(agent_state.kyc_status, 'value') else agent_state.kyc_status,
            "fraud_score": agent_state.fraud_score,
            "trust_score": agent_state.trust_score,
            "created_at": agent_state.created_at.isoformat(),
            "updated_at": agent_state.updated_at.isoformat(),
            "plan_b_offers": agent_state.plan_b_offers,
            "conversation_context": agent_state.conversation_context
        }
        
        # Serialize user profile with Decimal conversion
        if agent_state.user_profile:
            profile_dict = agent_state.user_profile.dict()
            # Convert Decimal fields to strings for JSON serialization
            profile_dict["income_in_cents"] = str(profile_dict["income_in_cents"])
            state_dict["user_profile"] = profile_dict
        
        # Serialize loan request with Decimal conversion
        if agent_state.loan_request:
            request_dict = agent_state.loan_request.dict()
            # Convert Decimal fields to strings
            request_dict["amount_in_cents"] = str(request_dict["amount_in_cents"])
            if request_dict.get("requested_rate"):
                request_dict["requested_rate"] = str(request_dict["requested_rate"])
            if request_dict.get("monthly_income_in_cents"):
                request_dict["monthly_income_in_cents"] = str(request_dict["monthly_income_in_cents"])
            if request_dict.get("existing_emi_in_cents"):
                request_dict["existing_emi_in_cents"] = str(request_dict["existing_emi_in_cents"])
            state_dict["loan_request"] = request_dict
        
        # Serialize EMI calculation with Decimal conversion
        if agent_state.emi_calculation:
            emi_dict = agent_state.emi_calculation.dict()
            # Convert all Decimal fields to strings
            for field in ["principal_in_cents", "rate_per_annum", "emi_in_cents", 
                         "total_interest_in_cents", "total_amount_in_cents"]:
                if field in emi_dict:
                    emi_dict[field] = str(emi_dict[field])
            state_dict["emi_calculation"] = emi_dict
        
        # Serialize KYC documents
        if agent_state.kyc_documents:
            kyc_docs = []
            for doc in agent_state.kyc_documents:
                doc_dict = doc.dict()
                # Handle datetime serialization
                if "uploaded_at" in doc_dict and doc_dict["uploaded_at"]:
                    doc_dict["uploaded_at"] = doc_dict["uploaded_at"].isoformat()
                if "verified_at" in doc_dict and doc_dict["verified_at"]:
                    doc_dict["verified_at"] = doc_dict["verified_at"].isoformat()
                kyc_docs.append(doc_dict)
            state_dict["kyc_documents"] = kyc_docs
        
        # Serialize sentiment history
        if agent_state.sentiment_history:
            sentiment_list = []
            for sentiment in agent_state.sentiment_history:
                sentiment_dict = sentiment.dict()
                # Convert datetime to string
                sentiment_dict["timestamp"] = sentiment_dict["timestamp"].isoformat()
                sentiment_list.append(sentiment_dict)
            state_dict["sentiment_history"] = sentiment_list
        
        logger.debug(f"AgentState serialized successfully for session: {agent_state.session_id}")
        return state_dict
    
    def _deserialize_agent_state(self, state_dict: Dict[str, Any]) -> AgentState:
        """
        Deserialize AgentState from LangGraph state storage with proper Decimal restoration.
        
        Args:
            state_dict: Serialized state dictionary
            
        Returns:
            AgentState: Reconstructed AgentState object
        """
        logger.debug(f"Deserializing AgentState for session: {state_dict.get('session_id')}")
        
        # Convert string values back to Decimal for loan_details
        loan_details = {}
        for k, v in state_dict.get("loan_details", {}).items():
            loan_details[k] = Decimal(str(v))
        
        # Create base AgentState object
        agent_state = AgentState(
            session_id=state_dict["session_id"],
            user_id=state_dict["user_id"],
            current_step=AgentStep(state_dict["current_step"]),
            loan_details=loan_details,
            kyc_status=KYCStatus(state_dict["kyc_status"]),
            fraud_score=state_dict["fraud_score"],
            trust_score=state_dict.get("trust_score"),
            created_at=datetime.fromisoformat(state_dict["created_at"]),
            updated_at=datetime.fromisoformat(state_dict["updated_at"]),
            plan_b_offers=state_dict.get("plan_b_offers", []),
            conversation_context=state_dict.get("conversation_context", {})
        )
        
        # Deserialize user profile with Decimal restoration
        if state_dict.get("user_profile"):
            profile_dict = state_dict["user_profile"]
            # Convert Decimal fields back
            profile_dict["income_in_cents"] = Decimal(str(profile_dict["income_in_cents"]))
            agent_state.user_profile = UserProfile(**profile_dict)
        
        # Deserialize loan request with Decimal restoration
        if state_dict.get("loan_request"):
            request_dict = state_dict["loan_request"]
            # Convert Decimal fields back
            request_dict["amount_in_cents"] = Decimal(str(request_dict["amount_in_cents"]))
            if request_dict.get("requested_rate"):
                request_dict["requested_rate"] = Decimal(str(request_dict["requested_rate"]))
            if request_dict.get("monthly_income_in_cents"):
                request_dict["monthly_income_in_cents"] = Decimal(str(request_dict["monthly_income_in_cents"]))
            if request_dict.get("existing_emi_in_cents"):
                request_dict["existing_emi_in_cents"] = Decimal(str(request_dict["existing_emi_in_cents"]))
            agent_state.loan_request = LoanRequest(**request_dict)
        
        # Deserialize EMI calculation with Decimal restoration
        if state_dict.get("emi_calculation"):
            emi_dict = state_dict["emi_calculation"]
            # Convert Decimal fields back
            for field in ["principal_in_cents", "rate_per_annum", "emi_in_cents", 
                         "total_interest_in_cents", "total_amount_in_cents"]:
                if field in emi_dict:
                    emi_dict[field] = Decimal(str(emi_dict[field]))
            agent_state.emi_calculation = EMICalculation(**emi_dict)
        
        # Deserialize KYC documents
        if state_dict.get("kyc_documents"):
            kyc_docs = []
            for doc_dict in state_dict["kyc_documents"]:
                # Handle datetime deserialization
                if "uploaded_at" in doc_dict and doc_dict["uploaded_at"]:
                    doc_dict["uploaded_at"] = datetime.fromisoformat(doc_dict["uploaded_at"])
                if "verified_at" in doc_dict and doc_dict["verified_at"]:
                    doc_dict["verified_at"] = datetime.fromisoformat(doc_dict["verified_at"])
                kyc_docs.append(KYCDocument(**doc_dict))
            agent_state.kyc_documents = kyc_docs
        
        # Deserialize sentiment history
        if state_dict.get("sentiment_history"):
            sentiment_list = []
            for sentiment_dict in state_dict["sentiment_history"]:
                # Convert datetime back
                sentiment_dict["timestamp"] = datetime.fromisoformat(sentiment_dict["timestamp"])
                sentiment_list.append(SentimentScore(**sentiment_dict))
            agent_state.sentiment_history = sentiment_list
        
        logger.debug(f"AgentState deserialized successfully for session: {agent_state.session_id}")
        return agent_state
    
    # Main Processing Functions
    
    async def process_user_request(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        agent_state: AgentState
    ) -> Dict[str, Any]:
        """
        Process user request through LangGraph orchestration.
        
        This is the main entry point for the Master Agent. It processes
        user requests through the complete LangGraph workflow and returns
        the orchestrated response.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            user_input: User's message or request
            agent_state: Current agent state
            
        Returns:
            Dict[str, Any]: Orchestrated response with updated state
            
        Raises:
            MasterAgentError: If orchestration fails
        """
        logger.info(f"Processing user request - Session: {session_id}, User: {user_id}")
        
        try:
            # Initialize LangGraph state
            initial_state = LoanProcessingState(
                session_id=session_id,
                user_id=user_id,
                current_step=agent_state.current_step.value if hasattr(agent_state.current_step, 'value') else agent_state.current_step,
                user_input=user_input,
                user_intent="",
                conversation_history=[],
                agent_state=self._serialize_agent_state(agent_state),
                sales_result=None,
                verification_result=None,
                underwriting_result=None,
                agent_response="",
                next_actions=[],
                requires_input=True,
                error_occurred=False,
                error_message=None,
                retry_count=0
            )
            
            # Execute LangGraph workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Prepare orchestrated response
            orchestrated_response = {
                "session_id": session_id,
                "agent_response": final_state["agent_response"],
                "current_step": final_state.get("current_step", agent_state.current_step.value if hasattr(agent_state.current_step, 'value') else agent_state.current_step),
                "next_actions": final_state["next_actions"],
                "requires_input": final_state["requires_input"],
                "user_intent": final_state.get("user_intent"),
                "processing_results": {
                    "sales_result": final_state.get("sales_result"),
                    "verification_result": final_state.get("verification_result"),
                    "underwriting_result": final_state.get("underwriting_result")
                },
                "conversation_history": final_state.get("conversation_history", []),
                "error_occurred": final_state["error_occurred"],
                "error_message": final_state.get("error_message"),
                "updated_agent_state": final_state["agent_state"],
                "processing_timestamp": datetime.now().isoformat()
            }
            
            logger.info(
                f"User request processed successfully - Session: {session_id}, "
                f"Intent: {final_state.get('user_intent')}, "
                f"Step: {final_state.get('current_step')}"
            )
            
            return orchestrated_response
            
        except Exception as e:
            logger.error(f"Master agent orchestration failed: {str(e)}")
            raise MasterAgentError(f"Orchestration failed: {str(e)}")

# Export main classes and functions
__all__ = [
    'MasterAgent',
    'LoanProcessingState',
    'IntentClassifier',
    'UserIntent',
    'MasterAgentError',
    'StateTransitionError',
    'WorkerDelegationError'
]