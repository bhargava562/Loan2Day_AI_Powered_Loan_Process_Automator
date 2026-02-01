# Requirements Document

## Introduction

Loan2Day is an Agentic AI Fintech platform that replaces traditional linear chatbot scripts with a dynamic state machine architecture. The system uses a centralized Master Agent to orchestrate specialized Worker Agents, providing intelligent loan processing with empathy, security, and mathematical precision.

## Glossary

- **Master_Agent**: The centralized orchestrator that manages session state and delegates tasks to Worker Agents
- **Worker_Agent**: Specialized agents that execute specific domain tasks (Sales, Verification, Underwriting)
- **AgentState**: The centralized state object that tracks user session, loan details, and processing status
- **LQM**: Logic Quantization Module - ensures zero-hallucination mathematical calculations using decimal.Decimal
- **SGS**: Spectral-Graph Sentinel - security module for deepfake detection and file validation
- **KYC**: Know Your Customer - identity verification process
- **EMI**: Equated Monthly Installment - loan repayment calculation
- **Plan_B_Logic**: Rejection recovery system that offers alternative loan products
- **Tanglish**: Tamil + English mixed language input processing
- **Sentiment_Analysis**: Real-time emotional state detection for empathetic responses
- **SBEF**: Semantic-Bayesian Evidence Fusion - algorithm for resolving data conflicts between user input and OCR data
- **Trust_Score**: Calculated confidence metric for resolving conflicting data sources
- **Sanction_Letter**: Legally binding PDF document generated upon loan approval
- **Redis_Cache**: In-memory data store for sub-millisecond AgentState retrieval
- **Kafka_Stream**: Event streaming platform for asynchronous inter-agent communication

## Requirements

### Requirement 1: Master Agent Orchestration

**User Story:** As a system architect, I want a centralized Master Agent to coordinate all loan processing activities, so that the system maintains consistent state and proper task delegation.

#### Acceptance Criteria

1. THE Master_Agent SHALL manage all session state without executing business logic directly
2. WHEN a user request is received, THE Master_Agent SHALL route traffic based on user intent and current AgentState
3. WHEN delegating tasks, THE Master_Agent SHALL maintain coordination between Worker_Agents
4. THE Master_Agent SHALL update AgentState after each Worker_Agent completion
5. WHEN system errors occur, THE Master_Agent SHALL handle graceful degradation and recovery

### Requirement 2: Sales Agent with Empathy

**User Story:** As a loan applicant, I want the system to understand my emotional state and provide empathetic responses, so that I feel supported throughout the loan process.

#### Acceptance Criteria

1. WHEN user input is received, THE Sales_Agent SHALL perform sentiment analysis on the content
2. THE Sales_Agent SHALL maintain sentiment_history in AgentState for context awareness
3. WHEN loan rejection occurs, THE Sales_Agent SHALL trigger Plan_B_Logic for alternative offers
4. THE Sales_Agent SHALL maximize conversion rates through empathetic communication
5. WHEN negotiating terms, THE Sales_Agent SHALL adapt responses based on user sentiment

### Requirement 3: Verification Agent Security

**User Story:** As a compliance officer, I want robust identity verification and fraud detection, so that the platform maintains regulatory compliance and prevents fraudulent applications.

#### Acceptance Criteria

1. WHEN documents are uploaded, THE Verification_Agent SHALL pass all files through SGS.scan_topology()
2. THE Verification_Agent SHALL perform KYC verification and update kyc_status in AgentState
3. WHEN deepfake detection is triggered, THE SGS SHALL analyze uploaded images for authenticity
4. THE Verification_Agent SHALL extract text from documents using OCR processing
5. THE Verification_Agent SHALL calculate and store fraud_score in AgentState
6. WHEN data conflicts arise between user input and OCR data, THE Verification_Agent SHALL use SBEF to calculate Trust_Score rather than rejecting applications

### Requirement 4: Underwriting Agent Mathematics

**User Story:** As a financial analyst, I want precise mathematical calculations for loan terms, so that all currency operations are accurate and compliant with financial regulations.

#### Acceptance Criteria

1. THE Underwriting_Agent SHALL use decimal.Decimal for ALL currency calculations
2. THE LQM SHALL implement reducing balance EMI formula with zero-hallucination math
3. WHEN calculating risk assessment, THE Underwriting_Agent SHALL store results in loan_details using Decimal type
4. THE Underwriting_Agent SHALL never use float data type for any monetary values
5. WHEN EMI calculations are performed, THE system SHALL validate precision to 2 decimal places

### Requirement 5: Polyglot Voice Interface

**User Story:** As a Tamil-speaking user, I want to communicate in mixed Tamil-English (Tanglish), so that I can interact naturally without language barriers.

#### Acceptance Criteria

1. WHEN speech input is received, THE Voice_Interface SHALL process Tanglish audio inputs
2. THE Voice_Interface SHALL convert speech to structured JSON intent for Agent processing
3. WHEN Agent responses are generated, THE Voice_Interface SHALL synthesize text back to audio
4. WHEN language switching occurs, THE system SHALL adapt without losing conversation context
5. THE Voice_Interface SHALL integrate with Twilio Programmable Voice for telephony support on non-smartphone devices

### Requirement 6: API Gateway and Routing

**User Story:** As a frontend developer, I want well-defined API endpoints, so that I can integrate the React frontend with the backend services.

#### Acceptance Criteria

1. THE API_Gateway SHALL expose POST /v1/chat/message as the main entry point
2. THE API_Gateway SHALL provide POST /v1/upload/kyc endpoint that pipes directly to SGS Module
3. THE API_Gateway SHALL offer GET /v1/loan/plan-b endpoint for rejection recovery logic
4. WHEN API requests are received, THE system SHALL validate input using Pydantic models
5. THE API_Gateway SHALL return structured JSON responses with proper HTTP status codes
6. THE API_Gateway SHALL maintain response latency of less than 2.0 seconds for natural conversation flow

### Requirement 7: Agent State Management

**User Story:** As a system administrator, I want centralized state management, so that user sessions are consistent and recoverable across system restarts.

#### Acceptance Criteria

1. THE AgentState SHALL track session_id, user_id, and current_step for all sessions
2. THE AgentState SHALL store loan_details using Decimal types in a structured dictionary
3. THE AgentState SHALL maintain kyc_status with values PENDING, VERIFIED, or REJECTED
4. THE AgentState SHALL preserve fraud_score and sentiment_history throughout the session
5. WHEN state transitions occur, THE system SHALL validate current_step progression
6. THE AgentState SHALL be persisted in Redis for sub-millisecond retrieval during active sessions

### Requirement 8: Security and Compliance

**User Story:** As a security officer, I want comprehensive security measures, so that the platform protects user data and prevents fraudulent activities.

#### Acceptance Criteria

1. THE system SHALL never hardcode API keys and SHALL use environment variables
2. WHEN file uploads occur, THE SGS SHALL scan topology before processing
3. THE system SHALL use structured logging instead of print statements
4. THE system SHALL implement fail-fast validation using Pydantic V2
5. WHEN sensitive operations are performed, THE system SHALL log security events

### Requirement 9: Database and Persistence

**User Story:** As a data engineer, I want efficient data storage and retrieval, so that the system can handle high transaction volumes with optimal performance.

#### Acceptance Criteria

1. THE system SHALL use PostgreSQL for persistent data storage
2. THE Database_Layer SHALL implement proper indexing for columns used in WHERE and JOIN operations
3. THE system SHALL follow Routes -> Services -> Repositories pattern for data access
4. WHEN database operations are performed, THE system SHALL use async/await for I/O-bound tasks
5. THE system SHALL maintain data consistency across Agent state transitions
6. THE system SHALL use Apache Kafka for asynchronous inter-agent communication and high-volume logging

### Requirement 10: Error Handling and Recovery

**User Story:** As a system operator, I want robust error handling, so that the system gracefully handles failures and provides meaningful feedback.

#### Acceptance Criteria

1. WHEN validation errors occur, THE system SHALL return structured error responses
2. THE system SHALL implement Plan_B_Logic for loan rejection recovery scenarios
3. WHEN Agent failures occur, THE Master_Agent SHALL handle graceful degradation
4. THE system SHALL log all errors with sufficient context for debugging
5. WHEN system recovery is needed, THE AgentState SHALL support session restoration

### Requirement 11: Sanction Letter Generation

**User Story:** As a successful loan applicant, I want to receive a downloadable, legally binding sanction letter immediately after approval, so that I have official documentation of my loan terms.

#### Acceptance Criteria

1. THE system SHALL use Python ReportLab to dynamically generate PDF sanction letters
2. THE Sanction_Letter SHALL populate PDF with verified data from AgentState including name, loan amount, EMI, and interest rate
3. WHEN loan approval occurs, THE system SHALL generate the document in real-time immediately upon Underwriting_Agent approval
4. THE Sanction_Letter SHALL be legally binding and include all required regulatory disclosures
5. THE system SHALL provide secure download link for the generated PDF document