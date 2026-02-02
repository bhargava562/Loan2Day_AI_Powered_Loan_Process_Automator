# Loan2Day Agentic AI Fintech Platform

**Master-Worker Agent Architecture for Intelligent Loan Processing**

Loan2Day is an advanced Agentic AI Fintech platform that replaces traditional linear chatbot scripts with a dynamic state machine architecture. The system uses a centralized Master Agent to orchestrate specialized Worker Agents, providing intelligent loan processing with empathy, security, and mathematical precision.

## 🏗️ Architecture Overview

### Hub-and-Spoke Agentic Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │  React Frontend │    │  Voice Interface (Twilio)      │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   API Gateway (FastAPI)                    │
│                    < 2s Response Latency                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Master-Worker Agent Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Master    │  │   Sales     │  │   Verification      │ │
│  │   Agent     │  │   Agent     │  │   Agent             │ │
│  │ (LangGraph) │  │ (Sentiment  │  │ (KYC + SGS + SBEF)  │ │
│  │             │  │ + Plan B)   │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────────────────────────────┐   │
│  │Underwriting │  │     Sanction Generator              │   │
│  │   Agent     │  │     (ReportLab PDF)                 │   │
│  │(LQM + EMI)  │  │                                     │   │
│  └─────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Redis     │  │ PostgreSQL  │  │   Kafka Streams     │ │
│  │  (Cache)    │  │(Persistent) │  │ (Async Events)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Security Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │SGS Module   │  │ LQM Module  │  │  SBEF Algorithm     │ │
│  │(Deepfake    │  │(Decimal     │  │(Conflict Resolution)│ │
│  │Detection)   │  │Mathematics) │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **Master Agent**: LangGraph orchestrator managing session state and Worker Agent coordination
- **Sales Agent**: Empathetic communication with sentiment analysis and Plan B logic
- **Verification Agent**: KYC processing with SGS security scanning and SBEF conflict resolution
- **Underwriting Agent**: Risk assessment and EMI calculations using LQM mathematics
- **Sanction Generator**: Real-time PDF generation for approved loans

## 🚀 Quick Start

### Prerequisites

- **Python 3.11 LTS** (Required for long-term stability)
- **Redis** (for session caching)
- **PostgreSQL** (for persistent data)
- **Kafka** (optional, for async communication)

### Environment Setup

1. **Create Python 3.11 Virtual Environment**
   ```bash
   # Ensure Python 3.11 is installed
   python3.11 --version
   
   # Create virtual environment
   python3.11 -m venv venv
   
   # Activate virtual environment
   # Linux/Mac:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   # Upgrade pip inside venv (NOT globally)
   pip install --upgrade pip
   
   # Install all dependencies
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your configuration
   # Required variables:
   # - DATABASE_URL=postgresql+asyncpg://user:password@localhost/loan2day
   # - REDIS_URL=redis://localhost:6379/0
   # - KAFKA_BOOTSTRAP_SERVERS=localhost:9092 (optional)
   ```

4. **Frontend Setup**
   ```bash
   # Initialize React frontend
   cd frontend
   npm install
   npm run dev
   ```

### Running the Application

1. **Start Backend Services**
   ```bash
   # Start Redis (required)
   redis-server
   
   # Start PostgreSQL (required)
   # Follow your system's PostgreSQL installation guide
   
   # Start Kafka (optional)
   # Follow Kafka installation guide for your system
   ```

2. **Start Loan2Day Platform**
   ```bash
   # From project root with activated venv
   python -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Start Frontend (separate terminal)**
   ```bash
   cd frontend
   npm run dev
   ```

4. **Access the Application**
   - **API Documentation**: http://localhost:8000/docs
   - **Frontend**: http://localhost:5173
   - **Health Check**: http://localhost:8000/health

## 📚 API Documentation

### Core Endpoints

#### Chat API
- **POST** `/v1/chat/message` - Main conversation endpoint
- **GET** `/v1/chat/session/{session_id}` - Get session information
- **DELETE** `/v1/chat/session/{session_id}` - Delete session

#### Upload API
- **POST** `/v1/upload/kyc` - Upload KYC documents with SGS scanning
- **GET** `/v1/upload/status/{upload_id}` - Check upload status

#### Plan B API
- **GET** `/v1/loan/plan-b` - Get alternative loan offers
- **POST** `/v1/loan/plan-b/select` - Select alternative offer

#### Documents API
- **POST** `/v1/documents/generate-sanction` - Generate sanction letter PDF
- **GET** `/v1/documents/download/{token}` - Secure document download

### Example API Usage

#### Start a Conversation
```bash
curl -X POST "http://localhost:8000/v1/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "I need a personal loan of ₹5,00,000",
    "message_type": "text"
  }'
```

#### Upload KYC Document
```bash
curl -X POST "http://localhost:8000/v1/upload/kyc" \
  -F "session_id=sess_20240301_123456_user123" \
  -F "user_id=user123" \
  -F "document_type=PAN_CARD" \
  -F "file=@pan_card.jpg"
```

#### Get Plan B Offers
```bash
curl -X GET "http://localhost:8000/v1/loan/plan-b?session_id=sess_123&user_id=user123"
```

## 🔒 Security Features

### SGS (Spectral-Graph Sentinel)
- **Deepfake Detection**: Advanced AI-based authenticity verification
- **File Topology Scanning**: Security analysis of all uploaded documents
- **Threat Level Assessment**: Real-time risk scoring

### LQM (Logic Quantization Module)
- **Zero-Hallucination Mathematics**: All currency calculations use `decimal.Decimal`
- **Precision Validation**: Exactly 2 decimal places for all monetary values
- **EMI Calculation**: Reducing balance formula with mathematical correctness

### SBEF (Semantic-Bayesian Evidence Fusion)
- **Conflict Resolution**: Intelligent handling of data discrepancies
- **Trust Score Calculation**: Confidence metrics for data sources
- **Evidence Fusion**: Combining user input with OCR data

## 🧠 Agent Architecture

### Master Agent (The Orchestrator)
- **LangGraph State Machine**: Dynamic workflow orchestration
- **Intent Classification**: Intelligent routing based on user input
- **State Management**: Centralized session state with Redis persistence
- **Error Handling**: Graceful degradation and recovery

### Sales Agent (The Negotiator)
- **Sentiment Analysis**: Real-time emotional state detection
- **Empathetic Responses**: Context-aware communication
- **Plan B Logic**: Rejection recovery with alternative offers
- **Conversion Optimization**: Maximizing approval rates

### Verification Agent (The Detective)
- **KYC Processing**: Document verification and validation
- **OCR Integration**: Text extraction from uploaded documents
- **Fraud Detection**: Risk assessment and scoring
- **SGS Integration**: Mandatory security scanning

### Underwriting Agent (The Accountant)
- **Risk Assessment**: Credit scoring and eligibility analysis
- **EMI Calculations**: Precise mathematical computations
- **LQM Integration**: Zero-hallucination mathematics
- **Decision Engine**: Approval/rejection logic

## 🗄️ Data Models

### AgentState Schema
```python
{
  "session_id": "sess_20240301_123456_user123",
  "user_id": "user123",
  "current_step": "NEGOTIATION",
  "loan_details": {
    "amount_in_cents": "50000000",  # ₹5,00,000 in cents
    "tenure_months": 36,
    "purpose": "PERSONAL"
  },
  "kyc_status": "VERIFIED",
  "fraud_score": 0.15,
  "sentiment_history": [...],
  "emi_calculation": {...},
  "plan_b_offers": [...]
}
```

### Database Schema
- **users**: User profiles and income information
- **loan_applications**: Loan requests and decisions
- **agent_sessions**: Session state and conversation history
- **kyc_documents**: Document verification records

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run property-based tests
pytest tests/property/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=app tests/
```

### Property-Based Testing
The platform uses Hypothesis for property-based testing to verify:
- Mathematical correctness (EMI calculations)
- State machine transitions
- Security processing (SGS scanning)
- Agent coordination
- Data persistence

### Test Categories
- **Unit Tests**: Specific examples and edge cases
- **Property Tests**: Universal correctness properties
- **Integration Tests**: End-to-end workflows
- **Performance Tests**: Response time validation

## 🚀 Deployment

### Docker Deployment
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Configuration
1. **Environment Variables**: Set production values in `.env`
2. **Database**: Configure PostgreSQL with proper indexing
3. **Redis**: Set up Redis cluster for high availability
4. **Kafka**: Configure Kafka cluster for async processing
5. **Security**: Enable HTTPS and proper authentication

### Health Monitoring
- **Health Endpoint**: `/health` - Basic service status
- **System Health**: `/health/system` - Detailed diagnostics
- **Security Audit**: `/health/security` - Security compliance check

## 📁 Project Structure

```
loan2day/
├── app/
│   ├── agents/                 # Master and Worker Agents
│   │   ├── master.py          # LangGraph Master Agent
│   │   ├── sales.py           # Sales Agent with Plan B
│   │   ├── verification.py    # KYC & SGS Agent
│   │   └── underwriting.py    # EMI & Risk Agent
│   ├── api/
│   │   ├── main.py            # FastAPI application
│   │   └── routes/            # API route handlers
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   ├── lqm.py             # Logic Quantization Module
│   │   ├── sgs.py             # Spectral-Graph Sentinel
│   │   ├── dependencies.py    # Dependency injection
│   │   └── error_handling.py  # Error management
│   ├── models/
│   │   ├── pydantic_models.py # API models
│   │   └── db_models.py       # Database models
│   └── services/
│       ├── session_service.py # Redis session management
│       ├── pdf_service.py     # PDF generation
│       └── kafka_service.py   # Async messaging
├── frontend/                   # React TypeScript frontend
├── tests/                      # Comprehensive test suite
├── .kiro/specs/               # Requirements and design docs
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Container orchestration
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables
```bash
# Application
DEBUG=false
VERSION=1.0.0
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost/loan2day
DB_POOL_SIZE=10

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_SESSION_TTL_SECONDS=3600

# Kafka (Optional)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_GROUP_ID=loan2day-agents

# External APIs (Managed by SecretManager)
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token

# LQM Configuration
DEFAULT_INTEREST_RATE=12.50
MAX_LOAN_AMOUNT_IN_CENTS=10000000
MIN_LOAN_AMOUNT_IN_CENTS=5000000

# Security
SGS_SECURITY_THRESHOLD=0.85
MAX_FILE_SIZE_MB=10
```

### Performance Tuning
- **API Response Timeout**: 2.0 seconds for natural conversation
- **Redis Session TTL**: 3600 seconds (1 hour)
- **Database Pool Size**: 10 connections
- **File Upload Limit**: 10MB maximum

## 📖 Documentation References

### Comprehensive Specifications
For detailed requirements, design decisions, and implementation guidelines, refer to:

- **Requirements Document**: `.kiro/specs/loan2day/requirements.md`
- **Design Document**: `.kiro/specs/loan2day/design.md`
- **Implementation Tasks**: `.kiro/specs/loan2day/tasks.md`

**Note**: The `.kiro` directory contains the complete specification and should NOT be added to `.gitignore` as it provides essential project documentation.

### Key Design Principles

1. **Zero-Hallucination Mathematics**: All monetary calculations use `decimal.Decimal`
2. **Security-First Architecture**: SGS scanning mandatory for all file uploads
3. **Master-Worker Pattern**: Centralized orchestration with specialized agents
4. **Empathetic AI**: Sentiment-aware responses and rejection recovery
5. **Sub-millisecond Performance**: Redis caching for session state

## 🤝 Contributing

### Development Workflow
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow coding standards**: Use type hints, docstrings, and LQM principles
4. **Write tests**: Both unit and property-based tests required
5. **Run test suite**: `pytest` must pass completely
6. **Submit pull request**: Include detailed description and test results

### Coding Standards
- **Python 3.11 LTS**: Required for compatibility
- **Type Safety**: Use `typing` annotations for all functions
- **LQM Standard**: Never use `float` for currency, always `decimal.Decimal`
- **Security**: All file uploads must pass SGS scanning
- **Documentation**: Google-style docstrings required

### Architecture Guidelines
- **Routes → Services → Repositories**: Follow layered architecture
- **Async First**: Use `async def` for I/O-bound operations
- **Dependency Injection**: Use the centralized container
- **Error Handling**: Structured responses with proper HTTP codes

## 📄 License

This project is proprietary software developed for Loan2Day Fintech Platform.

## 🆘 Support

### Getting Help
- **Documentation**: Check `.kiro/specs/` directory first
- **API Issues**: Use `/health` endpoints for diagnostics
- **Performance**: Monitor response times and Redis health
- **Security**: Review SGS scan results and audit logs

### Common Issues

#### Redis Connection Failed
```bash
# Check Redis status
redis-cli ping

# Restart Redis
sudo systemctl restart redis
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U loan2day -d loan2day
```

#### Agent Orchestration Failures
```bash
# Check dependency container health
curl http://localhost:8000/health/system

# Review application logs
tail -f loan2day.log
```

---

**Loan2Day Platform** - Intelligent Loan Processing with Agentic AI Architecture

*Built with Python 3.11 LTS, FastAPI, LangGraph, React, and Redis*